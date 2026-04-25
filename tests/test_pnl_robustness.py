"""Tests for PnL computation and robustness checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pnl import (
    compute_economic_impact,
    compute_cumulative_pnl,
    compute_regime_pnl,
)
from src.robustness import (
    compute_tc_sensitivity,
    compute_subperiod_performance,
    compute_strategy_ranking,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_daily_df(n: int = 100, start: str = "2023-01-02") -> pd.DataFrame:
    """Synthetic daily_df with all required columns."""
    idx = pd.bdate_range(start, periods=n, freq="B")
    rng = np.random.default_rng(0)
    eq_ret = rng.normal(0.0006, 0.01, n)
    def_ret = rng.normal(0.0001, 0.003, n)
    w_def   = np.where(np.arange(n) % 3 == 0, 1.0, 0.5)
    w_bull  = np.clip(rng.uniform(0.4, 1.0, n), 0, 1)
    w_mom   = np.clip(rng.uniform(0.25, 1.0, n), 0.25, 1)
    w_vol   = np.clip(rng.uniform(0.25, 1.0, n), 0.25, 1)

    def _cost(w, bps=5):
        return np.abs(np.diff(np.concatenate([[0], w]))) * bps / 10_000

    tc_def  = _cost(w_def)
    tc_bull = _cost(w_bull)
    tc_mom  = _cost(w_mom)
    tc_vol  = _cost(w_vol)

    return pd.DataFrame({
        "benchmark_ret":              eq_ret,
        "overlay_ret":                w_def * eq_ret + (1 - w_def) * def_ret - tc_def,
        "equity_weight":              w_def,
        "transaction_cost":           tc_def,
        "bull_aware_ret":             w_bull * eq_ret + (1 - w_bull) * def_ret - tc_bull,
        "bull_aware_equity_weight":   w_bull,
        "bull_aware_transaction_cost":tc_bull,
        "regime_momentum_ret":        w_mom * eq_ret + (1 - w_mom) * def_ret - tc_mom,
        "regime_momentum_equity_weight": w_mom,
        "regime_momentum_transaction_cost": tc_mom,
        "vol_managed_ret":            w_vol * eq_ret + (1 - w_vol) * def_ret - tc_vol,
        "vol_managed_equity_weight":  w_vol,
        "vol_managed_transaction_cost": tc_vol,
        "regime_label":               np.tile(["risk_on", "transition", "stress"], n)[:n],
    }, index=idx)


def _make_metrics_comparison(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal metrics_comparison DataFrame from a daily_df."""
    from src.metrics import compute_all_metrics
    strats = {
        "benchmark":          "benchmark_ret",
        "defensive_overlay":  "overlay_ret",
        "bull_aware_overlay": "bull_aware_ret",
        "regime_momentum":    "regime_momentum_ret",
        "vol_managed":        "vol_managed_ret",
    }
    rows_dict: dict = {}
    for strat, col in strats.items():
        m = compute_all_metrics(daily_df[col])
        for k, v in m.items():
            if k == "label":
                continue
            rows_dict.setdefault(k, {})[strat] = v
    rows_dict["avg_equity_exposure"] = {
        "benchmark": 1.0,
        "defensive_overlay":  float(daily_df["equity_weight"].mean()),
        "bull_aware_overlay": float(daily_df["bull_aware_equity_weight"].mean()),
        "regime_momentum":    float(daily_df["regime_momentum_equity_weight"].mean()),
        "vol_managed":        float(daily_df["vol_managed_equity_weight"].mean()),
    }
    rows_dict["total_transaction_cost"] = {
        "benchmark": 0.0,
        "defensive_overlay":  float(daily_df["transaction_cost"].sum()),
        "bull_aware_overlay": float(daily_df["bull_aware_transaction_cost"].sum()),
        "regime_momentum":    float(daily_df["regime_momentum_transaction_cost"].sum()),
        "vol_managed":        float(daily_df["vol_managed_transaction_cost"].sum()),
    }
    records = [{"metric": k, **v} for k, v in rows_dict.items()]
    return pd.DataFrame(records)


# ── PnL tests ──────────────────────────────────────────────────────────────────

class TestPnLComputation:
    def test_final_value_equals_100k_times_cumulative_return(self):
        """final_value = 100000 * (1 + total_return)."""
        daily_df = _make_daily_df()
        mc = _make_metrics_comparison(daily_df)
        eco = compute_economic_impact(daily_df, mc, initial_capital=100_000.0)

        for _, row in eco.iterrows():
            expected = 100_000.0 * (1 + row["total_return_pct"] / 100)
            assert abs(row["final_value"] - expected) < 1.0, (
                f"{row['strategy']}: final_value={row['final_value']:.2f}, "
                f"expected={expected:.2f}"
            )

    def test_simulated_pnl_equals_final_value_minus_100k(self):
        """simulated_pnl = final_value - 100000."""
        daily_df = _make_daily_df()
        mc = _make_metrics_comparison(daily_df)
        eco = compute_economic_impact(daily_df, mc, initial_capital=100_000.0)

        for _, row in eco.iterrows():
            expected = row["final_value"] - 100_000.0
            assert abs(row["simulated_pnl"] - expected) < 0.02, (
                f"{row['strategy']}: pnl={row['simulated_pnl']:.2f}, "
                f"expected={expected:.2f}"
            )

    def test_economic_impact_has_all_strategies(self):
        """Economic impact table must contain all 5 strategies."""
        daily_df = _make_daily_df()
        mc = _make_metrics_comparison(daily_df)
        eco = compute_economic_impact(daily_df, mc)
        strategies = set(eco["strategy"].tolist())
        expected = {"benchmark", "defensive_overlay", "bull_aware_overlay",
                    "regime_momentum", "vol_managed"}
        assert expected.issubset(strategies), (
            f"Missing strategies: {expected - strategies}"
        )

    def test_benchmark_has_zero_transaction_cost(self):
        """Benchmark always has 0% simulated transaction cost."""
        daily_df = _make_daily_df()
        mc = _make_metrics_comparison(daily_df)
        eco = compute_economic_impact(daily_df, mc)
        bench = eco[eco["strategy"] == "benchmark"]
        assert len(bench) > 0
        assert float(bench["total_transaction_cost_pct"].iloc[0]) == pytest.approx(0.0, abs=1e-9)

    def test_cumulative_pnl_starts_near_zero(self):
        """First row of cumulative PnL should be close to zero (first-day return * 100k)."""
        daily_df = _make_daily_df()
        cum = compute_cumulative_pnl(daily_df)
        assert len(cum) > 0
        # First row should be the first day's return (not 0, but small relative to 100k)
        for col in cum.columns:
            assert abs(cum[col].iloc[0]) < 5_000, (
                f"{col}: first PnL={cum[col].iloc[0]:.2f} seems too large"
            )

    def test_regime_pnl_covers_all_regimes(self):
        """Regime PnL must include risk_on, transition, stress rows."""
        daily_df = _make_daily_df()
        rpnl = compute_regime_pnl(daily_df)
        assert set(rpnl["regime"].unique()) == {"risk_on", "transition", "stress"}


# ── Robustness tests ──────────────────────────────────────────────────────────

class TestTransactionCostSensitivity:
    def test_output_shape(self):
        """TC sensitivity has a row per strategy × TC level combination."""
        daily_df = _make_daily_df()
        tc_df = compute_tc_sensitivity(daily_df, tc_levels_bps=[0, 5, 10])
        n_strats = len([c for c in daily_df.columns if c.endswith("_ret") and
                        any(s in c for s in ["benchmark", "overlay", "aware", "momentum", "managed"])])
        # 5 strategies × 3 TC levels = 15 rows
        assert len(tc_df) == 5 * 3, f"Expected 15 rows, got {len(tc_df)}"

    def test_zero_tc_gives_higher_or_equal_sharpe(self):
        """Sharpe at 0 bps must be ≥ Sharpe at 5 bps for all non-benchmark strategies."""
        daily_df = _make_daily_df()
        tc_df = compute_tc_sensitivity(daily_df, tc_levels_bps=[0, 5])
        for strat in ["defensive_overlay", "bull_aware_overlay",
                      "regime_momentum", "vol_managed"]:
            sub = tc_df[tc_df["strategy"] == strat].set_index("tc_bps")
            if 0 in sub.index and 5 in sub.index:
                s0 = sub.loc[0, "sharpe"]
                s5 = sub.loc[5, "sharpe"]
                assert s0 >= s5 - 1e-9, (
                    f"{strat}: Sharpe at 0bps ({s0:.4f}) < Sharpe at 5bps ({s5:.4f})"
                )

    def test_contains_all_strategies(self):
        """TC sensitivity output must contain all 5 strategies."""
        daily_df = _make_daily_df()
        tc_df = compute_tc_sensitivity(daily_df, tc_levels_bps=[0, 5])
        assert set(tc_df["strategy"].unique()) == {
            "benchmark", "defensive_overlay", "bull_aware_overlay",
            "regime_momentum", "vol_managed",
        }


class TestSubperiodPerformance:
    def test_covers_all_strategies(self):
        """Subperiod output contains all 5 strategies for each period."""
        daily_df = _make_daily_df(n=400, start="2023-01-02")
        subperiods = [{"name": "test_period", "start": "2023-01-02", "end": "2024-12-31"}]
        sp_df = compute_subperiod_performance(daily_df, subperiods)
        assert set(sp_df["strategy"].unique()) == {
            "benchmark", "defensive_overlay", "bull_aware_overlay",
            "regime_momentum", "vol_managed",
        }

    def test_empty_period_recorded_as_nan(self):
        """Subperiod with no data records NaN, not an error."""
        daily_df = _make_daily_df(n=100, start="2023-01-02")
        subperiods = [{"name": "future", "start": "2030-01-01", "end": "2030-12-31"}]
        sp_df = compute_subperiod_performance(daily_df, subperiods)
        assert len(sp_df) > 0
        assert sp_df["n_days"].iloc[0] == 0


class TestStrategyRanking:
    def test_contains_all_strategies(self):
        """Ranking output must contain all 5 strategies."""
        daily_df = _make_daily_df()
        mc = _make_metrics_comparison(daily_df)
        rank_df = compute_strategy_ranking(mc)
        strategies = set(rank_df["strategy"].tolist())
        expected = {"benchmark", "defensive_overlay", "bull_aware_overlay",
                    "regime_momentum", "vol_managed"}
        assert expected.issubset(strategies)

    def test_overall_rank_is_numeric(self):
        """Overall rank must be a finite numeric value for each strategy."""
        daily_df = _make_daily_df()
        mc = _make_metrics_comparison(daily_df)
        rank_df = compute_strategy_ranking(mc)
        assert "overall_rank" in rank_df.columns
        assert rank_df["overall_rank"].notna().all(), "NaN in overall_rank"

    def test_ranking_columns_present(self):
        """Ranking DF must have _rank columns for each objective."""
        daily_df = _make_daily_df()
        mc = _make_metrics_comparison(daily_df)
        rank_df = compute_strategy_ranking(mc)
        rank_cols = [c for c in rank_df.columns if c.endswith("_rank")]
        assert len(rank_cols) >= 3, f"Expected at least 3 rank columns, got {rank_cols}"
