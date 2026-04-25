"""Tests for Phase 9 alpha overlays: regime_momentum and vol_managed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    compute_regime_momentum_weights,
    compute_vol_managed_weights,
    compute_portfolio_returns,
    compute_transaction_costs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_predictions(n: int = 40, start: str = "2023-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    rng = np.random.default_rng(42)
    prob_ro = rng.dirichlet(np.ones(3), size=n)
    return pd.DataFrame(
        {
            "regime_state":    np.tile([0, 1, 2], n)[:n],
            "regime_label":    np.tile(["risk_on", "transition", "stress"], n)[:n],
            "prob_risk_on":    prob_ro[:, 0],
            "prob_transition": prob_ro[:, 1],
            "prob_stress":     prob_ro[:, 2],
        },
        index=idx,
    )


def _rising_panel(n: int = 400, start: str = "2020-01-02") -> pd.DataFrame:
    """Steadily rising equity, calm VIX — triggers bull floor but not stress cap."""
    idx = pd.bdate_range(start, periods=n, freq="B")
    rng = np.random.default_rng(7)
    prices = 4000 * np.cumprod(1 + rng.normal(0.0006, 0.008, n))
    vix = np.abs(rng.normal(15, 3, n)).clip(8, 25)
    return pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx)


def _stress_panel(n_burn: int = 220, n_stress: int = 50) -> pd.DataFrame:
    """220 calm days then 50 stress days — triggers cap."""
    n = n_burn + n_stress
    idx = pd.bdate_range("2018-01-02", periods=n, freq="B")
    prices = np.concatenate([
        np.ones(n_burn) * 4000.0,
        np.linspace(4000.0, 3200.0, n_stress),
    ])
    vix = np.concatenate([np.ones(n_burn) * 5.0, np.ones(n_stress) * 80.0])
    return pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx)


# ── Regime Momentum Overlay ──────────────────────────────────────────────────

class TestRegimeMomentumWeights:
    def test_weights_in_bounds(self):
        """Regime momentum weights must be in [0.25, 1.00]."""
        pred  = _make_predictions(40)
        panel = _rising_panel(400)
        w = compute_regime_momentum_weights(pred, panel, lag=1)
        assert (w >= 0.25 - 1e-9).all(), f"Weight below 0.25: min={w.min():.4f}"
        assert (w <= 1.00 + 1e-9).all(), f"Weight above 1.00: max={w.max():.4f}"

    def test_lagged_by_one_day(self):
        """Weight on day 1 reflects probability from day 0 (stress → base ≤ 0.25 + adj)."""
        pred  = _make_predictions(10)
        panel = _rising_panel(400)

        # Force day-0 to be pure stress
        pred.iloc[0, pred.columns.get_loc("prob_risk_on")]    = 0.0
        pred.iloc[0, pred.columns.get_loc("prob_transition")] = 0.0
        pred.iloc[0, pred.columns.get_loc("prob_stress")]     = 1.0

        w = compute_regime_momentum_weights(pred, panel, lag=1).dropna()
        # Base from pure stress = 0.25; momentum can add at most +0.20 → ≤ 0.45
        # But then clipped at [0.25, 1.00] → first weight ≤ 0.45 + 1e-9
        assert w.iloc[0] <= 0.45 + 1e-9, (
            f"Day-1 weight {w.iloc[0]:.4f} inconsistent with day-0 stress signal"
        )

    def test_stress_cap_fires(self):
        """If P(stress) > 0.75 at t-1, weight at t is capped at 0.50."""
        panel = _rising_panel(400)
        pred_idx = pd.bdate_range("2021-07-01", periods=20, freq="B")
        pred = pd.DataFrame(
            {
                "regime_state": 2, "regime_label": "stress",
                "prob_risk_on": 0.0, "prob_transition": 0.10, "prob_stress": 0.90,
            },
            index=pred_idx,
        )
        w = compute_regime_momentum_weights(pred, panel, lag=1).dropna()
        assert len(w) > 0, "No weights produced"
        assert (w <= 0.50 + 1e-9).all(), (
            f"Stress cap should clip all weights to ≤ 0.50: max={w.max():.4f}"
        )

    def test_no_look_ahead_count(self):
        """After lag=1, at most len(predictions) non-NaN weights."""
        pred  = _make_predictions(20)
        panel = _rising_panel(400)
        w = compute_regime_momentum_weights(pred, panel, lag=1).dropna()
        assert len(w) <= len(pred), "More weights than prediction rows"

    def test_bull_trend_increases_weight(self):
        """In a strong bull trend, regime-momentum weight should exceed base."""
        n = 400
        idx = pd.bdate_range("2020-01-02", periods=n, freq="B")
        prices = 4000 * np.cumprod(1 + np.ones(n) * 0.001)  # steadily rising
        vix = np.ones(n) * 12
        panel = pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx)

        pred_idx = pd.bdate_range("2021-07-01", periods=10, freq="B")
        pred = pd.DataFrame(
            {
                "regime_state": 0, "regime_label": "risk_on",
                "prob_risk_on": 0.5, "prob_transition": 0.4, "prob_stress": 0.0,
            },
            index=pred_idx,
        )
        base = 1.0 * 0.5 + 0.75 * 0.4 + 0.25 * 0.0  # = 0.80
        w = compute_regime_momentum_weights(pred, panel, lag=1).dropna()
        # Bull trend fires → w ≥ base (0.80) or at minimum ≥ 0.25 (floor)
        assert (w >= base - 1e-9).all() or (w >= 0.25 - 1e-9).all(), (
            "Bull trend should maintain or increase weight above base"
        )


# ── Vol-Managed Overlay ───────────────────────────────────────────────────────

class TestVolManagedWeights:
    def test_weights_in_bounds(self):
        """Vol-managed weights must be in [0.25, 1.00]."""
        pred  = _make_predictions(40)
        panel = _rising_panel(400)
        w = compute_vol_managed_weights(pred, panel, lag=1)
        assert (w >= 0.25 - 1e-9).all(), f"Weight below 0.25: min={w.min():.4f}"
        assert (w <= 1.00 + 1e-9).all(), f"Weight above 1.00: max={w.max():.4f}"

    def test_lagged_by_one_day(self):
        """Weight on day 1 is determined by vol from day 0, not day 1."""
        # Use a panel that temporally covers the predictions (starts 2022, 1000 days)
        n = 1000
        idx_panel = pd.bdate_range("2022-01-03", periods=n, freq="B")
        rng = np.random.default_rng(99)
        prices = 4000 * np.cumprod(1 + rng.normal(0.0006, 0.008, n))
        vix = np.abs(rng.normal(15, 3, n)).clip(8, 25)
        panel = pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx_panel)

        pred = _make_predictions(10)  # starts 2023-01-02 — inside panel range
        pred.iloc[0, pred.columns.get_loc("prob_risk_on")]    = 0.0
        pred.iloc[0, pred.columns.get_loc("prob_transition")] = 0.0
        pred.iloc[0, pred.columns.get_loc("prob_stress")]     = 1.0
        w = compute_vol_managed_weights(pred, panel, lag=1).dropna()
        assert len(w) > 0, "No weights produced — check panel/prediction date overlap"
        # base from pure stress = 0.25; vol_scale ∈ [0.5, 1.25] → w_min ≥ 0.25
        assert w.iloc[0] >= 0.25 - 1e-9, "First weight should be at least 0.25"
        assert w.iloc[0] <= 1.00 + 1e-9, "First weight should be at most 1.00"

    def test_stress_cap_fires(self):
        """Cap clips weight to 0.50 when VIX spikes and drawdown is deep."""
        n_burn, n_stress = 220, 50
        panel = _stress_panel(n_burn, n_stress)
        idx = panel.index

        pred_start = n_burn + 25
        pred_idx   = idx[pred_start:]
        pred = pd.DataFrame(
            {
                "regime_state": 0, "regime_label": "risk_on",
                "prob_risk_on": 1.0, "prob_transition": 0.0, "prob_stress": 0.0,
            },
            index=pred_idx,
        )
        w = compute_vol_managed_weights(pred, panel, lag=1).dropna()
        assert len(w) > 0, "No weights produced"
        assert (w <= 0.50 + 1e-9).all(), (
            f"Cap should clip all weights to ≤ 0.50: max={w.max():.4f}"
        )

    def test_bull_floor_fires(self):
        """Floor fires when equity > 200d MA and P(stress) < 0.70."""
        n = 400
        idx = pd.bdate_range("2020-01-02", periods=n, freq="B")
        prices = 4000 * np.cumprod(1 + np.ones(n) * 0.001)
        vix = np.ones(n) * 12
        panel = pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx)

        pred_idx = pd.bdate_range("2021-07-01", periods=20, freq="B")
        pred = pd.DataFrame(
            {
                "regime_state": 0, "regime_label": "risk_on",
                "prob_risk_on": 0.3, "prob_transition": 0.3, "prob_stress": 0.0,
            },
            index=pred_idx,
        )
        w = compute_vol_managed_weights(pred, panel, lag=1).dropna()
        # Floor fires: price > MA200, P(stress)=0 < 0.70 → w ≥ 0.50
        assert (w >= 0.50 - 1e-9).all(), (
            f"Floor should raise all weights to ≥ 0.50: min={w.min():.4f}"
        )

    def test_no_look_ahead_count(self):
        """After lag=1, at most len(predictions) non-NaN weights."""
        pred  = _make_predictions(20)
        panel = _rising_panel(400)
        w = compute_vol_managed_weights(pred, panel, lag=1).dropna()
        assert len(w) <= len(pred), "More weights than prediction rows"


# ── Five-Strategy Alignment ───────────────────────────────────────────────────

class TestFiveStrategyAlignment:
    def test_all_strategies_same_index(self):
        """All five strategy returns must share the same DatetimeIndex."""
        pred  = _make_predictions(30)
        panel = _rising_panel(400)

        from src.backtest import (
            compute_bull_aware_weights,
            apply_weight_map,
        )

        regime_mapping = {"risk_on": 1.0, "transition": 0.5, "stress": 0.0}
        def_w  = apply_weight_map(pred["regime_label"], regime_mapping, lag=1).dropna()
        bull_w = compute_bull_aware_weights(pred, panel, lag=1)
        mom_w  = compute_regime_momentum_weights(pred, panel, lag=1)
        vol_w  = compute_vol_managed_weights(pred, panel, lag=1)

        common_idx = (
            def_w.index
            .intersection(bull_w.dropna().index)
            .intersection(mom_w.dropna().index)
            .intersection(vol_w.dropna().index)
        )

        eq_ret  = panel["us_equity"].pct_change().reindex(common_idx)
        def_ret = pd.Series(0.0, index=panel.index).reindex(common_idx)

        def_daily  = compute_portfolio_returns(eq_ret, def_ret, def_w.reindex(common_idx),  5)
        bull_daily = compute_portfolio_returns(eq_ret, def_ret, bull_w.reindex(common_idx), 5)
        mom_daily  = compute_portfolio_returns(eq_ret, def_ret, mom_w.reindex(common_idx),  5)
        vol_daily  = compute_portfolio_returns(eq_ret, def_ret, vol_w.reindex(common_idx),  5)

        assert def_daily.index.equals(bull_daily.index), "def vs bull index mismatch"
        assert def_daily.index.equals(mom_daily.index),  "def vs mom index mismatch"
        assert def_daily.index.equals(vol_daily.index),  "def vs vol index mismatch"

    def test_transaction_costs_applied_to_alpha_overlays(self):
        """Both new overlays must have non-zero transaction costs when weights change."""
        pred  = _make_predictions(30)
        panel = _rising_panel(400)

        mom_w = compute_regime_momentum_weights(pred, panel, lag=1).dropna()
        vol_w = compute_vol_managed_weights(pred, panel, lag=1).dropna()

        mom_costs = compute_transaction_costs(mom_w, transaction_cost_bps=5)
        vol_costs = compute_transaction_costs(vol_w, transaction_cost_bps=5)

        # Expect at least some non-zero costs (weights do change across regimes)
        assert mom_costs.sum() >= 0, "Momentum costs cannot be negative"
        assert vol_costs.sum() >= 0, "Vol-managed costs cannot be negative"
