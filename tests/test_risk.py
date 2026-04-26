"""Tests for Phase 4 professional risk controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk import (
    apply_vol_target,
    apply_equity_curve_stop,
    apply_turnover_cap,
    compute_cvar_by_regime,
    build_risk_controlled_returns,
)
from src.metrics import TRADING_DAYS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_weights(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Regime-like weights: alternating blocks between 0.3 and 0.8
    w = np.where(np.arange(n) % 60 < 30, 0.7, 0.3)
    return pd.Series(w.astype(float))


def _make_returns(n: int = 300, mean: float = 0.0005, std: float = 0.01,
                  seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def _make_high_vol_returns(n: int = 300, seed: int = 1) -> pd.Series:
    """Returns with annualised vol ~40% (vs 8% target)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.025, n))   # daily std ~2.5% → ~40% annual


def _make_daily_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n, freq="B")
    eq  = pd.Series(rng.normal(0.0006, 0.010, n), index=idx)
    df  = pd.DataFrame({
        "benchmark_ret":                  eq,
        "overlay_ret":                    rng.normal(0.0003, 0.006, n),
        "equity_weight":                  np.clip(rng.normal(0.55, 0.15, n), 0.05, 0.95),
        "defensive_weight":               None,
        "transaction_cost":               np.abs(rng.normal(0.0001, 0.00005, n)),
        "regime_label":                   np.where(np.arange(n) % 90 < 45,
                                                   "risk_on",
                                                   np.where(np.arange(n) % 90 < 70,
                                                            "transition", "stress")),
        "bull_aware_equity_weight":       np.clip(rng.normal(0.60, 0.15, n), 0.05, 0.95),
        "bull_aware_defensive_weight":    None,
        "bull_aware_transaction_cost":    np.abs(rng.normal(0.0001, 0.00005, n)),
        "bull_aware_ret":                 rng.normal(0.0004, 0.007, n),
        "regime_momentum_equity_weight":  np.clip(rng.normal(0.50, 0.20, n), 0.05, 0.95),
        "regime_momentum_defensive_weight": None,
        "regime_momentum_transaction_cost": np.abs(rng.normal(0.0001, 0.00005, n)),
        "regime_momentum_ret":            rng.normal(0.0005, 0.008, n),
        "vol_managed_equity_weight":      np.clip(rng.normal(0.55, 0.15, n), 0.05, 0.95),
        "vol_managed_defensive_weight":   None,
        "vol_managed_transaction_cost":   np.abs(rng.normal(0.0001, 0.00005, n)),
        "vol_managed_ret":                rng.normal(0.0004, 0.007, n),
    }, index=idx)
    return df


# ── Vol targeting ─────────────────────────────────────────────────────────────

class TestVolTarget:
    def test_weights_bounded(self):
        """Vol-targeted weights must stay in [0, 1]."""
        w   = _make_weights()
        ret = _make_returns()
        out = apply_vol_target(w, ret, target_vol=0.08)
        assert out.min() >= 0.0, f"Weight below 0: {out.min()}"
        assert out.max() <= 1.0, f"Weight above 1: {out.max()}"

    def test_high_vol_reduces_weights(self):
        """When realised vol >> target, scale < 1 → weights should decrease on average."""
        w     = pd.Series(np.full(300, 0.80))
        ret   = _make_high_vol_returns()
        out   = apply_vol_target(w, ret, target_vol=0.08, min_scale=0.0, max_scale=1.0)
        # After warm-up period, most scaled weights should be less than original 0.80
        assert out.iloc[30:].mean() < 0.80, (
            f"High-vol case: expected mean weight < 0.80, got {out.iloc[30:].mean():.3f}"
        )

    def test_same_output_when_vol_on_target(self):
        """When realised vol matches target exactly, scale ≈ 1 → weights unchanged."""
        target = 0.10
        daily_target = target / np.sqrt(TRADING_DAYS)
        rng = np.random.default_rng(99)
        ret = pd.Series(rng.normal(0.0, daily_target, 300))
        w   = pd.Series(np.full(300, 0.60))
        out = apply_vol_target(w, ret, target_vol=target, min_scale=0.5, max_scale=2.0,
                               lookback=20)
        # After warm-up, scale ≈ 1, so output ≈ 0.60
        assert abs(out.iloc[50:].mean() - 0.60) < 0.10, (
            f"On-target vol: expected mean ~0.60, got {out.iloc[50:].mean():.3f}"
        )


# ── Equity-curve stop ─────────────────────────────────────────────────────────

class TestEquityCurveStop:
    def test_stop_triggers_at_threshold(self):
        """After a large drawdown, weights must not exceed defensive_floor."""
        n   = 200
        ret = pd.Series(np.full(n, -0.005))   # steady -0.5%/day → deep drawdown
        w   = pd.Series(np.full(n, 0.80))
        out, stop_active = apply_equity_curve_stop(
            w, ret, drawdown_threshold=-0.05, recovery_threshold=-0.01,
            defensive_floor=0.20,
        )
        # Eventually stop should have fired; weights after that must be ≤ 0.20
        assert stop_active.any(), "Stop-loss should have been triggered"
        assert out[stop_active].max() <= 0.20 + 1e-9, (
            f"Weights exceed floor when stop active: {out[stop_active].max():.4f}"
        )

    def test_no_stop_when_positive_returns(self):
        """With consistently positive returns, stop should never fire."""
        n   = 200
        ret = pd.Series(np.full(n, 0.003))    # +0.3%/day → no drawdown
        w   = pd.Series(np.full(n, 0.70))
        out, stop_active = apply_equity_curve_stop(
            w, ret, drawdown_threshold=-0.10, recovery_threshold=-0.05,
            defensive_floor=0.15,
        )
        assert not stop_active.any(), "Stop-loss should not fire with positive returns"
        assert (out == 0.70).all(), "Weights should be unchanged without stop"

    def test_recovery_restores_weight(self):
        """After recovery above recovery_threshold, weights should return to original."""
        n   = 500
        rets = (
            [-0.008] * 100    # drawdown phase → trigger stop at ~-0.10
          + [+0.010] * 150    # recovery phase → drawdown climbs above -0.05
          + [+0.0001] * 250   # gentle drift — new high, no second drawdown
        )
        ret = pd.Series(rets[:n])
        w   = pd.Series(np.full(n, 0.75))
        out, stop_active = apply_equity_curve_stop(
            w, ret, drawdown_threshold=-0.10, recovery_threshold=-0.05,
            defensive_floor=0.15,
        )
        # Stop should have been active during drawdown
        assert stop_active.iloc[:150].any(), "Stop should fire during drawdown"
        # After recovery + new all-time-high, stop should be off
        assert not stop_active.iloc[350:].any(), "Stop should lift after full recovery"


# ── Turnover cap ──────────────────────────────────────────────────────────────

class TestTurnoverCap:
    def test_max_change_never_exceeded(self):
        """Daily weight changes must never exceed max_daily_change."""
        rng = np.random.default_rng(5)
        w   = pd.Series(rng.uniform(0, 1, 300))
        out = apply_turnover_cap(w, max_daily_change=0.05)
        daily_changes = out.diff().abs().dropna()
        assert daily_changes.max() <= 0.05 + 1e-9, (
            f"Turnover cap violated: max change = {daily_changes.max():.5f}"
        )

    def test_small_changes_pass_through(self):
        """Weights that change slowly should be unmodified."""
        w   = pd.Series(np.linspace(0.3, 0.7, 100))   # ~0.004 change/day
        out = apply_turnover_cap(w, max_daily_change=0.10)
        assert np.allclose(w.values, out.values, atol=1e-9), (
            "Small daily changes should not be altered by the turnover cap"
        )


# ── CVaR ──────────────────────────────────────────────────────────────────────

class TestCVaR:
    def test_cvar_leq_var(self):
        """CVaR (ES) must be ≤ VaR for all rows (CVaR is deeper in the tail)."""
        df = _make_daily_df()
        cvar_df = compute_cvar_by_regime(df, confidence_levels=[0.95])
        for _, row in cvar_df.iterrows():
            assert row["cvar"] <= row["var"] + 1e-9, (
                f"CVaR ({row['cvar']:.4f}) > VaR ({row['var']:.4f}) for "
                f"{row['strategy']} / {row['regime']}"
            )

    def test_output_columns_present(self):
        """Output DataFrame must have all required columns."""
        df = _make_daily_df()
        cvar_df = compute_cvar_by_regime(df)
        required = {"strategy", "regime", "confidence", "var", "cvar", "n_obs"}
        assert required.issubset(cvar_df.columns), (
            f"Missing columns: {required - set(cvar_df.columns)}"
        )

    def test_stress_cvar_worse_than_riskOn(self):
        """CVaR should be worse (more negative) in stress than risk-on regimes."""
        df = _make_daily_df()
        # Override regime to make stress regime have uniformly bad returns
        stress_mask = df["regime_label"] == "stress"
        df.loc[stress_mask, "overlay_ret"]  = -0.015   # -1.5%/day in stress
        df.loc[~stress_mask, "overlay_ret"] =  0.005   # +0.5%/day otherwise
        cvar_df = compute_cvar_by_regime(df, confidence_levels=[0.95])
        def_cvar = cvar_df[cvar_df["strategy"] == "defensive_overlay"]
        cvar_stress  = float(def_cvar[def_cvar["regime"] == "stress"]["cvar"].iloc[0])
        cvar_riskOn  = float(def_cvar[def_cvar["regime"] == "risk_on"]["cvar"].iloc[0])
        assert cvar_stress < cvar_riskOn, (
            f"Stress CVaR ({cvar_stress:.4f}) should be worse than "
            f"risk-on CVaR ({cvar_riskOn:.4f})"
        )


# ── Build risk-controlled returns ──────────────────────────────────────────────

class TestBuildRiskControlledReturns:
    def test_output_columns_present(self):
        """build_risk_controlled_returns must add all rc_ columns."""
        df  = _make_daily_df()
        out = build_risk_controlled_returns(df)
        expected_cols = [
            "rc_def_equity_weight",  "rc_def_ret",  "rc_def_stop_active",
            "rc_bull_equity_weight", "rc_bull_ret", "rc_bull_stop_active",
            "rc_mom_equity_weight",  "rc_mom_ret",  "rc_mom_stop_active",
            "rc_vol_equity_weight",  "rc_vol_ret",  "rc_vol_stop_active",
        ]
        missing = [c for c in expected_cols if c not in out.columns]
        assert not missing, f"Missing output columns: {missing}"

    def test_rc_weights_bounded(self):
        """Risk-controlled equity weights must stay in [0, 1]."""
        df  = _make_daily_df()
        out = build_risk_controlled_returns(df)
        for col in [c for c in out.columns if c.endswith("_equity_weight")
                    and c.startswith("rc_")]:
            assert out[col].min() >= 0.0, f"{col} below 0"
            assert out[col].max() <= 1.0, f"{col} above 1"

    def test_original_columns_preserved(self):
        """All original columns must still be present in the output."""
        df  = _make_daily_df()
        out = build_risk_controlled_returns(df)
        for col in df.columns:
            if col not in (
                "defensive_weight", "bull_aware_defensive_weight",
                "regime_momentum_defensive_weight", "vol_managed_defensive_weight"
            ):
                assert col in out.columns, f"Original column '{col}' missing from output"
