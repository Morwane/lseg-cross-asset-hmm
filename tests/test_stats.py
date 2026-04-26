"""Tests for statistical robustness checks — Phase 3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stats import (
    deflated_sharpe_ratio,
    bootstrap_confidence_intervals,
    compute_all_stats,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_returns(n: int = 500, mean: float = 0.0006, std: float = 0.01,
                  seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def _make_daily_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n, freq="B")
    eq  = rng.normal(0.0006, 0.010, n)
    return pd.DataFrame({
        "benchmark_ret":       eq,
        "overlay_ret":         rng.normal(0.0002, 0.005, n),
        "bull_aware_ret":      rng.normal(0.0004, 0.007, n),
        "regime_momentum_ret": rng.normal(0.0005, 0.008, n),
        "vol_managed_ret":     rng.normal(0.0004, 0.007, n),
    }, index=idx)


# ── DSR tests ─────────────────────────────────────────────────────────────────

class TestDeflatedSharpeRatio:
    def test_dsr_is_probability(self):
        """DSR must always be in [0, 1]."""
        rets = _make_returns()
        result = deflated_sharpe_ratio(rets, n_trials=5)
        assert 0.0 <= result["dsr"] <= 1.0, f"DSR out of range: {result['dsr']}"

    def test_high_sr_gives_high_dsr(self):
        """Returns with annualised Sharpe ~2 should yield DSR > 0.90."""
        rets = _make_returns(mean=0.0015, std=0.008, n=500)
        result = deflated_sharpe_ratio(rets, n_trials=1)
        assert result["dsr"] > 0.90, (
            f"Expected DSR > 0.90 for high-SR returns, got {result['dsr']:.4f}"
        )

    def test_zero_sr_gives_near_half(self):
        """Demeaned returns (SR=0 exactly) with n_trials=1 should yield DSR=0.5."""
        rng = np.random.default_rng(7)
        raw  = pd.Series(rng.normal(0.0, 0.01, 2000))
        rets = raw - raw.mean()       # force daily mean = 0 → SR = 0 exactly
        result = deflated_sharpe_ratio(rets, n_trials=1)
        assert abs(result["dsr"] - 0.5) < 1e-6, (
            f"Demeaned-SR=0 should give DSR=0.5, got {result['dsr']:.6f}"
        )

    def test_more_trials_reduces_dsr(self):
        """More strategies tested → stricter multiple-comparison correction → lower DSR."""
        rets = _make_returns(mean=0.0006, std=0.010, n=500)
        dsr1 = deflated_sharpe_ratio(rets, n_trials=1)["dsr"]
        dsr5 = deflated_sharpe_ratio(rets, n_trials=5)["dsr"]
        assert dsr5 <= dsr1, (
            f"DSR with 5 trials ({dsr5:.4f}) should be ≤ DSR with 1 trial ({dsr1:.4f})"
        )

    def test_negative_sr_gives_low_dsr(self):
        """Consistently losing strategy should have DSR < 0.2."""
        rets = _make_returns(mean=-0.001, std=0.005, n=500)
        result = deflated_sharpe_ratio(rets, n_trials=5)
        assert result["dsr"] < 0.2, (
            f"Negative-SR strategy should have low DSR, got {result['dsr']:.4f}"
        )

    def test_output_keys_present(self):
        """Result dict must contain all required diagnostic keys."""
        required = {"dsr", "sr_annualized", "sr_daily", "sr_star_daily",
                    "sr_std", "skewness", "excess_kurtosis", "n_obs", "n_trials"}
        result = deflated_sharpe_ratio(_make_returns(), n_trials=3)
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_short_series_returns_nan(self):
        """Series with < 30 observations should return NaN DSR."""
        rets = _make_returns(n=20)
        result = deflated_sharpe_ratio(rets, n_trials=1)
        assert np.isnan(result["dsr"]), "Short series should return NaN"


# ── Bootstrap CI tests ────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_ci_ordering(self):
        """p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95 for Sharpe and CAGR."""
        rets = _make_returns(n=300)
        ci = bootstrap_confidence_intervals(rets, n_bootstrap=500, seed=0)
        for m in ("sharpe", "cagr", "mdd"):
            vals = [ci[f"{m}_p{p}"] for p in (5, 25, 50, 75, 95)]
            assert all(a <= b for a, b in zip(vals, vals[1:])), (
                f"{m} percentiles not ordered: {vals}"
            )

    def test_bootstrap_reproducible(self):
        """Same seed → identical CI values."""
        rets = _make_returns(n=200)
        ci1 = bootstrap_confidence_intervals(rets, n_bootstrap=200, seed=42)
        ci2 = bootstrap_confidence_intervals(rets, n_bootstrap=200, seed=42)
        assert ci1 == ci2, "Bootstrap result should be deterministic with fixed seed"

    def test_bootstrap_seed_matters(self):
        """Different seeds should (almost certainly) give different results."""
        rets = _make_returns(n=200)
        ci1 = bootstrap_confidence_intervals(rets, n_bootstrap=200, seed=1)
        ci2 = bootstrap_confidence_intervals(rets, n_bootstrap=200, seed=2)
        assert ci1["sharpe_p50"] != ci2["sharpe_p50"], (
            "Different seeds produced identical median — extremely unlikely"
        )

    def test_all_strategies_covered(self):
        """compute_all_stats must return a row for each of the 5 strategies."""
        daily_df = _make_daily_df()
        dsr_df, boot_df = compute_all_stats(daily_df, n_trials=5, n_bootstrap=200)
        expected = {"benchmark", "defensive_overlay", "bull_aware_overlay",
                    "regime_momentum", "vol_managed"}
        assert expected.issubset(set(dsr_df["strategy"])), (
            f"Missing strategies in DSR df: {expected - set(dsr_df['strategy'])}"
        )
        assert expected.issubset(set(boot_df["strategy"])), (
            f"Missing strategies in bootstrap df: {expected - set(boot_df['strategy'])}"
        )

    def test_ci_width_shrinks_with_more_data(self):
        """Longer return series → narrower CI (law of large numbers)."""
        rets_short = _make_returns(n=100, seed=1)
        rets_long  = _make_returns(n=1000, seed=1)
        ci_short = bootstrap_confidence_intervals(rets_short, n_bootstrap=500, seed=0)
        ci_long  = bootstrap_confidence_intervals(rets_long,  n_bootstrap=500, seed=0)
        width_short = ci_short["sharpe_p95"] - ci_short["sharpe_p5"]
        width_long  = ci_long["sharpe_p95"]  - ci_long["sharpe_p5"]
        assert width_long < width_short, (
            f"Longer series should give narrower CI: "
            f"short={width_short:.3f} long={width_long:.3f}"
        )
