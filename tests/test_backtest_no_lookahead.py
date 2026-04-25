"""Tests for backtest no-lookahead guarantee — Phase 7 implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    apply_weight_map,
    compute_transaction_costs,
    compute_portfolio_returns,
)


def _make_regime_series(labels: list[str], start: str = "2023-01-02") -> pd.Series:
    """Build a regime_label Series with a business-day index."""
    idx = pd.bdate_range(start, periods=len(labels), freq="B")
    return pd.Series(labels, index=idx)


def test_signal_on_day_t_uses_only_t_minus_1_information():
    """Equity weight on day t is derived from the regime label at day t-1.

    With lag=1: weight on idx[1] = f(label on idx[0]), etc.
    The first row (idx[0]) has no prior signal and is dropped.
    """
    mapping = {"risk_on": 1.0, "transition": 0.5, "stress": 0.0}
    labels = _make_regime_series(["risk_on", "stress", "transition", "risk_on"])
    idx = labels.index

    weights = apply_weight_map(labels, mapping, lag=1)
    # NaN for first row — drop manually to check clean values
    clean = weights.dropna()

    # weight on idx[1] = f(label on idx[0] = "risk_on") = 1.0
    assert clean[idx[1]] == pytest.approx(1.0)
    # weight on idx[2] = f(label on idx[1] = "stress") = 0.0
    assert clean[idx[2]] == pytest.approx(0.0)
    # weight on idx[3] = f(label on idx[2] = "transition") = 0.5
    assert clean[idx[3]] == pytest.approx(0.5)

    # First backtest date must be idx[1] — idx[0] is dropped (no prior signal)
    assert clean.index[0] == idx[1]
    assert len(clean) == len(labels) - 1


def test_transaction_costs_deducted_on_weight_change():
    """Costs are applied proportionally to the absolute weight change.

    On the first day, the prior position is 0 (entering from cash).
    On days where weight is unchanged, cost must be 0.
    """
    cost_bps = 5
    idx = pd.bdate_range("2023-01-02", periods=4, freq="B")
    weights = pd.Series([1.0, 1.0, 0.5, 0.0], index=idx)

    costs = compute_transaction_costs(weights, transaction_cost_bps=cost_bps)

    # Day 0: enter from 0 → |1.0 − 0.0| × 5/10000
    assert costs.iloc[0] == pytest.approx(1.0 * cost_bps / 10_000)
    # Day 1: no change (1.0 → 1.0) → cost = 0
    assert costs.iloc[1] == pytest.approx(0.0)
    # Day 2: |0.5 − 1.0| = 0.5 → 0.5 × 5/10000
    assert costs.iloc[2] == pytest.approx(0.5 * cost_bps / 10_000)
    # Day 3: |0.0 − 0.5| = 0.5 → same
    assert costs.iloc[3] == pytest.approx(0.5 * cost_bps / 10_000)


def test_benchmark_and_overlay_use_aligned_dates():
    """Benchmark and overlay return series share exactly the same DatetimeIndex."""
    idx = pd.bdate_range("2023-01-02", periods=10, freq="B")
    # Equity and defensive returns cover the same dates as the weight index
    eq_ret = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, len(idx)), index=idx)
    def_ret = pd.Series(0.0001, index=idx)
    weights = pd.Series([1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5, 1.0], index=idx)

    daily = compute_portfolio_returns(eq_ret, def_ret, weights, transaction_cost_bps=5)

    assert list(daily["benchmark_ret"].index) == list(daily["overlay_ret"].index)
    assert daily.index.equals(weights.index)
    assert not daily["benchmark_ret"].isna().any()
    assert not daily["overlay_ret"].isna().any()
