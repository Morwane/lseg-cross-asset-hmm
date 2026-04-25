"""Tests for performance metrics — Phase 7 implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_cagr,
    compute_annualised_vol,
    compute_sharpe,
    compute_max_drawdown,
    compute_sortino,
    compute_calmar,
)

TRADING_DAYS = 252


def test_cagr_on_flat_returns():
    """CAGR on a constant daily return matches the analytical compound formula."""
    daily_ret = 0.001  # 0.1% per day
    returns = pd.Series([daily_ret] * TRADING_DAYS)
    expected = (1 + daily_ret) ** TRADING_DAYS - 1
    result = compute_cagr(returns, trading_days=TRADING_DAYS)
    assert abs(result - expected) < 1e-10

    # Zero returns → CAGR = 0
    zeros = pd.Series([0.0] * TRADING_DAYS)
    assert abs(compute_cagr(zeros)) < 1e-12


def test_annualised_volatility():
    """Annualised vol equals daily std × sqrt(252)."""
    rng = np.random.default_rng(0)
    daily_std = 0.01
    returns = pd.Series(rng.normal(0.0005, daily_std, 2000))
    result = compute_annualised_vol(returns, trading_days=TRADING_DAYS)
    expected = returns.std() * np.sqrt(TRADING_DAYS)
    assert abs(result - expected) < 1e-12

    # Constant returns → vol = 0
    constant = pd.Series([0.001] * 100)
    assert compute_annualised_vol(constant) == 0.0


def test_sharpe_ratio():
    """Sharpe equals (CAGR − rf) / annualised_vol, verified against direct computation."""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))

    result = compute_sharpe(returns, risk_free=0.0, trading_days=TRADING_DAYS)
    cagr = compute_cagr(returns, TRADING_DAYS)
    vol = compute_annualised_vol(returns, TRADING_DAYS)
    expected = cagr / vol

    assert abs(result - expected) < 1e-10

    # Zero-vol series → Sharpe is NaN
    assert np.isnan(compute_sharpe(pd.Series([0.001] * 100), risk_free=0.0))


def test_max_drawdown():
    """Max drawdown is correctly computed from a known peak-to-trough sequence."""
    # Sequence: +10%, then −20%. Cumulative: 1.10 → 0.88. DD = 0.88/1.10 − 1 = −0.2
    returns = pd.Series([0.10, -0.20])
    result = compute_max_drawdown(returns)
    expected = 0.88 / 1.10 - 1  # ≈ -0.18182
    assert abs(result - expected) < 1e-10

    # Monotonically rising → drawdown = 0
    rising = pd.Series([0.01] * 50)
    assert compute_max_drawdown(rising) == 0.0

    # Single large loss
    returns2 = pd.Series([0.05, 0.05, -0.50, 0.02])
    result2 = compute_max_drawdown(returns2)
    assert result2 < -0.40  # must be at least −40%
