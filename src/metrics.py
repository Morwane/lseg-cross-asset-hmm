"""Performance and risk metrics for backtesting."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def compute_cagr(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Compound annual growth rate from a series of daily returns."""
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    total = float((1 + clean).prod())
    return float(total ** (trading_days / len(clean)) - 1)


def compute_annualised_vol(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of daily returns."""
    clean = returns.dropna()
    vol = float(clean.std() * np.sqrt(trading_days))
    return vol if vol > 1e-12 else 0.0


def compute_sharpe(
    returns: pd.Series,
    risk_free: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio: (CAGR − risk_free) / annualised_vol."""
    vol = compute_annualised_vol(returns, trading_days)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return (compute_cagr(returns, trading_days) - risk_free) / vol


def compute_sortino(
    returns: pd.Series,
    risk_free: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio using downside deviation below zero."""
    clean = returns.dropna()
    daily_rf = risk_free / trading_days
    downside = clean[clean < daily_rf] - daily_rf
    if len(downside) == 0:
        return float("nan")
    downside_vol = float(np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days))
    if downside_vol == 0:
        return float("nan")
    return (compute_cagr(returns, trading_days) - risk_free) / downside_vol


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction (e.g. -0.25 = -25%)."""
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    cumulative = (1 + clean).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())


def compute_calmar(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Calmar ratio: CAGR / |max drawdown|."""
    mdd = compute_max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return compute_cagr(returns, trading_days) / abs(mdd)


def compute_all_metrics(
    returns: pd.Series,
    label: str = "strategy",
    trading_days: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> dict:
    """Return a dict of all standard backtest metrics for the given return series."""
    clean = returns.dropna()
    return {
        "label": label,
        "n_days": len(clean),
        "cagr": compute_cagr(returns, trading_days),
        "annualised_vol": compute_annualised_vol(returns, trading_days),
        "sharpe": compute_sharpe(returns, risk_free, trading_days),
        "sortino": compute_sortino(returns, risk_free, trading_days),
        "max_drawdown": compute_max_drawdown(returns),
        "calmar": compute_calmar(returns, trading_days),
        "total_return": float((1 + clean).prod() - 1),
        "hit_rate": float((clean > 0).mean()),
        "worst_daily_loss": float(clean.min()) if len(clean) else float("nan"),
        "best_daily_gain": float(clean.max()) if len(clean) else float("nan"),
    }
