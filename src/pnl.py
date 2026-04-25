"""Simulated PnL and economic impact analysis on a notional position."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import compute_all_metrics, TRADING_DAYS

INITIAL_CAPITAL: float = 100_000.0

_STRATEGIES: dict[str, str] = {
    "benchmark":          "benchmark_ret",
    "defensive_overlay":  "overlay_ret",
    "bull_aware_overlay": "bull_aware_ret",
    "regime_momentum":    "regime_momentum_ret",
    "vol_managed":        "vol_managed_ret",
}

_TC_COLS: dict[str, str] = {
    "defensive_overlay":  "transaction_cost",
    "bull_aware_overlay": "bull_aware_transaction_cost",
    "regime_momentum":    "regime_momentum_transaction_cost",
    "vol_managed":        "vol_managed_transaction_cost",
}


def compute_cumulative_pnl(
    daily_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Cumulative simulated PnL in USD for each available strategy."""
    result: dict[str, pd.Series] = {}
    for name, col in _STRATEGIES.items():
        if col in daily_df.columns:
            cum_ret = (1 + daily_df[col].fillna(0)).cumprod()
            result[name] = (cum_ret - 1) * initial_capital
    return pd.DataFrame(result, index=daily_df.index)


def compute_daily_pnl(
    daily_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Daily simulated PnL in USD (not compounded — approx for diagnostics)."""
    result: dict[str, pd.Series] = {}
    for name, col in _STRATEGIES.items():
        if col in daily_df.columns:
            result[name] = daily_df[col].fillna(0) * initial_capital
    return pd.DataFrame(result, index=daily_df.index)


def compute_regime_pnl(
    daily_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Cumulative simulated PnL per strategy, segmented by regime label."""
    if "regime_label" not in daily_df.columns:
        return pd.DataFrame()
    rows = []
    for regime in ["risk_on", "transition", "stress"]:
        mask = daily_df["regime_label"] == regime
        window = daily_df[mask]
        if len(window) == 0:
            continue
        for strat, col in _STRATEGIES.items():
            if col not in window.columns:
                continue
            cum = float((1 + window[col].fillna(0)).prod() - 1)
            rows.append({
                "regime":   regime,
                "strategy": strat,
                "cumulative_return_pct": round(cum * 100, 3),
                "cumulative_pnl_usd":   round(cum * initial_capital, 2),
                "n_days":               int(mask.sum()),
            })
    return pd.DataFrame(rows)


def compute_economic_impact(
    daily_df: pd.DataFrame,
    metrics_comparison_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Economic impact table — all key metrics on a notional $100k position."""
    mc = metrics_comparison_df.set_index("metric")

    def _get(metric: str, strat: str) -> float:
        try:
            return float(mc.loc[metric, strat])
        except (KeyError, ValueError):
            return float("nan")

    rows = []
    for strat, ret_col in _STRATEGIES.items():
        if ret_col not in daily_df.columns:
            continue
        total_return = _get("total_return", strat)
        final_value  = initial_capital * (1 + total_return)
        pnl          = final_value - initial_capital
        tc_pct       = _get("total_transaction_cost", strat)
        mdd_pct      = _get("max_drawdown", strat)
        rets         = daily_df[ret_col].dropna()
        worst_pct    = float(rets.min()) if len(rets) else float("nan")

        rows.append({
            "strategy":                    strat,
            "initial_capital":             initial_capital,
            "final_value":                round(final_value, 2),
            "simulated_pnl":              round(pnl, 2),
            "total_return_pct":           round(total_return * 100, 4),
            "cagr_pct":                   round(_get("cagr", strat) * 100, 2),
            "annualised_vol_pct":         round(_get("annualised_vol", strat) * 100, 2),
            "sharpe":                     round(_get("sharpe", strat), 3),
            "max_drawdown_pct":           round(mdd_pct * 100, 2),
            "max_drawdown_usd":           round(mdd_pct * initial_capital, 2),
            "worst_daily_loss_pct":       round(worst_pct * 100, 4),
            "worst_daily_loss_usd":       round(worst_pct * initial_capital, 2),
            "total_transaction_cost_pct": round(tc_pct * 100, 4),
            "total_transaction_cost_usd": round(tc_pct * initial_capital, 2),
            "average_equity_exposure_pct":round(_get("avg_equity_exposure", strat) * 100, 1),
        })

    return pd.DataFrame(rows)
