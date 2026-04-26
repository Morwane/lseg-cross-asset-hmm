"""Robustness checks for the regime overlay backtest.

Three checks:
  1. Transaction-cost sensitivity  — Sharpe/CAGR/MDD at 0/5/10/25/50 bps
  2. Subperiod performance         — full-period breakdown across market regimes
  3. Strategy ranking              — multi-objective ranking across strategies
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import compute_all_metrics, TRADING_DAYS

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

DEFAULT_TC_LEVELS = [0, 5, 10, 25, 50]

DEFAULT_SUBPERIODS = [
    # Historical crises — in OOS window with 5-year initial training from 2005
    {"name": "2011 Eurozone crisis",  "start": "2011-07-01", "end": "2011-10-31"},
    {"name": "2015-16 China/oil",     "start": "2015-08-01", "end": "2016-02-29"},
    {"name": "2018 Q4 selloff",       "start": "2018-09-28", "end": "2018-12-31"},
    {"name": "2020 COVID crash",      "start": "2020-02-15", "end": "2020-04-30"},
    {"name": "2022 rates shock",      "start": "2022-01-01", "end": "2022-12-31"},
    # Recent OOS window
    {"name": "2023 recovery",         "start": "2023-04-04", "end": "2023-12-31"},
    {"name": "2024 bull market",      "start": "2024-01-01", "end": "2024-12-31"},
    {"name": "2025 tariff corr.",     "start": "2025-01-20", "end": "2025-04-30"},
    {"name": "2026 latest",           "start": "2026-01-01", "end": "2026-04-23"},
]


# ── Check 1: Transaction cost sensitivity ────────────────────────────────────

def compute_tc_sensitivity(
    daily_df: pd.DataFrame,
    tc_levels_bps: list[int] | None = None,
    original_tc_bps: int = 5,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Recompute Sharpe/CAGR/MDD for each strategy at each TC level.

    For non-benchmark strategies the original TC column is scaled to the new
    level without re-running the weight computation.
    """
    if tc_levels_bps is None:
        tc_levels_bps = DEFAULT_TC_LEVELS

    rows = []
    for strat, ret_col in _STRATEGIES.items():
        if ret_col not in daily_df.columns:
            continue
        tc_col = _TC_COLS.get(strat)

        for tc_bps in tc_levels_bps:
            if tc_col and tc_col in daily_df.columns and original_tc_bps > 0:
                orig_tc = daily_df[tc_col].fillna(0)
                gross   = daily_df[ret_col] + orig_tc
                new_tc  = orig_tc * (tc_bps / original_tc_bps)
                adj_ret = gross - new_tc
            else:
                adj_ret = daily_df[ret_col]

            m = compute_all_metrics(adj_ret.dropna(), strat, trading_days)
            rows.append({
                "strategy":     strat,
                "tc_bps":       tc_bps,
                "cagr":         m["cagr"],
                "sharpe":       m["sharpe"],
                "sortino":      m["sortino"],
                "calmar":       m["calmar"],
                "max_drawdown": m["max_drawdown"],
                "total_return": m["total_return"],
            })

    return pd.DataFrame(rows)


# ── Check 2: Subperiod performance ───────────────────────────────────────────

def compute_subperiod_performance(
    daily_df: pd.DataFrame,
    subperiods: list[dict] | None = None,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Compute key metrics for each strategy over named market subperiods."""
    if subperiods is None:
        subperiods = DEFAULT_SUBPERIODS

    rows = []
    for sp in subperiods:
        s = pd.Timestamp(sp["start"])
        e = pd.Timestamp(sp["end"])
        window = daily_df.loc[s:e] if s <= daily_df.index[-1] else pd.DataFrame()

        for strat, ret_col in _STRATEGIES.items():
            if len(window) == 0 or ret_col not in window.columns:
                rows.append({
                    "subperiod":    sp["name"],
                    "strategy":     strat,
                    "n_days":       0,
                    "total_return": float("nan"),
                    "cagr":         float("nan"),
                    "sharpe":       float("nan"),
                    "max_drawdown": float("nan"),
                })
                continue

            rets = window[ret_col].dropna()
            if len(rets) == 0:
                rows.append({
                    "subperiod":    sp["name"],
                    "strategy":     strat,
                    "n_days":       0,
                    "total_return": float("nan"),
                    "cagr":         float("nan"),
                    "sharpe":       float("nan"),
                    "max_drawdown": float("nan"),
                })
                continue

            m = compute_all_metrics(rets, strat, trading_days)
            rows.append({
                "subperiod":    sp["name"],
                "strategy":     strat,
                "n_days":       int(m["n_days"]),
                "total_return": m["total_return"],
                "cagr":         m["cagr"],
                "sharpe":       m["sharpe"],
                "max_drawdown": m["max_drawdown"],
            })

    return pd.DataFrame(rows)  # noqa: outside loop


# ── Check 3: Strategy ranking ─────────────────────────────────────────────────

_RANKING_OBJECTIVES: dict[str, tuple[str, float]] = {
    "cagr":                   ("higher_is_better",  1.0),
    "sharpe":                 ("higher_is_better",  1.0),
    "calmar":                 ("higher_is_better",  1.0),
    "max_drawdown":           ("lower_abs_better",  -1.0),  # closer to 0 is better
    "worst_daily_loss":       ("lower_abs_better",  -1.0),
    "avg_equity_exposure":    ("lower_is_better",   -1.0),  # lower = more defensive
    "total_transaction_cost": ("lower_is_better",   -1.0),
}


def compute_strategy_ranking(
    metrics_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank all strategies by multiple objectives (1 = best per objective)."""
    mc = metrics_comparison_df.set_index("metric")
    strategies = [c for c in mc.columns]

    raw_rows = []
    for strat in strategies:
        row: dict = {"strategy": strat}
        for metric in _RANKING_OBJECTIVES:
            if metric in mc.index:
                row[metric] = float(mc.loc[metric, strat])
            else:
                row[metric] = float("nan")
        raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows).set_index("strategy")

    # Rank: transform each metric so that "higher = better", then rank desc
    rank_df = pd.DataFrame(index=raw_df.index)
    for metric, (_, sign) in _RANKING_OBJECTIVES.items():
        if metric not in raw_df.columns:
            continue
        # Apply sign so higher transformed value is always better
        transformed = raw_df[metric] * sign
        rank_df[metric] = transformed.rank(ascending=False, na_option="bottom")

    rank_df["overall_rank"] = rank_df.mean(axis=1).rank()

    # Build output: raw values + per-metric rank columns + overall rank
    result = raw_df.copy()
    for metric in list(_RANKING_OBJECTIVES.keys()):
        if metric in rank_df.columns:
            result[f"{metric}_rank"] = rank_df[metric]
    result["overall_rank"] = rank_df["overall_rank"]

    return result.reset_index().sort_values("overall_rank")
