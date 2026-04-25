#!/usr/bin/env python3
"""Run robustness checks and generate charts 22-24.

Reads:
  output/reports/backtest_daily_returns.csv
  output/reports/backtest_metrics_comparison.csv

Writes:
  output/reports/transaction_cost_sensitivity.csv
  output/reports/subperiod_performance.csv
  output/reports/strategy_ranking_by_objective.csv

Generates:
  output/charts/22_transaction_cost_sensitivity.png/svg
  output/charts/23_subperiod_performance.png/svg
  output/charts/24_strategy_ranking.png/svg

Usage:
    python scripts/run_robustness.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_settings, load_model_config, PROJECT_ROOT
from src.utils import setup_logging, ensure_output_dirs
from src.robustness import (
    compute_tc_sensitivity,
    compute_subperiod_performance,
    compute_strategy_ranking,
)
from src.visualization import (
    plot_tc_sensitivity,
    plot_subperiod_performance,
    plot_strategy_ranking,
)


def main() -> None:
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = load_settings()

    data_dir   = PROJECT_ROOT / settings.outputs.data_dir
    report_dir = PROJECT_ROOT / settings.outputs.report_dir
    chart_dir  = PROJECT_ROOT / settings.outputs.chart_dir
    ensure_output_dirs(data_dir, report_dir, chart_dir)

    daily_path  = report_dir / "backtest_daily_returns.csv"
    metrics_cmp = report_dir / "backtest_metrics_comparison.csv"

    for p in (daily_path, metrics_cmp):
        if not p.exists():
            logger.error(
                "Required input not found: %s\n"
                "Run 'python scripts/run_backtest.py' first.", p
            )
            sys.exit(1)

    logger.info("Loading backtest outputs …")
    daily_df       = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    metrics_cmp_df = pd.read_csv(metrics_cmp)

    # ── Check 1: Transaction cost sensitivity ─────────────────────────────
    logger.info("Running TC sensitivity check (0, 5, 10, 25, 50 bps) …")
    tc_df = compute_tc_sensitivity(daily_df, tc_levels_bps=[0, 5, 10, 25, 50])
    tc_path = report_dir / "transaction_cost_sensitivity.csv"
    tc_df.to_csv(tc_path, index=False)
    logger.info("TC sensitivity saved → %s  shape=%s", tc_path, tc_df.shape)

    # ── Check 2: Subperiod performance ────────────────────────────────────
    logger.info("Running subperiod analysis …")
    sp_df = compute_subperiod_performance(daily_df)
    sp_path = report_dir / "subperiod_performance.csv"
    sp_df.to_csv(sp_path, index=False)
    logger.info("Subperiod performance saved → %s  shape=%s", sp_path, sp_df.shape)

    # ── Check 3: Strategy ranking ─────────────────────────────────────────
    logger.info("Running strategy ranking …")
    rank_df = compute_strategy_ranking(metrics_cmp_df)
    rank_path = report_dir / "strategy_ranking_by_objective.csv"
    rank_df.to_csv(rank_path, index=False)
    logger.info("Strategy ranking saved → %s  shape=%s", rank_path, rank_df.shape)

    # ── Charts 22–24 ──────────────────────────────────────────────────────
    chart22 = chart_dir / "22_transaction_cost_sensitivity.png"
    chart23 = chart_dir / "23_subperiod_performance.png"
    chart24 = chart_dir / "24_strategy_ranking.png"

    logger.info("Generating chart 22: TC sensitivity …")
    try:
        plot_tc_sensitivity(tc_df, chart22)
    except Exception as exc:
        logger.error("Chart 22 failed: %s", exc, exc_info=True)

    logger.info("Generating chart 23: subperiod performance …")
    try:
        plot_subperiod_performance(sp_df, chart23)
    except Exception as exc:
        logger.error("Chart 23 failed: %s", exc, exc_info=True)

    logger.info("Generating chart 24: strategy ranking …")
    try:
        plot_strategy_ranking(rank_df, chart24)
    except Exception as exc:
        logger.error("Chart 24 failed: %s", exc, exc_info=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n  Transaction cost sensitivity (Sharpe at 5 bps baseline):")
    print(f"  {'Strategy':<25}  {'Sharpe @0bps':>12}  {'Sharpe @5bps':>12}  "
          f"{'Sharpe @25bps':>13}  {'Sharpe @50bps':>13}")
    print("  " + "─" * 78)
    for strat in tc_df["strategy"].unique():
        sub = tc_df[tc_df["strategy"] == strat].set_index("tc_bps")["sharpe"]
        s0  = f"{sub.get(0,  float('nan')):.3f}" if not pd.isna(sub.get(0,  float('nan'))) else "n/a"
        s5  = f"{sub.get(5,  float('nan')):.3f}" if not pd.isna(sub.get(5,  float('nan'))) else "n/a"
        s25 = f"{sub.get(25, float('nan')):.3f}" if not pd.isna(sub.get(25, float('nan'))) else "n/a"
        s50 = f"{sub.get(50, float('nan')):.3f}" if not pd.isna(sub.get(50, float('nan'))) else "n/a"
        print(f"  {strat:<25}  {s0:>12}  {s5:>12}  {s25:>13}  {s50:>13}")

    print("\n  Strategy overall ranking (1 = best):")
    if "overall_rank" in rank_df.columns:
        for _, row in rank_df.sort_values("overall_rank").iterrows():
            print(f"    #{int(row['overall_rank']):.0f}  {row['strategy']}")

    print(f"\n  Charts saved:")
    for ch in [chart22, chart23, chart24]:
        exists = "✓" if ch.exists() else "✗"
        print(f"    {exists}  {ch.name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
