#!/usr/bin/env python3
"""Run Phase 3 statistical robustness checks.

Reads:
  output/reports/backtest_daily_returns.csv

Writes:
  output/reports/deflated_sharpe_ratios.csv
  output/reports/bootstrap_confidence_intervals.csv

Generates:
  output/charts/25_bootstrap_sharpe_ci.png/svg
  output/charts/26_deflated_sharpe_ratio.png/svg

Usage:
    python scripts/run_stats.py
    python scripts/run_stats.py --n-bootstrap 5000   # faster for testing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_settings, PROJECT_ROOT
from src.utils import setup_logging, ensure_output_dirs
from src.stats import compute_all_stats
from src.visualization import plot_bootstrap_sharpe_ci, plot_deflated_sharpe_ratio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 statistical robustness checks."
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10 000).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of strategies tested (used in DSR multiple-comparison correction; default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = load_settings()
    report_dir = PROJECT_ROOT / settings.outputs.report_dir
    chart_dir  = PROJECT_ROOT / settings.outputs.chart_dir
    ensure_output_dirs(report_dir, chart_dir)

    daily_path = report_dir / "backtest_daily_returns.csv"
    if not daily_path.exists():
        logger.error(
            "backtest_daily_returns.csv not found at %s\n"
            "Run 'python scripts/run_backtest.py' first.", daily_path
        )
        sys.exit(1)

    logger.info("Loading daily returns …")
    daily_df = pd.read_csv(daily_path, index_col=0, parse_dates=True)

    logger.info(
        "Running statistical robustness checks  n_trials=%d  n_bootstrap=%d …",
        args.n_trials, args.n_bootstrap,
    )
    dsr_df, boot_df = compute_all_stats(
        daily_df,
        n_trials=args.n_trials,
        n_bootstrap=args.n_bootstrap,
    )

    dsr_path  = report_dir / "deflated_sharpe_ratios.csv"
    boot_path = report_dir / "bootstrap_confidence_intervals.csv"

    dsr_df.to_csv(dsr_path,  index=False)
    boot_df.to_csv(boot_path, index=False)
    logger.info("DSR saved           → %s  shape=%s", dsr_path,  dsr_df.shape)
    logger.info("Bootstrap CI saved  → %s  shape=%s", boot_path, boot_df.shape)

    chart25 = chart_dir / "25_bootstrap_sharpe_ci.png"
    chart26 = chart_dir / "26_deflated_sharpe_ratio.png"

    logger.info("Generating chart 25: bootstrap Sharpe CI …")
    try:
        plot_bootstrap_sharpe_ci(boot_df, chart25)
    except Exception as exc:
        logger.error("Chart 25 failed: %s", exc, exc_info=True)

    logger.info("Generating chart 26: deflated Sharpe ratio …")
    try:
        plot_deflated_sharpe_ratio(dsr_df, chart26)
    except Exception as exc:
        logger.error("Chart 26 failed: %s", exc, exc_info=True)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STATISTICAL ROBUSTNESS SUMMARY")
    print("=" * 72)
    print(f"\n  Deflated Sharpe Ratio  (N={args.n_trials} strategies, Bailey & LdP 2014)")
    print(f"  {'Strategy':<25}  {'Ann. Sharpe':>12}  {'Skew':>6}  "
          f"{'Ex. Kurt':>9}  {'DSR':>6}")
    print("  " + "─" * 62)
    for _, row in dsr_df.iterrows():
        dsr_fmt = f"{row['dsr']:.3f}" if not pd.isna(row["dsr"]) else "n/a"
        sr_fmt  = f"{row['sr_annualized']:.3f}" if not pd.isna(row["sr_annualized"]) else "n/a"
        sk_fmt  = f"{row['skewness']:.2f}" if not pd.isna(row["skewness"]) else "n/a"
        ku_fmt  = f"{row['excess_kurtosis']:.2f}" if not pd.isna(row["excess_kurtosis"]) else "n/a"
        print(f"  {row['strategy']:<25}  {sr_fmt:>12}  {sk_fmt:>6}  {ku_fmt:>9}  {dsr_fmt:>6}")

    print(f"\n  Bootstrap Sharpe — 90% CI  (n_bootstrap={args.n_bootstrap})")
    print(f"  {'Strategy':<25}  {'p5':>7}  {'p50':>7}  {'p95':>7}  {'Width':>7}")
    print("  " + "─" * 58)
    for _, row in boot_df.iterrows():
        p5  = row["sharpe_p5"]
        p50 = row["sharpe_p50"]
        p95 = row["sharpe_p95"]
        width = p95 - p5
        print(f"  {row['strategy']:<25}  {p5:>7.3f}  {p50:>7.3f}  {p95:>7.3f}  {width:>7.3f}")

    print(f"\n  Charts saved:")
    for ch in [chart25, chart26]:
        mark = "✓" if ch.exists() else "✗"
        print(f"    {mark}  {ch.name}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
