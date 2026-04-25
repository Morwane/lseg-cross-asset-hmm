#!/usr/bin/env python3
"""Run the regime risk-overlay backtest.

Reads:
  output/reports/walk_forward_regime_predictions.csv  — OOS regime signals
  output/data/cleaned_dataset.csv                     — equity and defensive returns

Writes:
  output/reports/backtest_daily_returns.csv
  output/reports/backtest_metrics.csv
  output/reports/stress_period_analysis.csv

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --strategy risk_overlay
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings, load_model_config, PROJECT_ROOT
from src.backtest import run_backtest
from src.pnl import compute_economic_impact
from src.utils import setup_logging, ensure_output_dirs

import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HMM regime risk-overlay backtest."
    )
    parser.add_argument(
        "--strategy",
        choices=["risk_overlay"],
        default="risk_overlay",
        help="Backtest strategy (default: risk_overlay).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Console summary helpers
# ---------------------------------------------------------------------------


def _pct(v: float, decimals: int = 1) -> str:
    """Format a fraction as a percentage string, or 'n/a' for NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v * 100:.{decimals}f}%"


def _fmt(v: float, decimals: int = 2) -> str:
    """Format a float to given decimals, or 'n/a' for NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


def _print_summary(
    daily_df,
    metrics_df,
    stress_df,
    defensive_label: str,
    output_paths: dict,
) -> None:
    width = 76
    print("\n" + "=" * width)
    print("  BACKTEST SUMMARY")
    print("=" * width)
    print(f"  Strategy        : Risk Overlay (1-day lagged regime signal)")
    print(f"  Benchmark       : 100% US equity (us_equity, buy-and-hold)")
    print(f"  Defensive asset : {defensive_label}")
    print(f"  Backtest period : {daily_df.index[0].date()} → {daily_df.index[-1].date()}")
    print(f"  Trading days    : {len(daily_df)}")
    print(f"  Signal source   : walk_forward_regime_predictions.csv (OOS only)")
    print()

    metric_labels = {
        "cagr":                  ("CAGR",                         _pct),
        "annualised_vol":        ("Annualised Vol",                _pct),
        "sharpe":                ("Sharpe Ratio",                  lambda v: _fmt(v)),
        "sortino":               ("Sortino Ratio",                 lambda v: _fmt(v)),
        "max_drawdown":          ("Max Drawdown",                  _pct),
        "calmar":                ("Calmar Ratio",                  lambda v: _fmt(v)),
        "total_return":          ("Total Return",                  _pct),
        "hit_rate":              ("Hit Rate (positive days)",       _pct),
        "worst_daily_loss":      ("Worst Daily Loss",              _pct),
        "avg_equity_exposure":   ("Avg Equity Exposure",           _pct),
        "n_regime_switches":     ("Regime Switches",               lambda v: str(int(v)) if not np.isnan(float(v)) else "n/a"),
        "total_transaction_cost":("Total Transaction Cost",        _pct),
    }

    m = metrics_df.set_index("metric")
    print(f"  {'Metric':<30}  {'Benchmark':>12}  {'Overlay':>12}")
    print("  " + "─" * 58)
    for key, (label, fmt) in metric_labels.items():
        if key not in m.index:
            continue
        bench_val = float(m.loc[key, "benchmark"])
        ov_val = float(m.loc[key, "overlay"])
        print(f"  {label:<30}  {fmt(bench_val):>12}  {fmt(ov_val):>12}")

    print()
    print("  REGIME BREAKDOWN (1-day lagged signal applied)")
    counts = daily_df["regime_label"].value_counts().sort_index()
    print(f"  {'Label':<18}  {'Days':>6}  {'Freq %':>7}")
    print("  " + "─" * 35)
    for lbl, n in counts.items():
        print(f"  {lbl:<18}  {n:>6}  {n / len(daily_df) * 100:>6.1f}%")

    print()
    print("  STRESS PERIOD ANALYSIS")
    print(f"  {'Period':<28}  {'Coverage':<34}  {'Bench':>7}  {'Overlay':>8}")
    print("  " + "─" * 84)
    for _, row in stress_df.iterrows():
        bench_r = _pct(row["benchmark_total_ret"])
        ov_r = _pct(row["overlay_total_ret"])
        print(f"  {row['period']:<28}  {str(row['coverage']):<34}  {bench_r:>7}  {ov_r:>8}")

    print()
    print("  OUTPUT FILES")
    for name, path in output_paths.items():
        print(f"  {name:<40} → {path}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _parse_args()
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = load_settings()
    model_cfg = load_model_config()

    data_dir = PROJECT_ROOT / settings.outputs.data_dir
    report_dir = PROJECT_ROOT / settings.outputs.report_dir
    ensure_output_dirs(data_dir, report_dir)

    predictions_path = report_dir / "walk_forward_regime_predictions.csv"
    cleaned_path = data_dir / "cleaned_dataset.csv"

    if not predictions_path.exists():
        logger.error(
            "Walk-forward predictions not found at %s.\n"
            "Run 'python scripts/train_hmm.py --mode walk_forward' first.",
            predictions_path,
        )
        sys.exit(1)

    if not cleaned_path.exists():
        logger.error(
            "Cleaned dataset not found at %s.\n"
            "Run 'python scripts/build_dataset.py' first.",
            cleaned_path,
        )
        sys.exit(1)

    result = run_backtest(
        predictions_path=predictions_path,
        cleaned_path=cleaned_path,
        model_cfg=model_cfg,
        transaction_cost_bps=settings.backtest.transaction_cost_bps,
        trading_days=settings.backtest.trading_days,
    )

    daily_df     = result["daily"]
    metrics_df   = result["metrics"]
    stress_df    = result["stress"]
    defensive_label = result["defensive_label"]

    daily_path    = report_dir / "backtest_daily_returns.csv"
    metrics_path  = report_dir / "backtest_metrics.csv"
    stress_path   = report_dir / "stress_period_analysis.csv"
    cmp_path      = report_dir / "backtest_metrics_comparison.csv"
    stress_cmp    = report_dir / "stress_period_comparison.csv"
    bull_part     = report_dir / "bull_market_participation.csv"
    exposure_path = report_dir / "strategy_exposure_summary.csv"
    eco_path      = report_dir / "economic_impact_100k.csv"

    daily_df.to_csv(daily_path)
    logger.info("Daily returns saved → %s  shape=%s", daily_path, daily_df.shape)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved → %s", metrics_path)
    stress_df.to_csv(stress_path, index=False)
    logger.info("Stress analysis saved → %s", stress_path)
    result["metrics_comparison"].to_csv(cmp_path, index=False)
    logger.info("Metrics comparison saved → %s", cmp_path)
    result["stress_comparison"].to_csv(stress_cmp, index=False)
    logger.info("Stress comparison saved → %s", stress_cmp)
    result["bull_market_participation"].to_csv(bull_part, index=False)
    logger.info("Bull-market participation saved → %s", bull_part)
    result["strategy_exposure_summary"].to_csv(exposure_path, index=False)
    logger.info("Exposure summary saved → %s", exposure_path)

    eco_df = compute_economic_impact(daily_df, result["metrics_comparison"])
    eco_df.to_csv(eco_path, index=False)
    logger.info("Economic impact saved → %s", eco_path)

    output_paths = {
        "backtest_daily_returns.csv":         daily_path,
        "backtest_metrics.csv":               metrics_path,
        "backtest_metrics_comparison.csv":    cmp_path,
        "stress_period_analysis.csv":         stress_path,
        "stress_period_comparison.csv":       stress_cmp,
        "bull_market_participation.csv":      bull_part,
        "strategy_exposure_summary.csv":      exposure_path,
        "economic_impact_100k.csv":           eco_path,
    }
    _print_summary(daily_df, metrics_df, stress_df, defensive_label, output_paths)


if __name__ == "__main__":
    main()
