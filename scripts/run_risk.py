#!/usr/bin/env python3
"""Run Phase 4 professional risk layer.

Reads:
  output/reports/backtest_daily_returns.csv

Writes:
  output/reports/risk_controlled_returns.csv
  output/reports/cvar_by_regime.csv

Generates:
  output/charts/27_cvar_by_regime.png/svg
  output/charts/28_risk_controlled_overlay.png/svg

Usage:
    python scripts/run_risk.py
    python scripts/run_risk.py --target-vol 0.10
    python scripts/run_risk.py --drawdown-threshold -0.15
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
from src.risk import compute_cvar_by_regime, build_risk_controlled_returns
from src.visualization import plot_cvar_by_regime, plot_risk_controlled_overlay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply professional risk controls to the regime overlay strategies."
    )
    parser.add_argument(
        "--target-vol", type=float, default=0.08,
        help="Annualised vol target for vol-targeting control (default: 0.08 = 8%%).",
    )
    parser.add_argument(
        "--lookback", type=int, default=20,
        help="Lookback window (days) for realised vol estimation (default: 20).",
    )
    parser.add_argument(
        "--drawdown-threshold", type=float, default=-0.10,
        help="Drawdown level that triggers the stop-loss (default: -0.10 = -10%%).",
    )
    parser.add_argument(
        "--recovery-threshold", type=float, default=-0.05,
        help="Drawdown level at which stop-loss is lifted (default: -0.05 = -5%%).",
    )
    parser.add_argument(
        "--defensive-floor", type=float, default=0.15,
        help="Minimum equity weight when stop-loss is active (default: 0.15 = 15%%).",
    )
    parser.add_argument(
        "--max-daily-change", type=float, default=0.10,
        help="Turnover cap: max daily equity-weight change (default: 0.10 = 10pp).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings   = load_settings()
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

    # ── CVaR by regime ────────────────────────────────────────────────────────
    logger.info("Computing CVaR by regime …")
    cvar_df = compute_cvar_by_regime(daily_df, confidence_levels=[0.95, 0.99])

    cvar_path = report_dir / "cvar_by_regime.csv"
    cvar_df.to_csv(cvar_path, index=False)
    logger.info("CVaR saved  → %s  shape=%s", cvar_path, cvar_df.shape)

    # ── Risk-controlled returns ───────────────────────────────────────────────
    logger.info(
        "Building risk-controlled returns  "
        "target_vol=%.0f%%  drawdown_threshold=%.0f%%  recovery_threshold=%.0f%%  "
        "defensive_floor=%.0f%%  max_daily_change=%.0f%%  lookback=%d …",
        args.target_vol * 100, args.drawdown_threshold * 100,
        args.recovery_threshold * 100, args.defensive_floor * 100,
        args.max_daily_change * 100, args.lookback,
    )
    rc_df = build_risk_controlled_returns(
        daily_df,
        target_vol          = args.target_vol,
        lookback            = args.lookback,
        drawdown_threshold  = args.drawdown_threshold,
        recovery_threshold  = args.recovery_threshold,
        defensive_floor     = args.defensive_floor,
        max_daily_change    = args.max_daily_change,
    )

    rc_path = report_dir / "risk_controlled_returns.csv"
    rc_df.to_csv(rc_path)
    logger.info("Risk-controlled returns saved → %s  shape=%s", rc_path, rc_df.shape)

    # ── Charts ────────────────────────────────────────────────────────────────
    chart27 = chart_dir / "27_cvar_by_regime.png"
    chart28 = chart_dir / "28_risk_controlled_overlay.png"

    logger.info("Generating chart 27: CVaR by regime …")
    try:
        plot_cvar_by_regime(cvar_df, chart27)
    except Exception as exc:
        logger.error("Chart 27 failed: %s", exc, exc_info=True)

    logger.info("Generating chart 28: risk-controlled overlay …")
    try:
        plot_risk_controlled_overlay(rc_df, chart28)
    except Exception as exc:
        logger.error("Chart 28 failed: %s", exc, exc_info=True)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RISK LAYER SUMMARY")
    print("=" * 72)

    print(f"\n  CVaR by Regime  (95% confidence, daily returns)")
    df95 = cvar_df[cvar_df["confidence"] == 0.95]
    print(f"  {'Strategy':<25}  {'Regime':<12}  {'VaR':>7}  {'CVaR':>7}  {'N':>5}")
    print("  " + "─" * 60)
    for _, row in df95.iterrows():
        print(f"  {row['strategy']:<25}  {row['regime']:<12}  "
              f"{row['var']*100:>6.3f}%  {row['cvar']*100:>6.3f}%  {int(row['n_obs']):>5}")

    rc_cols = [c for c in rc_df.columns if c.startswith("rc_") and c.endswith("_ret")]
    if rc_cols:
        from src.metrics import TRADING_DAYS
        import numpy as np
        print(f"\n  Risk-Controlled Return Statistics  (annualised)")
        print(f"  {'Column':<30}  {'CAGR':>7}  {'Vol':>7}  {'Sharpe':>7}")
        print("  " + "─" * 55)
        for col in rc_cols:
            rets = rc_df[col].dropna()
            cagr  = float((1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1) * 100
            vol   = float(rets.std() * np.sqrt(TRADING_DAYS)) * 100
            sharpe = cagr / vol if vol > 0 else 0.0
            print(f"  {col:<30}  {cagr:>6.2f}%  {vol:>6.2f}%  {sharpe:>7.3f}")

        # Compare original defensive overlay
        ov_rets = rc_df["overlay_ret"].dropna()
        cagr_ov  = float((1 + ov_rets).prod() ** (TRADING_DAYS / len(ov_rets)) - 1) * 100
        vol_ov   = float(ov_rets.std() * np.sqrt(TRADING_DAYS)) * 100
        sharpe_ov = cagr_ov / vol_ov if vol_ov > 0 else 0.0
        print(f"  {'overlay_ret (original)':<30}  {cagr_ov:>6.2f}%  {vol_ov:>6.2f}%  "
              f"{sharpe_ov:>7.3f}")

    stop_col = "rc_def_stop_active"
    if stop_col in rc_df.columns:
        pct_stopped = rc_df[stop_col].mean() * 100
        print(f"\n  Stop-loss active:  {pct_stopped:.1f}% of trading days")

    print(f"\n  Charts saved:")
    for ch in [chart27, chart28]:
        mark = "✓" if ch.exists() else "✗"
        print(f"    {mark}  {ch.name}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
