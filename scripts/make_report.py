#!/usr/bin/env python3
"""Generate the latest market regime Markdown report."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import load_settings, load_model_config, PROJECT_ROOT
from src.utils import setup_logging, ensure_output_dirs
from src.reporting import generate_latest_report, save_report


def main() -> None:
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings  = load_settings()
    model_cfg = load_model_config()

    data_dir   = PROJECT_ROOT / settings.outputs.data_dir
    report_dir = PROJECT_ROOT / settings.outputs.report_dir
    ensure_output_dirs(data_dir, report_dir)

    pred_path     = report_dir / "walk_forward_regime_predictions.csv"
    daily_path    = report_dir / "backtest_daily_returns.csv"
    metrics_path  = report_dir / "backtest_metrics.csv"
    stress_path   = report_dir / "stress_period_analysis.csv"
    features_path = data_dir   / "features_dataset.csv"

    for p in (pred_path, daily_path, metrics_path, stress_path, features_path):
        if not p.exists():
            logger.error("Required input not found: %s", p)
            sys.exit(1)

    predictions  = pd.read_csv(pred_path,     index_col=0, parse_dates=True)
    daily        = pd.read_csv(daily_path,    index_col=0, parse_dates=True)
    metrics_df   = pd.read_csv(metrics_path)
    stress_df    = pd.read_csv(stress_path)
    features     = pd.read_csv(features_path, index_col=0, parse_dates=True)

    defensive_label = model_cfg.get("risk_overlay", {}).get("defensive_asset", "SHY.O")

    content = generate_latest_report(
        predictions=predictions,
        daily_returns=daily,
        metrics_df=metrics_df,
        stress_df=stress_df,
        features=features,
        defensive_label=defensive_label,
    )

    out_path = report_dir / "latest_market_regime_report.md"
    save_report(content, out_path)
    print(f"\nReport saved → {out_path}\n")


if __name__ == "__main__":
    main()
