#!/usr/bin/env python3
"""Generate all charts (01-24) and save to output/charts/."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings, load_model_config, PROJECT_ROOT
from src.utils import setup_logging, ensure_output_dirs
from src.visualization import generate_all_charts


def main() -> None:
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings  = load_settings()
    model_cfg = load_model_config()

    data_dir   = PROJECT_ROOT / settings.outputs.data_dir
    report_dir = PROJECT_ROOT / settings.outputs.report_dir
    chart_dir  = PROJECT_ROOT / settings.outputs.chart_dir
    ensure_output_dirs(data_dir, report_dir, chart_dir)

    panel_path    = data_dir   / "cleaned_dataset.csv"
    features_path = data_dir   / "features_dataset.csv"
    pred_path     = report_dir / "walk_forward_regime_predictions.csv"
    daily_path    = report_dir / "backtest_daily_returns.csv"
    metrics_path  = report_dir / "backtest_metrics.csv"

    for p in (panel_path, features_path, pred_path, daily_path, metrics_path):
        if not p.exists():
            logger.error("Required input not found: %s", p)
            sys.exit(1)

    metrics_cmp_path  = report_dir / "backtest_metrics_comparison.csv"
    bull_part_path    = report_dir / "bull_market_participation.csv"
    eco_path          = report_dir / "economic_impact_100k.csv"
    tc_path           = report_dir / "transaction_cost_sensitivity.csv"
    subperiod_path    = report_dir / "subperiod_performance.csv"
    ranking_path      = report_dir / "strategy_ranking_by_objective.csv"

    logger.info("Generating all charts → %s", chart_dir)
    saved = generate_all_charts(
        panel_path=panel_path,
        features_path=features_path,
        predictions_path=pred_path,
        daily_returns_path=daily_path,
        metrics_path=metrics_path,
        charts_dir=chart_dir,
        model_cfg=model_cfg,
        metrics_comparison_path=metrics_cmp_path,
        bull_participation_path=bull_part_path,
        economic_impact_path=eco_path,
        tc_sensitivity_path=tc_path,
        subperiod_path=subperiod_path,
        strategy_ranking_path=ranking_path,
    )

    print(f"\nGenerated {len(saved)} chart(s):")
    for p in saved:
        print(f"  {p}")
    print()


if __name__ == "__main__":
    main()
