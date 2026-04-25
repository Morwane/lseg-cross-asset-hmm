#!/usr/bin/env python3
"""Build the cross-asset feature matrix from the cleaned daily panel.

Reads output/data/cleaned_dataset.csv and writes:
  output/data/features_dataset.csv   — full feature matrix
  output/audit/features_summary.csv  — per-column statistics

Usage:
    python scripts/build_features.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings, PROJECT_ROOT
from src.features import build_all_features, load_cleaned_panel, save_features
from src.utils import setup_logging, ensure_output_dirs


def _print_summary(features, features_path: Path, summary_path: Path) -> None:
    width = 70
    print("\n" + "=" * width)
    print("  FEATURE BUILD SUMMARY")
    print("=" * width)
    print(f"  Features : {features.shape[0]:>6} rows × {features.shape[1]:>2} cols")
    print(f"  Range    : {features.index[0].date()} → {features.index[-1].date()}")
    n_complete = int(features.dropna().shape[0])
    print(f"  Complete : {n_complete} rows  (warm-up: {len(features) - n_complete} rows)")
    print()
    print(f"  {'Feature':<35}  {'Missing %':>9}  {'Mean':>12}  {'Std':>12}")
    print("  " + "-" * 72)
    for col in features.columns:
        s = features[col].dropna()
        miss_pct = features[col].isna().mean() * 100
        mean = float(s.mean()) if not s.empty else float("nan")
        std = float(s.std()) if not s.empty else float("nan")
        print(f"  {col:<35}  {miss_pct:>8.1f}%  {mean:>12.4f}  {std:>12.4f}")
    print("=" * width)
    print(f"  Saved → {features_path}")
    print(f"  Saved → {summary_path}")
    print("=" * width + "\n")


def main() -> None:
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = load_settings()
    data_dir = PROJECT_ROOT / settings.outputs.data_dir
    audit_dir = PROJECT_ROOT / settings.outputs.audit_dir
    ensure_output_dirs(data_dir, audit_dir)

    cleaned_path = data_dir / "cleaned_dataset.csv"
    if not cleaned_path.exists():
        logger.error(
            "Cleaned dataset not found at %s.\n"
            "Run 'python scripts/build_dataset.py' first.",
            cleaned_path,
        )
        sys.exit(1)

    logger.info("Loading cleaned panel from %s", cleaned_path)
    panel = load_cleaned_panel(cleaned_path)
    logger.info("Panel shape: %s  date_range=[%s → %s]",
                panel.shape, panel.index[0].date(), panel.index[-1].date())

    features = build_all_features(panel)

    if features.empty:
        logger.error("Feature build produced an empty DataFrame — check cleaned panel columns.")
        sys.exit(1)

    features_path = data_dir / "features_dataset.csv"
    summary_path = audit_dir / "features_summary.csv"
    save_features(features, features_path, summary_path)

    _print_summary(features, features_path, summary_path)


if __name__ == "__main__":
    main()
