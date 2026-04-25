#!/usr/bin/env python3
"""Retrieve full historical data for the validated universe and build the analysis dataset.

Reads output/audit/validated_universe.csv to determine which instruments to use.
Saves output/data/raw_dataset.csv and output/data/cleaned_dataset.csv.
Requires LSEG Workspace to be running.

Usage:
    python scripts/build_dataset.py --start 2019-04-25
    python scripts/build_dataset.py --start 2019-04-25 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings, PROJECT_ROOT
from src.preprocessing import (
    load_validated_universe,
    build_raw_panel,
    clean_panel,
    save_dataset_summary,
)
from src.session import open_lseg_session, close_lseg_session
from src.utils import setup_logging, ensure_output_dirs


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the cross-asset historical dataset from the validated LSEG universe."
    )
    parser.add_argument(
        "--start",
        required=True,
        help="History start date in YYYY-MM-DD format (e.g. 2019-04-25).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="History end date in YYYY-MM-DD format (default: today).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(raw, cleaned, data_dir: Path) -> None:
    """Print a human-readable dataset build summary to stdout."""
    width = 70
    print("\n" + "=" * width)
    print("  DATASET BUILD SUMMARY")
    print("=" * width)
    print(f"  Raw   : {raw.shape[0]:>6} rows × {raw.shape[1]:>2} cols"
          f"  [{raw.index[0].date()} → {raw.index[-1].date()}]")
    print(f"  Clean : {cleaned.shape[0]:>6} rows × {cleaned.shape[1]:>2} cols"
          f"  [{cleaned.index[0].date()} → {cleaned.index[-1].date()}]")
    print()
    print(f"  {'Column':<28}  {'Valid rows':>10}  {'Missing %':>9}  {'Last value':>12}")
    print("  " + "-" * 64)
    for col in cleaned.columns:
        s = cleaned[col]
        n_valid = int(s.notna().sum())
        miss_pct = s.isna().mean() * 100
        last_val = s.dropna().iloc[-1] if not s.dropna().empty else float("nan")
        print(f"  {col:<28}  {n_valid:>10}  {miss_pct:>8.1f}%  {last_val:>12.4f}")
    print("=" * width)
    print(f"  Saved → {data_dir / 'raw_dataset.csv'}")
    print(f"  Saved → {data_dir / 'cleaned_dataset.csv'}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch, clean, and save the cross-asset daily dataset."""
    args = _parse_args()
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = load_settings()

    start_date: str = args.start
    end_date: str = args.end or date.today().isoformat()

    # Validate date strings
    try:
        from datetime import date as _date
        _date.fromisoformat(start_date)
        _date.fromisoformat(end_date)
    except ValueError as exc:
        logger.error("Invalid date argument: %s", exc)
        sys.exit(1)

    # Paths
    audit_path = PROJECT_ROOT / settings.outputs.audit_dir / "validated_universe.csv"
    data_dir = PROJECT_ROOT / settings.outputs.data_dir
    audit_dir = PROJECT_ROOT / settings.outputs.audit_dir

    # Guard: audit must have been run first
    if not audit_path.exists():
        logger.error(
            "Validated universe not found at %s.\n"
            "Run 'python scripts/audit_lseg_universe.py' first.",
            audit_path,
        )
        sys.exit(1)

    ensure_output_dirs(data_dir, audit_dir)

    # Load universe
    selected = load_validated_universe(audit_path)
    if selected.empty:
        logger.error("No selected instruments in %s — run the audit script first.", audit_path)
        sys.exit(1)

    logger.info(
        "Building dataset: %d instruments  start=%s  end=%s",
        len(selected), start_date, end_date,
    )

    # Fetch and build
    try:
        open_lseg_session(settings.lseg.session_name)

        logger.info("--- Fetching historical data ---")
        raw = build_raw_panel(selected, start_date, end_date)

        if raw.empty:
            logger.error("Raw panel is empty — no data was retrieved. Exiting.")
            sys.exit(1)

        raw_path = data_dir / "raw_dataset.csv"
        raw.to_csv(raw_path)
        logger.info("Raw dataset saved → %s  shape=%s", raw_path, raw.shape)

        logger.info("--- Cleaning panel ---")
        cleaned = clean_panel(raw)

        if cleaned.empty:
            logger.error("Cleaned panel is empty after processing. Exiting.")
            sys.exit(1)

        cleaned_path = data_dir / "cleaned_dataset.csv"
        cleaned.to_csv(cleaned_path)
        logger.info("Cleaned dataset saved → %s  shape=%s", cleaned_path, cleaned.shape)

        # Dataset quality summary
        summary_path = audit_dir / "dataset_summary.csv"
        save_dataset_summary(raw, cleaned, summary_path)

        _print_summary(raw, cleaned, data_dir)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Dataset build failed with unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        close_lseg_session()


if __name__ == "__main__":
    main()
