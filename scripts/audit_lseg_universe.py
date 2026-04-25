#!/usr/bin/env python3
"""Audit all configured LSEG instruments and save output/audit/validated_universe.csv.

Usage:
    python scripts/audit_lseg_universe.py
    python scripts/audit_lseg_universe.py --window-days 90

Prerequisites:
    LSEG Workspace desktop application must be running before invoking this script.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings, load_instruments_config, PROJECT_ROOT
from src.session import open_lseg_session, close_lseg_session
from src.universe import audit_full_universe, select_validated_universe, save_validated_universe
from src.utils import setup_logging, ensure_output_dirs


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit all LSEG instrument candidates and save a validated universe CSV."
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=60,
        help="Days of recent history to use for the short audit check (default: 60).",
    )
    return parser.parse_args()


def _print_summary(results: list) -> None:
    """Print a human-readable audit summary table grouped by category."""
    width = 72
    print("\n" + "=" * width)
    print("  LSEG UNIVERSE AUDIT SUMMARY")
    print("=" * width)

    # Group by category
    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for category, items in categories.items():
        print(f"\n  [{category.upper()}]")
        for r in items:
            selected_tag = "  ← SELECTED" if r.selected else ""
            value_str = f"{r.last_value:.4f}" if r.last_value is not None else "N/A    "
            status_icon = {"OK": "✓", "UNAVAILABLE": "✗", "ERROR": "!"}.get(r.status, "?")
            print(
                f"    {status_icon} {r.ric:<20}  {r.status:<12}"
                f"  last={value_str:<10}  rows={r.rows_available:<5}"
                f"  {r.verdict}{selected_tag}"
            )
            if r.error_message:
                print(f"      └─ {r.error_message}")

    ok_count = sum(1 for r in results if r.status == "OK")
    unavail_count = sum(1 for r in results if r.status == "UNAVAILABLE")
    error_count = sum(1 for r in results if r.status == "ERROR")
    selected_count = sum(1 for r in results if r.selected)

    print("\n" + "-" * width)
    print(
        f"  Audited: {len(results)}   "
        f"OK: {ok_count}   "
        f"UNAVAILABLE: {unavail_count}   "
        f"ERROR: {error_count}"
    )
    print(f"  Selected instruments: {selected_count}")
    print("=" * width + "\n")


def main() -> None:
    """Run the LSEG universe audit end-to-end."""
    args = _parse_args()
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = load_settings()
    instruments_cfg = load_instruments_config()

    audit_path = PROJECT_ROOT / settings.outputs.audit_dir / "validated_universe.csv"

    ensure_output_dirs(
        PROJECT_ROOT / settings.outputs.audit_dir,
        PROJECT_ROOT / settings.outputs.data_dir,
        PROJECT_ROOT / settings.outputs.chart_dir,
        PROJECT_ROOT / settings.outputs.report_dir,
    )

    logger.info("Opening LSEG session: %s", settings.lseg.session_name)

    try:
        open_lseg_session(settings.lseg.session_name)

        logger.info("Starting universe audit (window=%d days) ...", args.window_days)
        results = audit_full_universe(instruments_cfg, window_days=args.window_days)

        _print_summary(results)

        selected = select_validated_universe(results)
        logger.info(
            "Selected %d instruments: %s",
            len(selected),
            ", ".join(selected.keys()),
        )

        save_validated_universe(results, audit_path)
        print(f"Audit saved → {audit_path}")

        # Exit with a non-zero code if any core block has no selected instrument
        core_blocks = {"us_equity", "vix", "us_2y", "us_5y", "us_10y", "us_30y"}
        missing_core = core_blocks - set(selected.keys())
        if missing_core:
            logger.warning(
                "Core blocks without a valid instrument: %s. "
                "Downstream steps will run in degraded mode.",
                missing_core,
            )

    except KeyboardInterrupt:
        logger.warning("Audit interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Audit failed with unexpected error: %s", exc)
        sys.exit(1)
    finally:
        close_lseg_session()


if __name__ == "__main__":
    main()
