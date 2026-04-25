#!/usr/bin/env python3
"""Debug script: capture the raw ld.get_data() response structure for TR.MIDYIELD.

Run once to understand the exact DataFrame shape returned by your LSEG environment.
Output is saved to output/audit/debug_tr_midyield_raw_US10YT_RR.csv.

Usage:
    python scripts/debug_tr_midyield.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings, PROJECT_ROOT
from src.session import open_lseg_session, close_lseg_session
from src.utils import setup_logging, ensure_output_dirs

RIC = "US10YT=RR"
FIELD = "TR.MIDYIELD"
START = "2024-10-01"
END = "2024-12-31"


def _describe_dataframe(df, label: str) -> None:
    """Print full structural description of a DataFrame to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  type(df)          : {type(df)}")
    if df is None:
        print("  df is None — nothing to inspect.")
        return
    print(f"  df.shape          : {df.shape}")
    print(f"  type(df.index)    : {type(df.index)}")
    print(f"  df.index.name     : {getattr(df.index, 'name', 'N/A')}")
    print(f"  df.index[:5]      : {df.index[:5].tolist()}")
    print(f"  type(df.columns)  : {type(df.columns)}")
    print(f"  df.columns        : {df.columns.tolist()}")
    print(f"  df.dtypes:\n{textwrap.indent(str(df.dtypes), '    ')}")
    print(f"  df.head(5):\n{textwrap.indent(df.head(5).to_string(), '    ')}")


def main() -> None:
    """Open LSEG, capture raw TR.MIDYIELD response, save debug CSV."""
    setup_logging("INFO")
    settings = load_settings()

    audit_dir = PROJECT_ROOT / settings.outputs.audit_dir
    ensure_output_dirs(audit_dir)

    debug_path = audit_dir / "debug_tr_midyield_raw_US10YT_RR.csv"

    print(f"\nDebug target : {RIC} / {FIELD}")
    print(f"Date range   : {START} → {END}")
    print(f"Output path  : {debug_path}")

    try:
        open_lseg_session(settings.lseg.session_name)

        import lseg.data as ld

        # --- Raw call 1: standard parameters dict ---
        print("\n[Call 1] ld.get_data() with SDate/EDate/Frq parameters dict ...")
        try:
            raw1 = ld.get_data(
                universe=[RIC],
                fields=[FIELD],
                parameters={"SDate": START, "EDate": END, "Frq": "D"},
            )
            _describe_dataframe(raw1, "Call 1: parameters={SDate, EDate, Frq}")
            if raw1 is not None and not raw1.empty:
                raw1.to_csv(debug_path)
                print(f"\n  Raw CSV saved → {debug_path}")
        except Exception as exc:
            print(f"  [FAILED] Call 1 raised: {type(exc).__name__}: {exc}")

        # --- Raw call 2: inline field parameters ---
        print("\n[Call 2] ld.get_data() with inline field parameter syntax ...")
        try:
            inline_field = f"{FIELD}(SDate={START},EDate={END},Frq=D)"
            raw2 = ld.get_data(universe=[RIC], fields=[inline_field])
            _describe_dataframe(raw2, f"Call 2: fields=['{inline_field}']")
        except Exception as exc:
            print(f"  [FAILED] Call 2 raised: {type(exc).__name__}: {exc}")

        # --- Raw call 3: ld.get_history (as cross-check) ---
        print("\n[Call 3] ld.get_history() as cross-check ...")
        try:
            raw3 = ld.get_history(universe=RIC, fields=[FIELD], start=START, end=END)
            _describe_dataframe(raw3, "Call 3: ld.get_history()")
        except Exception as exc:
            print(f"  [FAILED] Call 3 raised: {type(exc).__name__}: {exc}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    finally:
        close_lseg_session()

    print("\nDone. Share the output above (and the CSV) to diagnose the normaliser.")


if __name__ == "__main__":
    main()
