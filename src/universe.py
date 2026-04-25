"""Universe audit: test every RIC honestly and select the first valid candidate per block."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit result schema
# ---------------------------------------------------------------------------


@dataclass
class AuditResult:
    """Audit outcome for a single RIC and field combination."""

    logical_name: str
    category: str
    ric: str
    field: str
    label: str
    status: str           # "OK" | "UNAVAILABLE" | "ERROR"
    last_value: Optional[float]
    rows_available: int
    verdict: str
    selected: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Date window helper
# ---------------------------------------------------------------------------


def _audit_window(days: int = 60) -> tuple[str, str]:
    """Return (start, end) ISO date strings for a short audit window ending yesterday."""
    end_dt = datetime.now() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Individual RIC auditors
# ---------------------------------------------------------------------------


def audit_price_ric(
    logical_name: str,
    category: str,
    ric: str,
    fields: list[str],
    label: str,
    window_days: int = 60,
) -> AuditResult:
    """Attempt a snapshot and short price history for a single RIC.

    Records the first field that returns data. Returns an honest AuditResult — never fakes values.
    """
    from src.lseg_provider import get_snapshot, get_history_safe

    start, end = _audit_window(window_days)
    last_value: Optional[float] = None
    selected_field = fields[0]
    rows = 0

    try:
        # --- Snapshot (current value) ---
        snap = get_snapshot([ric], fields)
        if snap is not None and not snap.empty:
            for f in fields:
                if f in snap.columns:
                    vals = snap[f].dropna()
                    if not vals.empty:
                        try:
                            last_value = float(vals.iloc[0])
                            selected_field = f
                        except (TypeError, ValueError):
                            pass
                        break

        # --- Short history ---
        for f in fields:
            hist = get_history_safe(ric, [f], start, end)
            if hist is not None and not hist.empty:
                rows = max(rows, len(hist))
                if last_value is None:
                    col_vals = hist.iloc[:, 0].dropna()
                    if not col_vals.empty:
                        try:
                            last_value = float(col_vals.iloc[-1])
                        except (TypeError, ValueError):
                            pass
                selected_field = f
                break

        if last_value is not None or rows > 0:
            return AuditResult(
                logical_name=logical_name,
                category=category,
                ric=ric,
                field=selected_field,
                label=label,
                status="OK",
                last_value=last_value,
                rows_available=rows,
                verdict=f"PRICE_DATA_AVAILABLE (field={selected_field})",
            )

        return AuditResult(
            logical_name=logical_name,
            category=category,
            ric=ric,
            field=fields[0],
            label=label,
            status="UNAVAILABLE",
            last_value=None,
            rows_available=0,
            verdict="NO_DATA_RETURNED",
        )

    except Exception as exc:
        logger.warning("Exception auditing price RIC %s: %s", ric, exc)
        return AuditResult(
            logical_name=logical_name,
            category=category,
            ric=ric,
            field=fields[0],
            label=label,
            status="ERROR",
            last_value=None,
            rows_available=0,
            verdict="EXCEPTION_DURING_AUDIT",
            error_message=str(exc),
        )


def audit_true_yield_ric(
    logical_name: str,
    ric: str,
    field: str,
    label: str,
    window_days: int = 60,
) -> AuditResult:
    """Attempt TR.MIDYIELD history retrieval and validate that values look like yield percentages.

    Rejects values > 20 as price-like — flags them as ERROR rather than silently accepting.
    """
    from src.lseg_provider import get_tr_field_history_via_get_data

    start, end = _audit_window(window_days)

    try:
        hist = get_tr_field_history_via_get_data(ric, field, start, end)

        if hist is None or hist.empty:
            return AuditResult(
                logical_name=logical_name,
                category="rates_yield",
                ric=ric,
                field=field,
                label=label,
                status="UNAVAILABLE",
                last_value=None,
                rows_available=0,
                verdict="NO_YIELD_DATA_RETURNED",
            )

        value_col = field if field in hist.columns else hist.columns[0]
        vals = hist[value_col].dropna()

        if vals.empty:
            return AuditResult(
                logical_name=logical_name,
                category="rates_yield",
                ric=ric,
                field=field,
                label=label,
                status="UNAVAILABLE",
                last_value=None,
                rows_available=0,
                verdict="COLUMN_EMPTY_AFTER_DROPNA",
            )

        last_value = float(vals.iloc[-1])
        rows = len(vals)

        # Guard: reject price-like values (e.g. futures price ~130 returned instead of ~4% yield)
        if last_value > 20.0:
            return AuditResult(
                logical_name=logical_name,
                category="rates_yield",
                ric=ric,
                field=field,
                label=label,
                status="ERROR",
                last_value=last_value,
                rows_available=rows,
                verdict="VALUE_LOOKS_PRICE_NOT_YIELD",
                error_message=(
                    f"Last value {last_value:.4f} is outside yield-like range — "
                    "expected < 20%. Do not use as a yield level."
                ),
            )

        return AuditResult(
            logical_name=logical_name,
            category="rates_yield",
            ric=ric,
            field=field,
            label=label,
            status="OK",
            last_value=last_value,
            rows_available=rows,
            verdict="TRUE_YIELD_AVAILABLE",
        )

    except Exception as exc:
        logger.warning("Exception auditing yield RIC %s/%s: %s", ric, field, exc)
        return AuditResult(
            logical_name=logical_name,
            category="rates_yield",
            ric=ric,
            field=field,
            label=label,
            status="ERROR",
            last_value=None,
            rows_available=0,
            verdict="EXCEPTION_DURING_AUDIT",
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Full universe audit
# ---------------------------------------------------------------------------


def audit_full_universe(
    instruments_cfg: dict,
    window_days: int = 60,
) -> list[AuditResult]:
    """Audit all instrument categories defined in instruments.yaml.

    For groups with multiple candidates the first OK result is marked selected=True.
    True yield RICs and proxy instruments are each audited independently.
    """
    results: list[AuditResult] = []

    # --- Single-block categories with ordered candidate lists ---
    for category_key in ("equity", "volatility", "fx"):
        block = instruments_cfg.get(category_key)
        if block is None:
            logger.warning("Category '%s' missing from instruments config — skipping.", category_key)
            continue
        logical_name: str = block["logical_name"]
        first_ok_found = False
        for candidate in block["candidates"]:
            result = audit_price_ric(
                logical_name=logical_name,
                category=category_key,
                ric=candidate["ric"],
                fields=candidate["fields"],
                label=candidate["label"],
                window_days=window_days,
            )
            result.selected = result.status == "OK" and not first_ok_found
            if result.selected:
                first_ok_found = True
            results.append(result)
            logger.info("[%s] %s → %s", logical_name, candidate["ric"], result.verdict)
        if not first_ok_found:
            logger.warning(
                "No valid candidate found for '%s' — will operate in degraded mode.", logical_name
            )

    # --- Commodity sub-blocks (brent, wti) ---
    commodities_cfg = instruments_cfg.get("commodities", {})
    for subkey, block in commodities_cfg.items():
        logical_name = block["logical_name"]
        first_ok_found = False
        for candidate in block["candidates"]:
            result = audit_price_ric(
                logical_name=logical_name,
                category=f"commodity_{subkey}",
                ric=candidate["ric"],
                fields=candidate["fields"],
                label=candidate["label"],
                window_days=window_days,
            )
            result.selected = result.status == "OK" and not first_ok_found
            if result.selected:
                first_ok_found = True
            results.append(result)
            logger.info("[%s] %s → %s", logical_name, candidate["ric"], result.verdict)
        if not first_ok_found:
            logger.warning("No valid commodity candidate for '%s'.", logical_name)

    # --- True yield curve (confirmed RICs — audited individually) ---
    yield_cfg = instruments_cfg.get("rates_true_yield_curve", {})
    for ric_key, entry in yield_cfg.items():
        result = audit_true_yield_ric(
            logical_name=ric_key,
            ric=entry["ric"],
            field=entry["field"],
            label=entry["label"],
            window_days=window_days,
        )
        result.selected = result.status == "OK"
        results.append(result)
        logger.info("[%s] %s → %s", ric_key, entry["ric"], result.verdict)

    # --- Proxy instruments (futures and ETFs) ---
    proxies_cfg = instruments_cfg.get("rates_proxies", {})
    for proxy_type, proxy_list in proxies_cfg.items():
        for entry in proxy_list:
            result = audit_price_ric(
                logical_name=entry["ric"],
                category=f"rates_proxy_{proxy_type}",
                ric=entry["ric"],
                fields=[entry["field"]],
                label=entry["label"],
                window_days=window_days,
            )
            result.selected = result.status == "OK"
            results.append(result)
            logger.info("[rates_proxy_%s] %s → %s", proxy_type, entry["ric"], result.verdict)

    return results


# ---------------------------------------------------------------------------
# Selection and persistence
# ---------------------------------------------------------------------------


def select_validated_universe(results: list[AuditResult]) -> dict[str, AuditResult]:
    """Return a mapping of logical_name → selected AuditResult for use in downstream modules."""
    return {r.logical_name: r for r in results if r.selected}


def save_validated_universe(results: list[AuditResult], path: Path) -> None:
    """Serialise all audit results (including failed ones) to a CSV at the given path."""
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    df.insert(0, "audit_timestamp", datetime.now().isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Validated universe saved → %s  (%d rows)", path, len(df))
