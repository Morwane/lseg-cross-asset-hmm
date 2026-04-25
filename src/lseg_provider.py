"""LSEG data retrieval wrappers — conservative, version-tolerant, no keyword guessing."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Plausible date range for financial data — guards against numeric yields being
# misread as Unix timestamps when pd.to_datetime() is applied heuristically.
_DATE_FLOOR = pd.Timestamp("1990-01-01")
_DATE_CEIL = pd.Timestamp("2035-01-01")


# ---------------------------------------------------------------------------
# Public retrieval functions
# ---------------------------------------------------------------------------


def get_snapshot(universe: list[str], fields: list[str]) -> pd.DataFrame:
    """Retrieve a current-value snapshot for a list of RICs and fields."""
    import lseg.data as ld

    try:
        response = ld.get_data(universe=universe, fields=fields)
        if response is None:
            return pd.DataFrame()
        return response
    except Exception as exc:
        logger.error("Snapshot failed — universe=%s fields=%s: %s", universe, fields, exc)
        return pd.DataFrame()


def get_history_safe(
    ric: str,
    fields: list[str],
    start: str,
    end: str,
    interval: str = "1D",
) -> pd.DataFrame:
    """Retrieve price history for a single RIC via ld.get_history and normalise the result."""
    import lseg.data as ld

    try:
        response = ld.get_history(
            universe=ric,
            fields=fields,
            start=start,
            end=end,
            interval=interval,
        )
        if response is None or response.empty:
            return pd.DataFrame()
        return normalise_lseg_history(response, ric, fields[0])
    except Exception as exc:
        logger.error("History failed — ric=%s fields=%s: %s", ric, fields, exc)
        return pd.DataFrame()


def get_tr_field_history_via_get_data(
    ric: str,
    field: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Retrieve TR field history (e.g. TR.MIDYIELD) via ld.get_data with SDate/EDate.

    Does NOT pass use_field_names_in_headers — unsupported in some library versions.
    Falls back to inline field-parameter syntax if the first call returns empty.
    """
    import lseg.data as ld

    # --- Attempt 1: parameters dict (standard) ---
    try:
        response = ld.get_data(
            universe=[ric],
            fields=[field],
            parameters={"SDate": start, "EDate": end, "Frq": "D"},
        )
        if response is not None and not response.empty:
            result = normalise_lseg_history(response, ric, field)
            if not result.empty:
                return result
            logger.warning(
                "TR field history: Call 1 returned data but normaliser produced empty result "
                "for %s/%s — trying fallback.",
                ric,
                field,
            )
    except Exception as exc:
        logger.warning(
            "TR field history Call 1 raised %s for %s/%s — trying fallback: %s",
            type(exc).__name__,
            ric,
            field,
            exc,
        )

    # --- Attempt 2: inline field-parameter syntax ---
    try:
        inline_field = f"{field}(SDate={start},EDate={end},Frq=D)"
        response2 = ld.get_data(universe=[ric], fields=[inline_field])
        if response2 is not None and not response2.empty:
            result2 = normalise_lseg_history(response2, ric, field)
            if not result2.empty:
                logger.info("TR field history fallback succeeded for %s/%s.", ric, field)
                return result2
    except Exception as exc2:
        logger.warning(
            "TR field history Call 2 raised %s for %s/%s: %s",
            type(exc2).__name__,
            ric,
            field,
            exc2,
        )

    logger.error(
        "TR field history: both attempts returned no usable data for %s/%s.", ric, field
    )
    return pd.DataFrame()


def get_yield_history(ric: str, start: str, end: str) -> pd.DataFrame:
    """Retrieve TR.MIDYIELD history for a US Treasury RIC.

    Uses ld.get_history() as the primary path — confirmed to return a clean DatetimeIndex
    in the target LSEG environment. Falls back to get_tr_field_history_via_get_data() if
    get_history() returns empty (environment-dependent).
    """
    field = "TR.MIDYIELD"

    result = get_history_safe(ric, [field], start, end)
    if not result.empty:
        return result

    logger.warning(
        "get_history() returned empty for %s/%s — falling back to get_data() path.", ric, field
    )
    return get_tr_field_history_via_get_data(ric, field, start, end)


# ---------------------------------------------------------------------------
# Normalisation — handles every known LSEG response shape
# ---------------------------------------------------------------------------


def normalise_lseg_history(df: pd.DataFrame, ric: str, field: str) -> pd.DataFrame:
    """Convert any raw LSEG response shape to a clean DatetimeIndex single-column DataFrame.

    Shapes handled (in order of priority):
      A. DatetimeIndex + value column (typical ld.get_history response).
      B. MultiIndex (Instrument × Date) on the row index.
      C. Object/string index whose values are parseable as trading dates.
      D. Flat RangeIndex with an explicit date column ('Date', 'Calc Date', etc.)
         — handles duplicate column names and MultiIndex column headers first.

    The returned DataFrame always has:
      - index   : pd.DatetimeIndex, name="date", no duplicates, sorted ascending
      - columns : [field]   (single column, named after the requested field)
      - values  : numeric
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ── Pre-processing ──────────────────────────────────────────────────────

    # Flatten MultiIndex columns (e.g. (RIC, "TR.MIDYIELD") → "RIC TR.MIDYIELD")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(str(c) for c in col if str(c) not in ("", "nan")).strip()
            for col in df.columns.values
        ]

    # Remove duplicate column names before any selection
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # ── Shape B: MultiIndex row index (e.g. (Instrument, Date)) ────────────
    if isinstance(df.index, pd.MultiIndex):
        df = _flatten_multiindex_rows(df)
        # After flattening, fall through to shapes A / C / D below

    # ── Shape A: DatetimeIndex already present ──────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        value_col = _find_value_column(df, ric, field)
        if value_col is None:
            logger.warning(
                "Shape A — no value column for %s/%s; columns=%s",
                ric, field, df.columns.tolist(),
            )
            return pd.DataFrame()
        result = df[[value_col]].rename(columns={value_col: field})
        result.index.name = "date"
        return _clean(result)

    # ── Shape C: non-DatetimeIndex whose values look like trading dates ─────
    idx_dt = _try_as_datetime_index(df.index)
    if idx_dt is not None:
        df.index = idx_dt
        df.index.name = "date"
        value_col = _find_value_column(df, ric, field)
        if value_col is not None:
            result = df[[value_col]].rename(columns={value_col: field})
            return _clean(result)

    # ── Shape D: flat format with an explicit date column ───────────────────
    date_col = _find_date_column(df)
    value_col = _find_value_column(df, ric, field)

    if date_col is not None and value_col is not None:
        try:
            subset = df[[date_col, value_col]].copy()
            subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
            subset = subset.dropna(subset=[date_col])
            subset = subset.set_index(date_col)
            subset.index.name = "date"
            subset = subset.rename(columns={value_col: field})
            return _clean(subset)
        except Exception as exc:
            logger.warning(
                "Shape D set_index failed for %s/%s: %s", ric, field, exc
            )

    # ── Fallback: value found but no date ───────────────────────────────────
    if value_col is not None:
        logger.warning(
            "No parseable date for %s/%s — returning without DatetimeIndex (unreliable).",
            ric, field,
        )
        return df[[value_col]].rename(columns={value_col: field})

    logger.warning(
        "Cannot parse LSEG response for %s/%s — columns=%s index_type=%s",
        ric, field, df.columns.tolist(), type(df.index).__name__,
    )
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_multiindex_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Reset a MultiIndex row index and promote the level that looks like dates."""
    try:
        df_r = df.reset_index()
    except Exception:
        return df

    n_levels = df.index.nlevels
    for i in range(n_levels):
        col = df_r.columns[i]
        dt_idx = _try_as_datetime_index(df_r[col].astype(str))
        if dt_idx is not None:
            df_r.index = dt_idx
            df_r.index.name = "date"
            # Drop all former index columns (they are now redundant)
            former_index_cols = df_r.columns[:n_levels].tolist()
            df_r = df_r.drop(columns=former_index_cols, errors="ignore")
            return df_r

    # No date level found — just reset and return flat
    return df_r


def _try_as_datetime_index(values: "pd.Index | pd.Series") -> Optional[pd.DatetimeIndex]:
    """Return a DatetimeIndex if ≥80% of values parse as plausible trading dates.

    Applies a date-range guard to prevent numeric yield values (e.g. 4.3224) from
    being mistaken for Unix timestamps.
    """
    try:
        str_vals = list(values.astype(str))
        if not str_vals:
            return None
        parsed = pd.to_datetime(str_vals, errors="coerce", format="mixed")
        valid = parsed.notna()
        if valid.mean() < 0.8:
            return None
        # Guard: at least 80% of valid dates must be within the plausible range
        in_range = (parsed[valid] >= _DATE_FLOOR) & (parsed[valid] <= _DATE_CEIL)
        if in_range.mean() < 0.8:
            return None
        return parsed
    except Exception:
        return None


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Identify a date/time column in a flat LSEG response DataFrame.

    Priority order:
      1. Column whose name exactly matches a known date keyword.
      2. Column whose name contains 'date' (short names only, avoids false matches).
      3. Heuristic: column whose sample values all parse as plausible trading dates.
    """
    # Priority 1: exact keyword
    date_keywords = {"date", "timestamp", "time", "period", "calc date", "calcdate",
                     "trade date", "tradedate", "settldate", "settl date"}
    for col in df.columns:
        if str(col).lower() in date_keywords:
            return col

    # Priority 2: 'date' substring in short column names
    for col in df.columns:
        col_lower = str(col).lower()
        if "date" in col_lower and len(col_lower) <= 20:
            return col

    # Priority 3: heuristic parse (with range guard via _try_as_datetime_index)
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if sample.empty:
            continue
        dt = _try_as_datetime_index(sample)
        if dt is not None:
            return col

    return None


def _find_value_column(df: pd.DataFrame, ric: str, field: str) -> Optional[str]:
    """Identify the numeric value column in a LSEG response DataFrame.

    Priority order:
      1. Exact case-insensitive field name match.
      2. Field name as substring (handles 'TR.MIDYIELD(Frq=D)' style).
      3. RIC name match.
      4. Column literally named 'value'.
      5. First numeric column that is not a date/instrument identifier.
    """
    skip = {
        "instrument", "ric", "date", "timestamp", "time", "period",
        "calc date", "calcdate", "trade date", "tradedate",
    }

    # 1. Exact field name
    for col in df.columns:
        if str(col).lower() == field.lower():
            return col

    # 2. Field name as substring
    for col in df.columns:
        if field.lower() in str(col).lower():
            return col

    # 3. RIC name
    for col in df.columns:
        if str(col).lower() == ric.lower():
            return col

    # 4. Generic 'value' column
    for col in df.columns:
        if str(col).lower().strip() == "value":
            return col

    # 5. First numeric column that is not a skip-listed identifier
    for col in df.select_dtypes(include="number").columns:
        if str(col).lower().strip() not in skip:
            return col

    return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by DatetimeIndex and remove duplicate dates (keep last occurrence)."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
