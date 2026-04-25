"""Historical data retrieval, alignment, and cleaning for the cross-asset daily panel."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Category tag used for true yield curve instruments
_YIELD_CATEGORY = "rates_yield"

# Forward-fill limit in calendar days (covers US market holidays and short gaps)
_MAX_FFILL_DAYS: int = 3

# Columns with fewer than this fraction of non-null rows are dropped
_MIN_NON_NULL_FRACTION: float = 0.90


# ---------------------------------------------------------------------------
# 1. Universe loading
# ---------------------------------------------------------------------------


def load_validated_universe(path: Path) -> pd.DataFrame:
    """Load validated_universe.csv and return only the selected instruments as a DataFrame."""
    df = pd.read_csv(path)
    # Coerce the 'selected' column — CSV booleans can arrive as strings or True/False
    df["selected"] = df["selected"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
    )
    selected = df[df["selected"]].reset_index(drop=True)
    logger.info(
        "Loaded validated universe: %d selected instruments from %s", len(selected), path
    )
    return selected


# ---------------------------------------------------------------------------
# 2. Series fetching (routes yield vs price to the right LSEG call)
# ---------------------------------------------------------------------------


def fetch_series(
    ric: str,
    field: str,
    category: str,
    start: str,
    end: str,
) -> pd.Series:
    """Retrieve a single daily time-series from LSEG and return as a named pd.Series.

    Routing rule:
      - rates_yield  →  get_yield_history()  (ld.get_history preferred path)
      - everything else  →  get_history_safe()  (standard price history)

    The returned Series has a DatetimeIndex and is named after the RIC.
    An empty Series is returned (not an exception) if LSEG returns no data.
    """
    from src.lseg_provider import get_history_safe, get_yield_history

    try:
        if category == _YIELD_CATEGORY:
            raw = get_yield_history(ric, start, end)
        else:
            raw = get_history_safe(ric, [field], start, end)
    except Exception as exc:
        logger.error("Unexpected error fetching %s / %s: %s", ric, field, exc)
        return pd.Series(dtype=float, name=ric)

    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=ric)

    # raw is a normalised single-column DataFrame with DatetimeIndex
    series = raw.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    series = series[~series.index.duplicated(keep="last")]
    series = series.sort_index()
    return series


# ---------------------------------------------------------------------------
# 3. Panel assembly
# ---------------------------------------------------------------------------


def build_raw_panel(
    selected: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch history for every selected instrument and outer-join into a raw panel.

    Column names come from `logical_name` in the validated universe, giving consistent
    downstream references (e.g. 'us_equity', 'us_10y', 'SHY.O').

    Returns a DataFrame with DatetimeIndex sorted ascending. Missing observations
    (e.g. dates where only some markets have data) are left as NaN for clean_panel()
    to handle.
    """
    series_dict: dict[str, pd.Series] = {}
    failed: list[str] = []

    for _, row in selected.iterrows():
        ric = str(row["ric"])
        field = str(row["field"])
        category = str(row["category"])
        logical_name = str(row["logical_name"])

        logger.info(
            "Fetching  %-28s  ric=%-14s  field=%s",
            logical_name, ric, field,
        )

        series = fetch_series(ric, field, category, start, end)

        if series.empty:
            logger.warning("  → NO DATA  —  skipping %s", logical_name)
            failed.append(logical_name)
            continue

        first = series.index[0].date() if not series.empty else "?"
        last = series.index[-1].date() if not series.empty else "?"
        logger.info("  → %d rows  [%s → %s]", len(series), first, last)

        series_dict[logical_name] = series

    if failed:
        logger.warning(
            "Failed to retrieve data for %d instrument(s): %s", len(failed), failed
        )

    if not series_dict:
        logger.error("No data fetched for any instrument — raw panel is empty.")
        return pd.DataFrame()

    # Outer join on DatetimeIndex — dates missing in some series become NaN
    raw = pd.DataFrame(series_dict)
    raw.index.name = "date"
    raw = raw.sort_index()
    return raw


# ---------------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------------


def clean_panel(
    raw: pd.DataFrame,
    max_ffill_days: int = _MAX_FFILL_DAYS,
    min_non_null_fraction: float = _MIN_NON_NULL_FRACTION,
) -> pd.DataFrame:
    """Produce a clean, analysis-ready daily panel from the raw outer-join panel.

    Steps applied in order:
      1. Remove weekend rows (Sat/Sun — no major market trading).
      2. Forward-fill short gaps up to max_ffill_days (covers public holidays).
      3. Drop columns with excessive missing data (below min_non_null_fraction).
      4. Drop rows where all remaining columns are NaN.
      5. Coerce to float64 and sort index.

    Note: does NOT perform an inner-join / dropna-all-cols to avoid discarding
    valid data. Remaining NaNs are handled by each downstream module as needed.
    """
    if raw is None or raw.empty:
        logger.error("clean_panel() received empty input — returning empty DataFrame.")
        return pd.DataFrame()

    logger.info(
        "Cleaning panel: input shape=%s  date_range=[%s → %s]",
        raw.shape, raw.index[0].date(), raw.index[-1].date(),
    )

    df = raw.copy()

    # Step 1 — remove weekends
    df = df[df.index.dayofweek < 5]
    logger.info("  After weekend removal: %d rows", len(df))

    # Step 2 — forward-fill short gaps (holidays, non-overlapping closures)
    df = df.ffill(limit=max_ffill_days)

    # Step 3 — drop columns with too many NaNs
    non_null_frac = df.notna().mean()
    bad_cols = non_null_frac[non_null_frac < min_non_null_fraction].index.tolist()
    if bad_cols:
        logger.warning(
            "  Dropping %d column(s) with < %.0f%% non-null data: %s",
            len(bad_cols), min_non_null_fraction * 100, bad_cols,
        )
        df = df.drop(columns=bad_cols)
    else:
        logger.info("  All columns meet the %.0f%% non-null threshold.", min_non_null_fraction * 100)

    # Step 4 — drop rows where every remaining column is NaN
    all_nan_rows = df.isna().all(axis=1).sum()
    if all_nan_rows:
        logger.warning("  Dropping %d row(s) where all values are NaN.", all_nan_rows)
    df = df.dropna(how="all")

    # Step 5 — coerce to float64 and sort
    df = df.astype(float)
    df = df.sort_index()

    logger.info(
        "  Output shape=%s  date_range=[%s → %s]",
        df.shape, df.index[0].date(), df.index[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# 5. Summary and persistence
# ---------------------------------------------------------------------------


def save_dataset_summary(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    path: Path,
) -> None:
    """Save a per-column data-quality summary CSV to the given path."""
    rows = []
    for col in cleaned.columns:
        s = cleaned[col].dropna()
        rows.append({
            "column": col,
            "first_date": str(s.index[0].date()) if not s.empty else "",
            "last_date": str(s.index[-1].date()) if not s.empty else "",
            "n_total": len(cleaned),
            "n_valid": int(s.notna().sum()),
            "missing_pct": round(cleaned[col].isna().mean() * 100, 2),
            "mean": round(float(s.mean()), 6) if not s.empty else None,
            "min": round(float(s.min()), 6) if not s.empty else None,
            "max": round(float(s.max()), 6) if not s.empty else None,
        })
    summary = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    logger.info("Dataset summary saved → %s", path)
