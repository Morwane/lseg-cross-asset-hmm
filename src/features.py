"""Cross-asset and yield-curve feature engineering.

All features are strictly backward-looking:
  - pct_change(n)  uses data at [t-n, t] only.
  - rolling(w)     uses the trailing w observations ending at t.
  - expanding()    uses all observations from the start up to t.
  - diff(1)        uses data at [t-1, t] only.

No future data is used at any point. The warm-up period (first ~252 rows for
z-score features) produces NaN — this is intentional and honest.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS: int = 252  # canonical annualisation constant


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_cleaned_panel(path: Path) -> pd.DataFrame:
    """Load cleaned_dataset.csv and return a DataFrame with DatetimeIndex."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def save_features(
    features: pd.DataFrame,
    features_path: Path,
    summary_path: Path,
) -> None:
    """Save feature dataset and per-column summary CSV to disk."""
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path)
    logger.info("Feature dataset saved → %s  shape=%s", features_path, features.shape)

    rows = []
    for col in features.columns:
        s = features[col].dropna()
        rows.append({
            "feature": col,
            "n_total": len(features),
            "n_valid": len(s),
            "missing_pct": round(features[col].isna().mean() * 100, 2),
            "mean": round(float(s.mean()), 6) if not s.empty else None,
            "std": round(float(s.std()), 6) if not s.empty else None,
            "min": round(float(s.min()), 6) if not s.empty else None,
            "max": round(float(s.max()), 6) if not s.empty else None,
        })
    summary = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    logger.info("Feature summary saved → %s", summary_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_cols(panel: pd.DataFrame, cols: list[str], block: str) -> bool:
    """Return True if all cols are present, else log a warning and return False."""
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        logger.warning("Block '%s' skipped — missing columns: %s", block, missing)
        return False
    return True


def _rolling_std_ann(returns: pd.Series, window: int) -> pd.Series:
    """Annualised rolling standard deviation of daily returns."""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score: (x - rolling_mean) / rolling_std over the last `window` days."""
    roll = series.rolling(window, min_periods=window)
    mu = roll.mean()
    sigma = roll.std()
    sigma = sigma.where(sigma > 0, other=float("nan"))  # avoid division by zero
    return (series - mu) / sigma


# ---------------------------------------------------------------------------
# Feature block builders
# ---------------------------------------------------------------------------


def build_equity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return 5 equity features derived from the us_equity price column.

    equity_ret_1d            — 1-day log-linear return
    equity_ret_5d            — 5-day return (one trading week)
    equity_ret_20d           — 20-day return (one trading month)
    equity_realised_vol_20d  — 20-day annualised realised volatility
    equity_drawdown          — current drawdown from expanding peak (≤ 0)
    """
    if not _require_cols(panel, ["us_equity"], "equity"):
        return pd.DataFrame(index=panel.index)

    price = panel["us_equity"]
    ret_1d = price.pct_change(1)

    return pd.DataFrame({
        "equity_ret_1d":           ret_1d,
        "equity_ret_5d":           price.pct_change(5),
        "equity_ret_20d":          price.pct_change(20),
        "equity_realised_vol_20d": _rolling_std_ann(ret_1d, 20),
        "equity_drawdown":         price / price.expanding(min_periods=1).max() - 1,
    }, index=panel.index)


def build_volatility_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return 3 VIX features; falls back to equity realised vol if vix is absent.

    vix_level       — raw VIX index level (or realised vol proxy)
    vix_change_1d   — absolute 1-day change in VIX points
    vix_zscore_252d — trailing 252-day z-score of VIX level
    """
    if "vix" in panel.columns:
        vix = panel["vix"]
    elif "us_equity" in panel.columns:
        logger.warning("VIX unavailable — using 20-day equity realised vol as proxy.")
        ret_1d = panel["us_equity"].pct_change(1)
        vix = _rolling_std_ann(ret_1d, 20) * 100  # scale to VIX-like units
    else:
        logger.warning("Block 'volatility' skipped — no vix or us_equity column.")
        return pd.DataFrame(index=panel.index)

    return pd.DataFrame({
        "vix_level":       vix,
        "vix_change_1d":   vix.diff(1),
        "vix_zscore_252d": _zscore(vix, 252),
    }, index=panel.index)


def build_fx_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return 2 US Dollar index features.

    dxy_ret_1d      — daily return of the USD index
    dxy_zscore_252d — trailing 252-day z-score of the USD index level
    """
    if not _require_cols(panel, ["usd_index"], "fx"):
        return pd.DataFrame(index=panel.index)

    dxy = panel["usd_index"]

    return pd.DataFrame({
        "dxy_ret_1d":      dxy.pct_change(1),
        "dxy_zscore_252d": _zscore(dxy, 252),
    }, index=panel.index)


def build_commodity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return up to 5 commodity features (Brent, WTI, spread).

    brent_ret_1d           — Brent 1-day return
    brent_ret_5d           — Brent 5-day return
    brent_realised_vol_20d — Brent 20-day annualised realised volatility
    wti_ret_1d             — WTI 1-day return
    brent_wti_spread       — Brent price minus WTI price (USD, usually positive)
    """
    has_brent = "brent_front" in panel.columns
    has_wti = "wti_front" in panel.columns

    if not has_brent and not has_wti:
        logger.warning("Block 'commodity' skipped — neither brent_front nor wti_front present.")
        return pd.DataFrame(index=panel.index)

    feats: dict[str, pd.Series] = {}

    if has_brent:
        brent = panel["brent_front"]
        brent_ret_1d = brent.pct_change(1)
        feats["brent_ret_1d"]           = brent_ret_1d
        feats["brent_ret_5d"]           = brent.pct_change(5)
        feats["brent_realised_vol_20d"] = _rolling_std_ann(brent_ret_1d, 20)

    if has_wti:
        feats["wti_ret_1d"] = panel["wti_front"].pct_change(1)

    if has_brent and has_wti:
        feats["brent_wti_spread"] = panel["brent_front"] - panel["wti_front"]

    return pd.DataFrame(feats, index=panel.index)


def build_yield_level_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return 4 raw yield level features (TR.MIDYIELD values in percent).

    us_2y_yield  — US 2Y Treasury yield (%)
    us_5y_yield  — US 5Y Treasury yield (%)
    us_10y_yield — US 10Y Treasury yield (%)
    us_30y_yield — US 30Y Treasury yield (%)
    """
    if not _require_cols(panel, ["us_2y", "us_5y", "us_10y", "us_30y"], "yield_levels"):
        return pd.DataFrame(index=panel.index)

    return pd.DataFrame({
        "us_2y_yield":  panel["us_2y"],
        "us_5y_yield":  panel["us_5y"],
        "us_10y_yield": panel["us_10y"],
        "us_30y_yield": panel["us_30y"],
    }, index=panel.index)


def build_yield_change_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return 4 daily yield change features expressed in basis points.

    Yields are stored in percent (e.g. 4.32 = 4.32%). A daily change of 0.01 pp
    equals 1 bp, so multiplying diff() by 100 converts to basis points.

    us_2y_change_bps  — daily change in 2Y yield (bps)
    us_5y_change_bps  — daily change in 5Y yield (bps)
    us_10y_change_bps — daily change in 10Y yield (bps)
    us_30y_change_bps — daily change in 30Y yield (bps)
    """
    if not _require_cols(panel, ["us_2y", "us_5y", "us_10y", "us_30y"], "yield_changes"):
        return pd.DataFrame(index=panel.index)

    return pd.DataFrame({
        "us_2y_change_bps":  panel["us_2y"].diff(1) * 100,
        "us_5y_change_bps":  panel["us_5y"].diff(1) * 100,
        "us_10y_change_bps": panel["us_10y"].diff(1) * 100,
        "us_30y_change_bps": panel["us_30y"].diff(1) * 100,
    }, index=panel.index)


def build_yield_curve_structure_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return 5 yield curve structure features.

    slope_10y_2y           — 10Y minus 2Y yield spread (pp); negative = inverted
    slope_30y_5y           — 30Y minus 5Y yield spread (pp)
    curve_butterfly_2s10s30s — 2×10Y − 2Y − 30Y; positive = humped belly
    curve_level            — simple average of 2Y/5Y/10Y/30Y yields (pp)
    curve_slope_change_bps — daily change in slope_10y_2y (bps)
    """
    if not _require_cols(panel, ["us_2y", "us_5y", "us_10y", "us_30y"], "yield_curve_structure"):
        return pd.DataFrame(index=panel.index)

    y2  = panel["us_2y"]
    y5  = panel["us_5y"]
    y10 = panel["us_10y"]
    y30 = panel["us_30y"]
    slope_10y_2y = y10 - y2

    return pd.DataFrame({
        "slope_10y_2y":            slope_10y_2y,
        "slope_30y_5y":            y30 - y5,
        "curve_butterfly_2s10s30s": 2 * y10 - y2 - y30,
        "curve_level":             (y2 + y5 + y10 + y30) / 4,
        "curve_slope_change_bps":  slope_10y_2y.diff(1) * 100,
    }, index=panel.index)


def build_defensive_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Return up to 3 Treasury ETF daily return features.

    SHY.O_ret_1d — short-duration Treasury ETF daily return
    IEF.O_ret_1d — 7-10Y Treasury ETF daily return
    TLT.O_ret_1d — long-duration Treasury ETF daily return
    """
    feats: dict[str, pd.Series] = {}
    for col, name in [("SHY.O", "SHY.O_ret_1d"), ("IEF.O", "IEF.O_ret_1d"), ("TLT.O", "TLT.O_ret_1d")]:
        if col in panel.columns:
            feats[name] = panel[col].pct_change(1)
        else:
            logger.warning("Defensive feature '%s' skipped — '%s' not in panel.", name, col)

    return pd.DataFrame(feats, index=panel.index)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_all_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build all 31 cross-asset features from the cleaned daily panel.

    Combines 8 feature blocks into one aligned DataFrame. Feature values are NaN
    during warm-up periods (e.g. the first 252 rows for z-score features). Callers
    decide how to handle the warm-up — it is not dropped here.
    """
    blocks = [
        build_equity_features(panel),
        build_volatility_features(panel),
        build_fx_features(panel),
        build_commodity_features(panel),
        build_yield_level_features(panel),
        build_yield_change_features(panel),
        build_yield_curve_structure_features(panel),
        build_defensive_features(panel),
    ]

    non_empty = [b for b in blocks if not b.empty]
    if not non_empty:
        logger.error("All feature blocks returned empty — check cleaned panel columns.")
        return pd.DataFrame(index=panel.index)

    features = pd.concat(non_empty, axis=1)
    features.index.name = "date"

    n_total = len(features)
    n_complete = int(features.dropna().shape[0])
    logger.info(
        "Feature matrix: %d rows total, %d complete (warm-up: %d rows)",
        n_total, n_complete, n_total - n_complete,
    )

    return features
