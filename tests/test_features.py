"""Tests for feature engineering — Phase 4 implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_equity_features,
    build_yield_change_features,
    build_yield_curve_structure_features,
    build_all_features,
    _zscore,
)


def _make_panel(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily panel with all required columns, DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n, freq="B")

    equity = 3000 + (rng.standard_normal(n) * 20).cumsum()
    vix = 15 + rng.standard_normal(n) * 2
    usd = 100 + rng.standard_normal(n).cumsum()
    brent = 70 + rng.standard_normal(n).cumsum()
    wti = 65 + rng.standard_normal(n).cumsum()
    y2 = 1.0 + rng.standard_normal(n) * 0.05
    y5 = 1.5 + rng.standard_normal(n) * 0.05
    y10 = 2.0 + rng.standard_normal(n) * 0.05
    y30 = 2.5 + rng.standard_normal(n) * 0.05

    return pd.DataFrame({
        "us_equity": equity,
        "vix": vix,
        "usd_index": usd,
        "brent_front": brent,
        "wti_front": wti,
        "us_2y": y2,
        "us_5y": y5,
        "us_10y": y10,
        "us_30y": y30,
        "SHY.O": 84 + rng.standard_normal(n).cumsum(),
        "IEF.O": 100 + rng.standard_normal(n).cumsum(),
        "TLT.O": 120 + rng.standard_normal(n).cumsum(),
    }, index=idx)


def test_returns_are_correctly_calculated():
    """Equity returns match manual pct_change computation."""
    panel = _make_panel(30)
    feats = build_equity_features(panel)

    expected_1d = panel["us_equity"].pct_change(1)
    expected_5d = panel["us_equity"].pct_change(5)
    expected_20d = panel["us_equity"].pct_change(20)

    pd.testing.assert_series_equal(feats["equity_ret_1d"], expected_1d, check_names=False)
    pd.testing.assert_series_equal(feats["equity_ret_5d"], expected_5d, check_names=False)
    pd.testing.assert_series_equal(feats["equity_ret_20d"], expected_20d, check_names=False)

    # drawdown must always be ≤ 0
    drawdown = feats["equity_drawdown"].dropna()
    assert (drawdown <= 1e-10).all(), "Drawdown should be non-positive"


def test_yield_changes_are_in_bps():
    """Yield level changes are expressed in basis points (×100)."""
    panel = _make_panel(30)
    feats = build_yield_change_features(panel)

    # A 1 percentage-point move in yield → 100 bps
    for tenor, col in [("us_2y", "us_2y_change_bps"),
                        ("us_5y", "us_5y_change_bps"),
                        ("us_10y", "us_10y_change_bps"),
                        ("us_30y", "us_30y_change_bps")]:
        expected = panel[tenor].diff(1) * 100
        pd.testing.assert_series_equal(feats[col], expected, check_names=False)

    # Verify scale: changes should be order-of-magnitude ~bps, not pct
    # With yield std ~0.05 pp, daily changes ≈ 5 bps → abs mean > 0.1
    assert feats["us_10y_change_bps"].dropna().abs().mean() > 0.1


def test_slope_10y_2y_is_correct():
    """slope_10y_2y equals us_10y_yield minus us_2y_yield."""
    panel = _make_panel(30)
    feats = build_yield_curve_structure_features(panel)

    expected = panel["us_10y"] - panel["us_2y"]
    pd.testing.assert_series_equal(
        feats["slope_10y_2y"], expected, check_names=False
    )

    # slope_30y_5y sanity
    expected_30y5y = panel["us_30y"] - panel["us_5y"]
    pd.testing.assert_series_equal(
        feats["slope_30y_5y"], expected_30y5y, check_names=False
    )


def test_curve_butterfly_is_correct():
    """Butterfly = 2×10Y − 2Y − 30Y."""
    panel = _make_panel(30)
    feats = build_yield_curve_structure_features(panel)

    expected = 2 * panel["us_10y"] - panel["us_2y"] - panel["us_30y"]
    pd.testing.assert_series_equal(
        feats["curve_butterfly_2s10s30s"], expected, check_names=False
    )

    # curve_level = simple mean of 4 tenors
    expected_level = (panel["us_2y"] + panel["us_5y"] + panel["us_10y"] + panel["us_30y"]) / 4
    pd.testing.assert_series_equal(
        feats["curve_level"], expected_level, check_names=False
    )


def test_no_future_data_in_rolling_features():
    """Rolling features at time t use only data available at t.

    Method: build features on a prefix [0:k], then on the full panel, and assert
    that the value at position k-1 is identical. If future data leaked, the value
    would differ once more rows were appended.
    """
    panel = _make_panel(60)
    k = 30  # split point — half the panel

    full_feats = build_all_features(panel)
    prefix_feats = build_all_features(panel.iloc[:k])

    # Row index k-1 is the last row of the prefix
    split_date = panel.index[k - 1]

    # Check a rolling feature (realised vol) and a z-score feature
    for col in ["equity_realised_vol_20d", "vix_zscore_252d", "slope_10y_2y"]:
        if col not in full_feats.columns or col not in prefix_feats.columns:
            continue
        full_val = full_feats.loc[split_date, col]
        prefix_val = prefix_feats.loc[split_date, col]
        # Both may be NaN if warm-up window hasn't completed — that's fine
        if pd.isna(full_val) and pd.isna(prefix_val):
            continue
        assert abs(full_val - prefix_val) < 1e-10, (
            f"Look-ahead detected in '{col}': full={full_val:.6f} vs prefix={prefix_val:.6f}"
        )
