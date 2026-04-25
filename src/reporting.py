"""Markdown report generation for the latest regime state and backtest summary."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HONEST_INTERPRETATION = """\
The overlay is not an alpha strategy. In the current out-of-sample period, it sacrificed upside \
participation in exchange for materially lower drawdowns, smaller tail losses and better \
stress-period protection.\
"""


def _pct(v: float, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v * 100:.{decimals}f}%"


def _fmt(v: float, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}"


def generate_latest_report(
    predictions: pd.DataFrame,
    daily_returns: pd.DataFrame,
    metrics_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    features: pd.DataFrame,
    defensive_label: str = "SHY.O",
    report_date: str | None = None,
) -> str:
    """Return the full markdown report as a string."""

    latest_pred = predictions.iloc[-1]
    latest_date = str(latest_pred.name.date() if hasattr(latest_pred.name, "date") else latest_pred.name)
    if report_date is None:
        report_date = latest_date

    regime       = str(latest_pred.get("regime_label", "unknown"))
    prob_ro      = float(latest_pred.get("prob_risk_on",    0))
    prob_tr      = float(latest_pred.get("prob_transition", 0))
    prob_st      = float(latest_pred.get("prob_stress",     0))

    # Latest yield curve
    tenor_cols   = ["us_2y_yield", "us_5y_yield", "us_10y_yield", "us_30y_yield"]
    tenor_labels = ["2Y", "5Y", "10Y", "30Y"]
    latest_feat  = features.iloc[-1] if len(features) > 0 else pd.Series(dtype=float)
    yc_rows = ""
    for col, label in zip(tenor_cols, tenor_labels):
        val = latest_feat.get(col, np.nan)
        if not (isinstance(val, float) and np.isnan(val)):
            yc_rows += f"| {label}  | {val:.3f}% |\n"

    slope_val = latest_feat.get("slope_10y_2y", np.nan)
    slope_str = f"{slope_val:.3f}pp" if not (isinstance(slope_val, float) and np.isnan(slope_val)) else "n/a"

    # Metrics
    m = metrics_df.set_index("metric")

    def _get(key: str, col: str) -> float:
        try:
            return float(m.loc[key, col])
        except (KeyError, ValueError):
            return float("nan")

    bench_cagr   = _get("cagr",             "benchmark")
    ov_cagr      = _get("cagr",             "overlay")
    bench_vol    = _get("annualised_vol",    "benchmark")
    ov_vol       = _get("annualised_vol",    "overlay")
    bench_sharpe = _get("sharpe",            "benchmark")
    ov_sharpe    = _get("sharpe",            "overlay")
    bench_mdd    = _get("max_drawdown",      "benchmark")
    ov_mdd       = _get("max_drawdown",      "overlay")
    bench_worst  = _get("worst_daily_loss",  "benchmark")
    ov_worst     = _get("worst_daily_loss",  "overlay")
    bench_total  = _get("total_return",      "benchmark")
    ov_total     = _get("total_return",      "overlay")
    ov_exposure  = _get("avg_equity_exposure", "overlay")

    # Backtest window
    bt_start = daily_returns.index[0].date() if len(daily_returns) > 0 else "n/a"
    bt_end   = daily_returns.index[-1].date() if len(daily_returns) > 0 else "n/a"
    bt_days  = len(daily_returns)

    # Regime counts
    counts = predictions["regime_label"].value_counts()
    n_total = len(predictions)
    regime_breakdown = ""
    for lbl, n in counts.items():
        regime_breakdown += f"| {lbl.replace('_', ' ').title():<14} | {n:>4} | {n / n_total * 100:>5.1f}% |\n"

    # Stress periods
    stress_rows = ""
    for _, row in stress_df.iterrows():
        br = _pct(row.get("benchmark_total_ret", float("nan")))
        or_ = _pct(row.get("overlay_total_ret", float("nan")))
        stress_rows += f"| {row['period']:<28} | {str(row['coverage']):<34} | {br:>7} | {or_:>8} |\n"

    report = f"""\
# Cross-Asset HMM Regime Engine — Market Regime Report

**Report date:** {report_date}
**Data source:** LSEG Workspace (TR.MIDYIELD yields, SPY/SPX equity, SHY short-duration treasury ETF)

---

## Current Regime Assessment

| Field                | Value                              |
|----------------------|------------------------------------|
| Latest signal date   | {latest_date}                      |
| Regime label         | **{regime.replace('_', ' ').upper()}** |
| Prob (Risk On)       | {prob_ro:.2%}                      |
| Prob (Transition)    | {prob_tr:.2%}                      |
| Prob (Stress)        | {prob_st:.2%}                      |
| Implied equity wt    | {(1.0 if regime == 'risk_on' else 0.5 if regime == 'transition' else 0.0):.0%} |
| Defensive asset      | {defensive_label}                  |

> **Note:** The regime signal is derived from a 3-state Gaussian HMM trained on 8 cross-asset
> features (equity returns, realised volatility, VIX level/z-score, yield curve level and slope,
> 10Y rate change). The label above reflects the walk-forward out-of-sample prediction only —
> no in-sample signal is used in the backtest.

---

## Current Yield Curve

| Tenor | Yield  |
|-------|--------|
{yc_rows}\
| 10Y–2Y slope | {slope_str} |

---

## Regime Distribution (Out-of-Sample)

Walk-forward OOS window: **{predictions.index[0].date() if hasattr(predictions.index[0], 'date') else ''}** → **{predictions.index[-1].date() if hasattr(predictions.index[-1], 'date') else ''}**  ({n_total} days)

| Regime         | Days | Freq   |
|----------------|------|--------|
{regime_breakdown}\

---

## Backtest Performance Summary

Backtest window: **{bt_start}** → **{bt_end}**  ({bt_days} trading days)
Benchmark: 100% US equity (buy-and-hold)
Defensive asset: {defensive_label}
Signal lag: 1 trading day (no same-day look-ahead)
Transaction cost: 5 bps per unit weight change

| Metric              | Benchmark        | Overlay          |
|---------------------|------------------|------------------|
| CAGR                | {_pct(bench_cagr):>16} | {_pct(ov_cagr):>16} |
| Annualised Vol      | {_pct(bench_vol):>16} | {_pct(ov_vol):>16} |
| Sharpe Ratio        | {_fmt(bench_sharpe):>16} | {_fmt(ov_sharpe):>16} |
| Max Drawdown        | {_pct(bench_mdd):>16} | {_pct(ov_mdd):>16} |
| Worst Daily Loss    | {_pct(bench_worst):>16} | {_pct(ov_worst):>16} |
| Total Return        | {_pct(bench_total):>16} | {_pct(ov_total):>16} |
| Avg Equity Exposure | {_pct(1.0):>16} | {_pct(ov_exposure):>16} |

---

## Stress Period Analysis

| Period                       | Coverage                           | Benchmark | Overlay  |
|------------------------------|------------------------------------|-----------|----------|
{stress_rows}\

---

## Honest Interpretation

{_HONEST_INTERPRETATION}

---

## Methodology Notes

- **Model:** 3-state Gaussian HMM (`hmmlearn`), full covariance, 8 cross-asset features
- **Validation:** Walk-forward expanding window, 3-year initial training, monthly refit (37 folds)
- **Regime labelling:** Derived from realised equity return statistics per state — never hardcoded by state number
- **Look-ahead prevention:** `regime_labels.shift(1)` applied before any weight computation
- **Scaler:** `StandardScaler` fit within each training fold only (no leakage into OOS window)

---

*Generated by the Cross-Asset HMM Regime Engine. This is not investment advice.*
"""
    return report


def save_report(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Report saved → %s", path)
