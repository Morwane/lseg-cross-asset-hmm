"""Institutional-quality chart generation — 300 DPI PNG + SVG."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Color palette ────────────────────────────────────────────────────────────
REGIME_COLORS = {
    "risk_on":    "#27ae60",
    "transition": "#f39c12",
    "stress":     "#e74c3c",
}
C_BENCH = "#2c3e50"   # dark slate — benchmark
C_DEF   = "#2471a3"   # blue       — defensive overlay
C_BULL  = "#1e8449"   # green      — bull-aware overlay
C_MOM   = "#7d3c98"   # purple     — regime momentum
C_VOL   = "#148f77"   # teal       — vol managed
C_GRID  = "#e8e8e8"
C_SPINE = "#cccccc"
C_TEXT  = "#1a1a2e"
C_SUB   = "#666666"

# Ordered strategy colour map for consistent legend / bar ordering
STRATEGY_COLORS: dict[str, str] = {
    "benchmark":          C_BENCH,
    "defensive_overlay":  C_DEF,
    "bull_aware_overlay": C_BULL,
    "regime_momentum":    C_MOM,
    "vol_managed":        C_VOL,
}
STRATEGY_LABELS: dict[str, str] = {
    "benchmark":          "Benchmark",
    "defensive_overlay":  "Defensive",
    "bull_aware_overlay": "Bull-Aware",
    "regime_momentum":    "Regime Momentum",
    "vol_managed":        "Vol Managed",
}

# ── Stress events to annotate ─────────────────────────────────────────────────
_EVENTS = [
    {"date": "2025-01-20", "label": "Tariff\nshock", "color": "#c0392b"},
]

DPI = 400
_RIBBON_RATIO = [18, 1]   # main chart : ribbon height ratio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_SPINE)
    ax.spines["bottom"].set_color(C_SPINE)
    ax.tick_params(colors="#444", labelsize=9)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
    ax.grid(axis="x", color=C_GRID, linewidth=0.4, linestyle=":", zorder=0)


def _regime_spans(dates: pd.DatetimeIndex, labels: pd.Series):
    """Yield (start, end, color) tuples from a regime-label series."""
    aligned = labels.reindex(dates).ffill()
    prev, t0 = None, None
    for date, lbl in aligned.items():
        if lbl != prev:
            if prev is not None:
                yield t0, date, REGIME_COLORS.get(str(prev), "#aaaaaa")
            prev, t0 = lbl, date
    if prev is not None:
        yield t0, dates[-1], REGIME_COLORS.get(str(prev), "#aaaaaa")


def _shade(ax: plt.Axes, dates: pd.DatetimeIndex, labels: pd.Series,
           alpha: float = 0.08) -> None:
    for s, e, c in _regime_spans(dates, labels):
        ax.axvspan(s, e, color=c, alpha=alpha, linewidth=0, zorder=1)


def _ribbon(ax: plt.Axes, dates: pd.DatetimeIndex, labels: pd.Series) -> None:
    """Draw a solid regime-color ribbon on a dedicated thin axis."""
    for s, e, c in _regime_spans(dates, labels):
        ax.axvspan(s, e, color=c, alpha=1.0, linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel("Regime", fontsize=7, color=C_SUB, rotation=0, labelpad=28, va="center")
    for sp in ax.spines.values():
        sp.set_visible(False)


def _annotate_events(ax: plt.Axes, xlim=None) -> None:
    lo, hi = xlim if xlim else (None, None)
    for ev in _EVENTS:
        t = pd.Timestamp(ev["date"])
        if lo and t < lo:
            continue
        if hi and t > hi:
            continue
        ax.axvline(t, color=ev["color"], linewidth=1.0, linestyle="--", alpha=0.7, zorder=5)
        ax.text(t, ax.get_ylim()[1], f" {ev['label']}",
                fontsize=7, color=ev["color"], va="top", alpha=0.85)


def _drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    return cum / cum.cummax() - 1


def _save(fig: plt.Figure, path: Path, dpi: int = DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    svg = path.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s + .svg", path.name)


def _patches() -> list:
    return [Patch(color=REGIME_COLORS["risk_on"],    label="Risk On",    alpha=0.8),
            Patch(color=REGIME_COLORS["transition"], label="Transition", alpha=0.8),
            Patch(color=REGIME_COLORS["stress"],     label="Stress",     alpha=0.8)]


def _with_ribbon(figsize=(20, 8), sharex=True):
    """Return (fig, ax_main, ax_ribbon) with a thin ribbon subplot."""
    fig, (ax, rib) = plt.subplots(
        2, 1, figsize=figsize, sharex=sharex,
        gridspec_kw={"height_ratios": _RIBBON_RATIO},
    )
    fig.set_facecolor("white")
    fig.subplots_adjust(hspace=0.04, left=0.07, right=0.97, top=0.88, bottom=0.10)
    return fig, ax, rib


# ─────────────────────────────────────────────────────────────────────────────
# Chart 01 — US equity price with regime overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_regime_overlay(panel: pd.DataFrame, predictions: pd.DataFrame,
                               path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    oos_start = predictions.index[0]
    price = panel["us_equity"].loc[oos_start:]

    _shade(ax, price.index, predictions["regime_label"])
    ax.plot(price.index, price.values, color=C_BENCH, linewidth=1.4, zorder=4)
    _style(ax)

    ax.set_title("US Equity Price — HMM Regime Overlay", fontsize=14,
                 fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Out-of-sample walk-forward predictions  |  Ribbon = active regime",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Price Level", fontsize=10, color="#444")
    ax.legend(handles=_patches(), loc="upper left", framealpha=0.9, fontsize=9,
              edgecolor=C_SPINE)
    ax.tick_params(labelbottom=False)

    xlim = (price.index[0], price.index[-1])
    _annotate_events(ax, xlim)

    _ribbon(rib, price.index, predictions["regime_label"])
    rib.set_xlim(xlim)

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 02 — Regime probabilities stacked area
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_probabilities(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.5))
    fig.set_facecolor("white")

    probs = predictions[["prob_risk_on", "prob_transition", "prob_stress"]].copy().fillna(0.0)
    row_sums = probs.sum(axis=1).replace(0, 1)
    probs = probs.div(row_sums, axis=0)

    ax.stackplot(
        probs.index,
        probs["prob_risk_on"],
        probs["prob_transition"],
        probs["prob_stress"],
        labels=["Risk On", "Transition", "Stress"],
        colors=[REGIME_COLORS["risk_on"], REGIME_COLORS["transition"], REGIME_COLORS["stress"]],
        alpha=0.88,
    )
    _style(ax)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("HMM Regime Probabilities", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Walk-forward out-of-sample  |  3-state Gaussian HMM, monthly refit",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Probability", fontsize=10, color="#444")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9, edgecolor=C_SPINE)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7)

    _annotate_events(ax, (probs.index[0], probs.index[-1]))

    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 03 — Average yield curve by regime
# ─────────────────────────────────────────────────────────────────────────────

def plot_yield_curve_by_regime(features: pd.DataFrame, predictions: pd.DataFrame,
                                path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.set_facecolor("white")

    tenor_cols   = ["us_2y_yield", "us_5y_yield", "us_10y_yield", "us_30y_yield"]
    tenor_labels = ["2Y", "5Y", "10Y", "30Y"]

    oos = features.reindex(predictions.index).copy()
    oos["regime_label"] = predictions["regime_label"]

    regimes = ["risk_on", "transition", "stress"]
    x = np.arange(len(tenor_cols))
    width = 0.26

    for i, regime in enumerate(regimes):
        subset = oos[oos["regime_label"] == regime]
        if len(subset) == 0:
            continue
        means = [subset[t].mean() for t in tenor_cols if t in subset.columns]
        if len(means) != len(tenor_cols):
            continue
        ax.bar(x + i * width, means, width=width,
               label=regime.replace("_", " ").title(),
               color=REGIME_COLORS[regime], alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(tenor_labels, fontsize=10)
    _style(ax)
    ax.set_title("Average Yield Curve by Regime", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Mean yield (%) per tenor, grouped by OOS regime label",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Yield (%)", fontsize=10, color="#444")
    ax.legend(framealpha=0.9, fontsize=9, edgecolor=C_SPINE)

    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 04 — 10Y-2Y slope by regime (box plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_slope_by_regime(features: pd.DataFrame, predictions: pd.DataFrame,
                          path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.set_facecolor("white")

    oos = features.reindex(predictions.index).copy()
    oos["regime_label"] = predictions["regime_label"]

    regimes = ["risk_on", "transition", "stress"]
    data, labels, colors = [], [], []
    for regime in regimes:
        vals = oos[oos["regime_label"] == regime]["slope_10y_2y"].dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(regime.replace("_", " ").title())
            colors.append(REGIME_COLORS[regime])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", linewidth=2.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set_color("#777777")

    ax.axhline(0, color="#555", linestyle="--", linewidth=0.9, alpha=0.6)
    _style(ax)
    ax.set_title("10Y–2Y Yield Curve Slope by Regime", fontsize=14,
                 fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Distribution of slope (pp) per OOS regime  |  Dashed = flat curve",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("10Y − 2Y (percentage points)", fontsize=10, color="#444")

    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 05 — Strategy vs benchmark (cumulative returns)
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_vs_benchmark(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    bench_cum = (1 + daily_returns["benchmark_ret"].fillna(0)).cumprod()
    def_cum   = (1 + daily_returns["overlay_ret"].fillna(0)).cumprod()

    _shade(ax, daily_returns.index, daily_returns.get("regime_label",
           pd.Series(dtype=str)))
    ax.plot(bench_cum.index, bench_cum.values, color=C_BENCH, linewidth=1.6,
            label="Benchmark (100% equity)", zorder=4)
    ax.plot(def_cum.index,   def_cum.values,   color=C_DEF,   linewidth=1.6,
            label="Defensive overlay",         zorder=4)
    _style(ax)

    ax.set_title("Cumulative Returns: Defensive Overlay vs Benchmark", fontsize=14,
                 fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Growth of $1  |  1-day lagged signal  |  5 bps transaction costs",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Growth of $1", fontsize=10, color="#444")
    ax.legend(framealpha=0.9, fontsize=9, edgecolor=C_SPINE)
    ax.tick_params(labelbottom=False)

    xlim = (daily_returns.index[0], daily_returns.index[-1])
    _annotate_events(ax, xlim)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim(xlim)

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 06 — Drawdown comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_drawdown_comparison(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    bench_dd = _drawdown_series(daily_returns["benchmark_ret"])
    def_dd   = _drawdown_series(daily_returns["overlay_ret"])

    ax.fill_between(bench_dd.index, bench_dd.values, 0,
                    alpha=0.25, color=C_BENCH, label="Benchmark")
    ax.fill_between(def_dd.index,   def_dd.values,   0,
                    alpha=0.40, color=C_DEF,   label="Defensive overlay")
    ax.plot(bench_dd.index, bench_dd.values, color=C_BENCH, linewidth=1.2, zorder=4)
    ax.plot(def_dd.index,   def_dd.values,   color=C_DEF,   linewidth=1.2, zorder=4)
    _style(ax)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Drawdown: Defensive Overlay vs Benchmark", fontsize=14,
                 fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Peak-to-trough drawdown on daily NAV",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Drawdown", fontsize=10, color="#444")
    ax.legend(framealpha=0.9, fontsize=9, edgecolor=C_SPINE)
    ax.tick_params(labelbottom=False)

    xlim = (daily_returns.index[0], daily_returns.index[-1])
    _annotate_events(ax, xlim)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim(xlim)

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 07 — Regime feature heatmap (mean z-score)
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_feature_heatmap(features: pd.DataFrame, predictions: pd.DataFrame,
                                 core_features: list[str], path: Path) -> None:
    available = [f for f in core_features if f in features.columns]
    if not available:
        logger.warning("No core features found — skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.set_facecolor("white")

    oos = features[available].reindex(predictions.index).copy()
    oos["regime_label"] = predictions["regime_label"]

    mu  = oos[available].mean()
    std = oos[available].std().replace(0, 1e-12)
    z   = (oos[available] - mu) / std
    z["regime_label"] = oos["regime_label"]

    regimes = ["risk_on", "transition", "stress"]
    matrix = np.array([
        z[z["regime_label"] == regime][available].mean().values
        for regime in regimes
    ])

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")

    ax.set_xticks(np.arange(len(available)))
    ax.set_yticks(np.arange(len(regimes)))
    ax.set_xticklabels([f.replace("_", "\n") for f in available], fontsize=8.5)
    ax.set_yticklabels([r.replace("_", " ").title() for r in regimes], fontsize=10)

    for i in range(len(regimes)):
        for j in range(len(available)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            tc = "white" if abs(val) > 1.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=tc,
                    fontweight="bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Mean Z-Score", fontsize=9, color="#444")
    cb.ax.tick_params(labelsize=8)

    ax.set_title("Regime Feature Heatmap — Mean Z-Score by Regime", fontsize=13,
                 fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.03, "Green = feature elevated  |  Red = feature depressed  |  OOS window",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 08 — Latest dashboard (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_latest_dashboard(panel: pd.DataFrame, features: pd.DataFrame,
                           predictions: pd.DataFrame, daily_returns: pd.DataFrame,
                           metrics_df: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(18, 11))
    fig.set_facecolor("white")
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── P1: Latest regime + probability bars ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")
    latest    = predictions.iloc[-1]
    regime    = str(latest.get("regime_label", "unknown"))
    probs_v   = [float(latest.get("prob_risk_on", 0)),
                 float(latest.get("prob_transition", 0)),
                 float(latest.get("prob_stress", 0))]
    b_labels  = ["Risk On", "Transition", "Stress"]
    b_colors  = [REGIME_COLORS["risk_on"], REGIME_COLORS["transition"], REGIME_COLORS["stress"]]

    bars = ax1.barh(b_labels, probs_v, color=b_colors, alpha=0.88, height=0.48)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax1.set_xlim(0, 1.0)
    for bar, val in zip(bars, probs_v):
        ax1.text(min(val + 0.02, 0.88), bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=10, fontweight="bold")
    dt = latest.name.date() if hasattr(latest.name, "date") else str(latest.name)
    weight_map = {"risk_on": 1.0, "transition": 0.5, "stress": 0.0}
    eq_w = weight_map.get(regime, 0.5)
    ax1.set_title(
        f"Latest Regime: {regime.replace('_', ' ').upper()}\n"
        f"Signal date: {dt}  |  Defensive equity weight: {eq_w:.0%}",
        fontsize=10.5, fontweight="bold", color=REGIME_COLORS.get(regime, C_TEXT),
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", color=C_GRID, linewidth=0.7)

    # ── P2: Current yield curve ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("white")
    tenor_c = ["us_2y_yield", "us_5y_yield", "us_10y_yield", "us_30y_yield"]
    tenor_x = ["2Y", "5Y", "10Y", "30Y"]
    lf = features.iloc[-1]
    yc = [(l, lf.get(t, np.nan)) for l, t in zip(tenor_x, tenor_c)
          if not (isinstance(lf.get(t, np.nan), float) and np.isnan(lf.get(t, np.nan)))]
    if yc:
        tl, tv = zip(*yc)
        ax2.plot(list(tl), list(tv), marker="o", color="#2980b9", linewidth=2.2,
                 markersize=8, zorder=4)
        ax2.fill_between(list(tl), list(tv), min(tv) * 0.97,
                         alpha=0.12, color="#2980b9")
        fd = features.index[-1].date() if hasattr(features.index[-1], "date") else ""
        ax2.set_title(f"Current Yield Curve  ({fd})", fontsize=10.5, fontweight="bold",
                      color=C_TEXT)
        ax2.set_ylabel("Yield (%)", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(color=C_GRID, linewidth=0.7)

    # ── P3: Backtest metrics table ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    show = [("cagr",             "CAGR",          lambda v: f"{v*100:.1f}%"),
            ("annualised_vol",   "Ann. Vol",       lambda v: f"{v*100:.1f}%"),
            ("sharpe",           "Sharpe",         lambda v: f"{v:.2f}"),
            ("max_drawdown",     "Max Drawdown",   lambda v: f"{v*100:.1f}%"),
            ("worst_daily_loss", "Worst Day",      lambda v: f"{v*100:.2f}%"),
            ("total_return",     "Total Return",   lambda v: f"{v*100:.1f}%")]
    m = metrics_df.set_index("metric")
    td = []
    for key, lbl, fmt in show:
        if key in m.index:
            b = float(m.loc[key, "benchmark"])
            o = float(m.loc[key, "overlay"])
            td.append([lbl, fmt(b), fmt(o)])
    tbl = ax3.table(cellText=td, colLabels=["Metric", "Benchmark", "Def. Overlay"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.15, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#dddddd")
    ax3.set_title("Backtest Metrics (OOS)", fontsize=10.5, fontweight="bold",
                  color=C_TEXT, pad=6)

    # ── P4: Regime time allocation pie ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("white")
    counts = predictions["regime_label"].value_counts()
    pie_c = [REGIME_COLORS.get(l, "#aaa") for l in counts.index]
    wedges, texts, autotexts = ax4.pie(
        counts.values,
        labels=[l.replace("_", " ").title() for l in counts.index],
        colors=pie_c, autopct="%1.1f%%", startangle=90, pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in texts:
        t.set_fontsize(10)
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax4.set_title("Regime Time Allocation (OOS)", fontsize=10.5, fontweight="bold",
                  color=C_TEXT)

    fig.suptitle("Cross-Asset HMM Regime Engine — Defensive Overlay Dashboard",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.08)
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 09 — Three-way cumulative returns
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_way_performance(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    bench = (1 + daily_returns["benchmark_ret"].fillna(0)).cumprod()
    def_  = (1 + daily_returns["overlay_ret"].fillna(0)).cumprod()
    bull  = (1 + daily_returns["bull_aware_ret"].fillna(0)).cumprod()

    _shade(ax, daily_returns.index,
           daily_returns.get("regime_label", pd.Series(dtype=str)))

    ax.plot(bench.index, bench.values, color=C_BENCH, linewidth=1.8, linestyle="-",
            label="Benchmark (100% equity)", zorder=5)
    ax.plot(bull.index,  bull.values,  color=C_BULL,  linewidth=1.8, linestyle="-",
            label="Bull-aware overlay", zorder=5)
    ax.plot(def_.index,  def_.values,  color=C_DEF,   linewidth=1.8, linestyle="--",
            label="Defensive overlay", zorder=5)
    _style(ax)

    ax.set_title("Three-Way Cumulative Returns", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=10)
    ax.text(0, 1.01,
            "Benchmark  ·  Bull-aware (prob-weighted + risk controls)  ·  Defensive (fixed regime weights)",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Growth of $1", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE, loc="upper left")
    ax.tick_params(labelbottom=False)

    xlim = (daily_returns.index[0], daily_returns.index[-1])
    _annotate_events(ax, xlim)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim(xlim)

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 10 — Three-way drawdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_three_way_drawdown(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    bench_dd = _drawdown_series(daily_returns["benchmark_ret"])
    def_dd   = _drawdown_series(daily_returns["overlay_ret"])
    bull_dd  = _drawdown_series(daily_returns["bull_aware_ret"])

    ax.fill_between(bench_dd.index, bench_dd.values, 0, alpha=0.18, color=C_BENCH)
    ax.fill_between(bull_dd.index,  bull_dd.values,  0, alpha=0.18, color=C_BULL)
    ax.fill_between(def_dd.index,   def_dd.values,   0, alpha=0.25, color=C_DEF)
    ax.plot(bench_dd.index, bench_dd.values, color=C_BENCH, linewidth=1.4,
            label="Benchmark", zorder=5)
    ax.plot(bull_dd.index,  bull_dd.values,  color=C_BULL,  linewidth=1.4,
            label="Bull-aware", zorder=5)
    ax.plot(def_dd.index,   def_dd.values,   color=C_DEF,   linewidth=1.4,
            linestyle="--", label="Defensive", zorder=5)
    _style(ax)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Three-Way Drawdown Comparison", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Peak-to-trough drawdown on daily NAV",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Drawdown", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE)
    ax.tick_params(labelbottom=False)

    xlim = (daily_returns.index[0], daily_returns.index[-1])
    _annotate_events(ax, xlim)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim(xlim)

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 11 — Equity exposure comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_exposure_comparison(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    has_bull = "bull_aware_equity_weight" in daily_returns.columns

    ax.plot(daily_returns.index, np.ones(len(daily_returns)),
            color=C_BENCH, linewidth=1.2, linestyle=":", label="Benchmark (100%)", alpha=0.7)
    if has_bull:
        ax.plot(daily_returns.index, daily_returns["bull_aware_equity_weight"].values,
                color=C_BULL, linewidth=1.3, alpha=0.85, label="Bull-aware overlay")
    ax.plot(daily_returns.index, daily_returns["equity_weight"].values,
            color=C_DEF, linewidth=1.3, alpha=0.85,
            linestyle="--" if has_bull else "-", label="Defensive overlay")
    _style(ax)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("Equity Exposure Over Time", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Daily equity weight applied (1-day lagged signal)",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Equity Weight", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE)
    ax.tick_params(labelbottom=False)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim((daily_returns.index[0], daily_returns.index[-1]))

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 12 — Bull-aware dashboard (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bull_aware_dashboard(daily_returns: pd.DataFrame,
                               metrics_comparison: pd.DataFrame,
                               path: Path) -> None:
    fig = plt.figure(figsize=(18, 11))
    fig.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.40)

    # ── P1: 3-way metrics table ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    show = [("cagr",               "CAGR",         lambda v: f"{v*100:.1f}%"),
            ("annualised_vol",     "Ann. Vol",      lambda v: f"{v*100:.1f}%"),
            ("sharpe",             "Sharpe",        lambda v: f"{v:.2f}"),
            ("max_drawdown",       "Max Drawdown",  lambda v: f"{v*100:.1f}%"),
            ("worst_daily_loss",   "Worst Day",     lambda v: f"{v*100:.2f}%"),
            ("avg_equity_exposure","Avg Eq. Exp",   lambda v: f"{v*100:.1f}%"),
            ("total_transaction_cost","Total TC",   lambda v: f"{v*100:.2f}%")]
    mc = metrics_comparison.set_index("metric")
    td, headers = [], ["Metric", "Benchmark", "Defensive", "Bull-Aware"]
    for key, lbl, fmt in show:
        if key in mc.index:
            row = [lbl]
            for col in ["benchmark", "defensive_overlay", "bull_aware_overlay"]:
                try:
                    row.append(fmt(float(mc.loc[key, col])))
                except (KeyError, ValueError):
                    row.append("n/a")
            td.append(row)
    tbl = ax1.table(cellText=td, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.75)
    hdr_colors = ["#2c3e50", C_BENCH, C_DEF, C_BULL]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor(hdr_colors[c] if c < len(hdr_colors) else "#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#f5f5f5")
    ax1.set_title("Three-Way Metrics Comparison (OOS)", fontsize=10.5, fontweight="bold",
                  color=C_TEXT, pad=6)

    # ── P2: 3-way cumulative returns (mini) ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("white")
    bench = (1 + daily_returns["benchmark_ret"].fillna(0)).cumprod()
    def_  = (1 + daily_returns["overlay_ret"].fillna(0)).cumprod()
    bull  = (1 + daily_returns["bull_aware_ret"].fillna(0)).cumprod()
    ax2.plot(bench.index, bench.values, color=C_BENCH, linewidth=1.5, label="Benchmark")
    ax2.plot(bull.index,  bull.values,  color=C_BULL,  linewidth=1.5, label="Bull-aware")
    ax2.plot(def_.index,  def_.values,  color=C_DEF,   linewidth=1.5, linestyle="--",
             label="Defensive")
    _style(ax2)
    ax2.set_title("Cumulative Returns", fontsize=10.5, fontweight="bold", color=C_TEXT)
    ax2.set_ylabel("Growth of $1", fontsize=9)
    ax2.legend(framealpha=0.9, fontsize=8, edgecolor=C_SPINE)
    ax2.tick_params(axis="x", labelsize=8)

    # ── P3: Equity exposure timeline ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("white")
    if "bull_aware_equity_weight" in daily_returns.columns:
        ax3.plot(daily_returns.index, daily_returns["bull_aware_equity_weight"].values,
                 color=C_BULL, linewidth=1.2, label="Bull-aware")
    ax3.plot(daily_returns.index, daily_returns["equity_weight"].values,
             color=C_DEF, linewidth=1.2, linestyle="--", label="Defensive")
    ax3.axhline(1.0, color=C_BENCH, linewidth=0.8, linestyle=":", alpha=0.6,
                label="Benchmark")
    _style(ax3)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax3.set_ylim(-0.05, 1.10)
    ax3.set_title("Equity Exposure Comparison", fontsize=10.5, fontweight="bold",
                  color=C_TEXT)
    ax3.set_ylabel("Equity Weight", fontsize=9)
    ax3.legend(framealpha=0.9, fontsize=8, edgecolor=C_SPINE)
    ax3.tick_params(axis="x", labelsize=8)

    # ── P4: Drawdown comparison ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("white")
    bench_dd = _drawdown_series(daily_returns["benchmark_ret"])
    def_dd   = _drawdown_series(daily_returns["overlay_ret"])
    bull_dd  = _drawdown_series(daily_returns["bull_aware_ret"])
    ax4.plot(bench_dd.index, bench_dd.values, color=C_BENCH, linewidth=1.3,
             label="Benchmark")
    ax4.plot(bull_dd.index,  bull_dd.values,  color=C_BULL,  linewidth=1.3,
             label="Bull-aware")
    ax4.plot(def_dd.index,   def_dd.values,   color=C_DEF,   linewidth=1.3,
             linestyle="--", label="Defensive")
    _style(ax4)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax4.set_title("Drawdown Comparison", fontsize=10.5, fontweight="bold", color=C_TEXT)
    ax4.set_ylabel("Drawdown", fontsize=9)
    ax4.legend(framealpha=0.9, fontsize=8, edgecolor=C_SPINE)
    ax4.tick_params(axis="x", labelsize=8)

    fig.suptitle("Cross-Asset HMM Regime Engine — Bull-Aware Overlay Dashboard",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.08)
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 13 — Five-strategy cumulative returns
# ─────────────────────────────────────────────────────────────────────────────

def plot_five_strategy_performance(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    bench = (1 + daily_returns["benchmark_ret"].fillna(0)).cumprod()
    def_  = (1 + daily_returns["overlay_ret"].fillna(0)).cumprod()
    bull  = (1 + daily_returns["bull_aware_ret"].fillna(0)).cumprod()
    mom   = (1 + daily_returns["regime_momentum_ret"].fillna(0)).cumprod()
    vol   = (1 + daily_returns["vol_managed_ret"].fillna(0)).cumprod()

    if "regime_label" in daily_returns.columns:
        _shade(ax, daily_returns.index, daily_returns["regime_label"])

    ax.plot(bench.index, bench.values, color=C_BENCH, linewidth=1.8,
            label="Benchmark (100% equity)", zorder=6)
    ax.plot(mom.index,   mom.values,   color=C_MOM,   linewidth=1.6,
            label="Regime momentum", zorder=5)
    ax.plot(vol.index,   vol.values,   color=C_VOL,   linewidth=1.6,
            label="Vol managed", zorder=5)
    ax.plot(bull.index,  bull.values,  color=C_BULL,  linewidth=1.4, linestyle="--",
            label="Bull-aware", zorder=4)
    ax.plot(def_.index,  def_.values,  color=C_DEF,   linewidth=1.4, linestyle=":",
            label="Defensive", zorder=4)
    _style(ax)

    ax.set_title("Five-Strategy Cumulative Returns", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=10)
    ax.text(0, 1.01,
            "Benchmark  ·  Regime momentum  ·  Vol managed  ·  Bull-aware  ·  Defensive  "
            "|  1-day lag  |  5 bps TC",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Growth of $1", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE, loc="upper left")
    ax.tick_params(labelbottom=False)

    xlim = (daily_returns.index[0], daily_returns.index[-1])
    _annotate_events(ax, xlim)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim(xlim)

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 14 — Five-strategy drawdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_five_strategy_drawdown(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    bench_dd = _drawdown_series(daily_returns["benchmark_ret"])
    def_dd   = _drawdown_series(daily_returns["overlay_ret"])
    bull_dd  = _drawdown_series(daily_returns["bull_aware_ret"])
    mom_dd   = _drawdown_series(daily_returns["regime_momentum_ret"])
    vol_dd   = _drawdown_series(daily_returns["vol_managed_ret"])

    ax.fill_between(bench_dd.index, bench_dd.values, 0, alpha=0.12, color=C_BENCH)
    ax.plot(bench_dd.index, bench_dd.values, color=C_BENCH, linewidth=1.6,
            label="Benchmark", zorder=6)
    ax.plot(mom_dd.index,   mom_dd.values,   color=C_MOM,   linewidth=1.4,
            label="Regime momentum", zorder=5)
    ax.plot(vol_dd.index,   vol_dd.values,   color=C_VOL,   linewidth=1.4,
            label="Vol managed", zorder=5)
    ax.plot(bull_dd.index,  bull_dd.values,  color=C_BULL,  linewidth=1.2, linestyle="--",
            label="Bull-aware", zorder=4)
    ax.plot(def_dd.index,   def_dd.values,   color=C_DEF,   linewidth=1.2, linestyle=":",
            label="Defensive", zorder=4)
    _style(ax)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Five-Strategy Drawdown Comparison", fontsize=14, fontweight="bold",
                 color=C_TEXT, pad=10)
    ax.text(0, 1.01, "Peak-to-trough drawdown on daily NAV  |  All strategies share same OOS index",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Drawdown", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE)
    ax.tick_params(labelbottom=False)

    xlim = (daily_returns.index[0], daily_returns.index[-1])
    _annotate_events(ax, xlim)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim(xlim)

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 15 — Alpha overlay equity exposure
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_exposure(daily_returns: pd.DataFrame, path: Path) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    ax.axhline(1.0, color=C_BENCH, linewidth=0.9, linestyle=":", alpha=0.5,
               label="Benchmark (100%)")
    if "regime_momentum_equity_weight" in daily_returns.columns:
        ax.plot(daily_returns.index, daily_returns["regime_momentum_equity_weight"].values,
                color=C_MOM, linewidth=1.3, alpha=0.88, label="Regime momentum")
    if "vol_managed_equity_weight" in daily_returns.columns:
        ax.plot(daily_returns.index, daily_returns["vol_managed_equity_weight"].values,
                color=C_VOL, linewidth=1.3, alpha=0.88, label="Vol managed",
                linestyle="--")
    if "bull_aware_equity_weight" in daily_returns.columns:
        ax.plot(daily_returns.index, daily_returns["bull_aware_equity_weight"].values,
                color=C_BULL, linewidth=1.0, alpha=0.60, label="Bull-aware",
                linestyle=":")
    _style(ax)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0.15, 1.15)
    ax.set_title("Alpha Overlay Equity Exposure Over Time", fontsize=14,
                 fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01,
            "Regime momentum  ·  Vol managed  ·  Bull-aware  |  1-day lagged signal",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Equity Weight", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE)
    ax.tick_params(labelbottom=False)

    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim((daily_returns.index[0], daily_returns.index[-1]))

    # margins set in _with_ribbon
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 16 — Bull-market participation bars
# ─────────────────────────────────────────────────────────────────────────────

def plot_bull_participation_bars(bull_participation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.set_facecolor("white")

    if len(bull_participation) == 0:
        logger.warning("Bull participation DataFrame is empty — skipping chart 16.")
        plt.close(fig)
        return

    row = bull_participation.iloc[0]
    strategies = {
        "Benchmark":       ("benchmark_return",          C_BENCH),
        "Regime Momentum": ("regime_momentum_return",    C_MOM),
        "Vol Managed":     ("vol_managed_return",        C_VOL),
        "Bull-Aware":      ("bull_aware_overlay_return", C_BULL),
        "Defensive":       ("defensive_overlay_return",  C_DEF),
    }

    labels, values, colors = [], [], []
    for label, (col, color) in strategies.items():
        val = row.get(col, float("nan"))
        if not (isinstance(val, float) and np.isnan(val)):
            labels.append(label)
            values.append(float(val))
            colors.append(color)

    x = np.arange(len(labels))
    bars = ax.bar(x, [v * 100 for v in values], color=colors, alpha=0.88,
                  width=0.55, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        y_pos = bar.get_height() + 0.3 if bar.get_height() >= 0 else bar.get_height() - 1.2
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    _style(ax)

    n_bull = int(row.get("n_days_in_bull", 0))
    pct_bull = float(row.get("pct_time_in_bull", 0))
    ax.set_title("Bull-Market Participation by Strategy", fontsize=14,
                 fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01,
            f"Rule: 60d return > 0 AND price > 200d MA  |  "
            f"{n_bull} bull days ({pct_bull:.1f}% of OOS)",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Cumulative Return During Bull Periods (%)", fontsize=10, color="#444")

    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 17 — Alpha overlay dashboard (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_dashboard(daily_returns: pd.DataFrame,
                          metrics_comparison: pd.DataFrame,
                          bull_participation: pd.DataFrame,
                          path: Path) -> None:
    fig = plt.figure(figsize=(18, 11))
    fig.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.40)

    # ── P1: 5-strategy metrics table ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    show = [("cagr",               "CAGR",         lambda v: f"{v*100:.1f}%"),
            ("annualised_vol",     "Ann. Vol",      lambda v: f"{v*100:.1f}%"),
            ("sharpe",             "Sharpe",        lambda v: f"{v:.2f}"),
            ("max_drawdown",       "Max DD",        lambda v: f"{v*100:.1f}%"),
            ("avg_equity_exposure","Avg Eq. Exp",   lambda v: f"{v*100:.1f}%"),
            ("total_transaction_cost","Total TC",   lambda v: f"{v*100:.2f}%")]
    mc = metrics_comparison.set_index("metric")
    strat_cols = ["benchmark", "defensive_overlay", "bull_aware_overlay",
                  "regime_momentum", "vol_managed"]
    strat_labels = ["Bench", "Def", "Bull", "Mom", "Vol"]
    td = []
    for key, lbl, fmt in show:
        if key in mc.index:
            row = [lbl]
            for col in strat_cols:
                try:
                    row.append(fmt(float(mc.loc[key, col])))
                except (KeyError, ValueError):
                    row.append("n/a")
            td.append(row)
    headers = ["Metric"] + strat_labels
    tbl = ax1.table(cellText=td, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.05, 1.65)
    hdr_colors = ["#2c3e50", C_DEF, C_BULL, C_MOM, C_VOL]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            fc = hdr_colors[c - 1] if c > 0 and c - 1 < len(hdr_colors) else "#2c3e50"
            cell.set_facecolor(fc)
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#f5f5f5")
    ax1.set_title("Five-Strategy Metrics (OOS)", fontsize=10.5, fontweight="bold",
                  color=C_TEXT, pad=6)

    # ── P2: 5-strategy cumulative returns (mini) ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("white")
    bench = (1 + daily_returns["benchmark_ret"].fillna(0)).cumprod()
    def_  = (1 + daily_returns["overlay_ret"].fillna(0)).cumprod()
    bull  = (1 + daily_returns["bull_aware_ret"].fillna(0)).cumprod()
    mom   = (1 + daily_returns["regime_momentum_ret"].fillna(0)).cumprod()
    vol   = (1 + daily_returns["vol_managed_ret"].fillna(0)).cumprod()
    ax2.plot(bench.index, bench.values, color=C_BENCH, linewidth=1.5, label="Benchmark")
    ax2.plot(mom.index,   mom.values,   color=C_MOM,   linewidth=1.3, label="Mom")
    ax2.plot(vol.index,   vol.values,   color=C_VOL,   linewidth=1.3, label="Vol")
    ax2.plot(bull.index,  bull.values,  color=C_BULL,  linewidth=1.1, linestyle="--",
             label="Bull")
    ax2.plot(def_.index,  def_.values,  color=C_DEF,   linewidth=1.1, linestyle=":",
             label="Def")
    _style(ax2)
    ax2.set_title("Cumulative Returns", fontsize=10.5, fontweight="bold", color=C_TEXT)
    ax2.set_ylabel("Growth of $1", fontsize=9)
    ax2.legend(framealpha=0.9, fontsize=8, edgecolor=C_SPINE, ncol=3)
    ax2.tick_params(axis="x", labelsize=8)

    # ── P3: Alpha overlay exposure ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("white")
    ax3.axhline(1.0, color=C_BENCH, linewidth=0.8, linestyle=":", alpha=0.5)
    if "regime_momentum_equity_weight" in daily_returns.columns:
        ax3.plot(daily_returns.index, daily_returns["regime_momentum_equity_weight"].values,
                 color=C_MOM, linewidth=1.2, label="Regime momentum")
    if "vol_managed_equity_weight" in daily_returns.columns:
        ax3.plot(daily_returns.index, daily_returns["vol_managed_equity_weight"].values,
                 color=C_VOL, linewidth=1.2, linestyle="--", label="Vol managed")
    _style(ax3)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax3.set_ylim(0.15, 1.15)
    ax3.set_title("Alpha Overlay Exposure", fontsize=10.5, fontweight="bold", color=C_TEXT)
    ax3.set_ylabel("Equity Weight", fontsize=9)
    ax3.legend(framealpha=0.9, fontsize=8, edgecolor=C_SPINE)
    ax3.tick_params(axis="x", labelsize=8)

    # ── P4: Bull participation bars ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("white")
    if len(bull_participation) > 0:
        row = bull_participation.iloc[0]
        strategies_p = {
            "Bench": ("benchmark_return",          C_BENCH),
            "Mom":   ("regime_momentum_return",    C_MOM),
            "Vol":   ("vol_managed_return",        C_VOL),
            "Bull":  ("bull_aware_overlay_return", C_BULL),
            "Def":   ("defensive_overlay_return",  C_DEF),
        }
        lbls, vals, clrs = [], [], []
        for label, (col, color) in strategies_p.items():
            val = row.get(col, float("nan"))
            if not (isinstance(val, float) and np.isnan(val)):
                lbls.append(label)
                vals.append(float(val) * 100)
                clrs.append(color)
        if lbls:
            xp = np.arange(len(lbls))
            ax4.bar(xp, vals, color=clrs, alpha=0.88, width=0.55,
                    edgecolor="white", linewidth=0.5)
            ax4.set_xticks(xp)
            ax4.set_xticklabels(lbls, fontsize=9)
            ax4.axhline(0, color="#555", linewidth=0.8)
            ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
            _style(ax4)
    ax4.set_title("Bull-Market Returns by Strategy", fontsize=10.5, fontweight="bold",
                  color=C_TEXT)
    ax4.set_ylabel("Cum. Return (%)", fontsize=9)
    ax4.tick_params(axis="x", labelsize=8)

    fig.suptitle("Cross-Asset HMM Regime Engine — Alpha Overlay Dashboard",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.08)
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 18 — Cumulative simulated PnL in USD
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_pnl(daily_returns: pd.DataFrame, path: Path,
                         initial_capital: float = 100_000.0) -> None:
    fig, ax, rib = _with_ribbon(figsize=(20, 8))

    strats = [
        ("benchmark_ret",        "Benchmark",       C_BENCH, "-",  1.8),
        ("regime_momentum_ret",  "Regime Momentum", C_MOM,   "-",  1.5),
        ("vol_managed_ret",      "Vol Managed",     C_VOL,   "-",  1.5),
        ("bull_aware_ret",       "Bull-Aware",      C_BULL,  "--", 1.3),
        ("overlay_ret",          "Defensive",       C_DEF,   ":",  1.3),
    ]
    for col, lbl, clr, ls, lw in strats:
        if col not in daily_returns.columns:
            continue
        cum = (1 + daily_returns[col].fillna(0)).cumprod()
        pnl = (cum - 1) * initial_capital
        ax.plot(pnl.index, pnl.values, color=clr, linewidth=lw,
                linestyle=ls, label=lbl, zorder=5)

    if "regime_label" in daily_returns.columns:
        _shade(ax, daily_returns.index, daily_returns["regime_label"])

    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"${x:,.0f}" if abs(x) < 1e6 else f"${x/1e3:,.0f}k"
    ))
    _style(ax)
    ax.set_title("Simulated Cumulative PnL — $100k Notional",
                 fontsize=14, fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01,
            "All strategies  |  1-day lagged signal  |  5 bps transaction costs  "
            "|  Not realised PnL — backtested simulation",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Simulated PnL (USD)", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE, loc="upper left")
    ax.tick_params(labelbottom=False)
    xlim = (daily_returns.index[0], daily_returns.index[-1])
    _annotate_events(ax, xlim)
    if "regime_label" in daily_returns.columns:
        _ribbon(rib, daily_returns.index, daily_returns["regime_label"])
    rib.set_xlim(xlim)
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 19 — Daily PnL distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_daily_pnl_distribution(daily_returns: pd.DataFrame, path: Path,
                                  initial_capital: float = 100_000.0) -> None:
    strats = [
        ("benchmark_ret",       "Benchmark",       C_BENCH),
        ("regime_momentum_ret", "Regime Momentum", C_MOM),
        ("vol_managed_ret",     "Vol Managed",     C_VOL),
        ("bull_aware_ret",      "Bull-Aware",      C_BULL),
        ("overlay_ret",         "Defensive",       C_DEF),
    ]
    available = [(col, lbl, clr) for col, lbl, clr in strats
                 if col in daily_returns.columns]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6), sharey=False)
    fig.set_facecolor("white")
    if n == 1:
        axes = [axes]

    for ax, (col, lbl, clr) in zip(axes, available):
        daily_pnl = daily_returns[col].dropna() * initial_capital
        ax.hist(daily_pnl, bins=50, color=clr, alpha=0.75, edgecolor="white", linewidth=0.3)
        mu, sigma = daily_pnl.mean(), daily_pnl.std()
        ax.axvline(mu, color="#333", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axvline(daily_pnl.quantile(0.05), color="#e74c3c", linewidth=1.0,
                   linestyle=":", alpha=0.8, label="5th pct")
        _style(ax)
        ax.set_title(lbl, fontsize=10, fontweight="bold", color=C_TEXT)
        ax.set_xlabel("Daily PnL (USD)", fontsize=8.5, color="#444")
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda x, _: f"${x:,.0f}"
        ))
        ax.text(0.97, 0.97, f"μ=${mu:,.0f}\nσ=${sigma:,.0f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                color="#333")

    axes[0].set_ylabel("Frequency", fontsize=9, color="#444")
    fig.suptitle("Daily Simulated PnL Distribution — $100k Notional",
                 fontsize=13, fontweight="bold", color=C_TEXT, y=1.02)
    fig.text(0.5, -0.01, "Dashed = mean  |  Dotted = 5th percentile  |  Not realised PnL",
             ha="center", fontsize=8.5, color=C_SUB)
    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 20 — Final simulated PnL bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_pnl_bars(economic_impact: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.set_facecolor("white")

    df = economic_impact.copy()
    if "simulated_pnl" not in df.columns or len(df) == 0:
        plt.close(fig)
        return

    df = df.sort_values("simulated_pnl", ascending=True)
    colors = [STRATEGY_COLORS.get(s, "#aaaaaa") for s in df["strategy"]]
    labels = [STRATEGY_LABELS.get(s, s) for s in df["strategy"]]

    bars = ax.barh(labels, df["simulated_pnl"].values,
                   color=colors, alpha=0.88, height=0.55,
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df["simulated_pnl"].values):
        xpos = val + (abs(val) * 0.015) if val >= 0 else val - (abs(val) * 0.015)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", ha=ha, fontsize=10, fontweight="bold")

    ax.axvline(0, color="#555", linewidth=0.8)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"${x:,.0f}" if abs(x) < 1e6 else f"${x/1e3:,.0f}k"
    ))
    _style(ax)
    ax.set_title("Final Simulated PnL by Strategy — $100k Notional",
                 fontsize=14, fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01,
            "Not realised PnL — backtested simulation  |  OOS period  |  5 bps TC",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_xlabel("Simulated PnL (USD)", fontsize=10, color="#444")
    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 21 — PnL by regime
# ─────────────────────────────────────────────────────────────────────────────

def plot_pnl_by_regime(regime_pnl: pd.DataFrame, path: Path) -> None:
    if len(regime_pnl) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor("white")

    regimes   = [r for r in ["risk_on", "transition", "stress"] if r in regime_pnl["regime"].values]
    strategies = list(regime_pnl["strategy"].unique())
    n_strats   = len(strategies)
    x          = np.arange(len(regimes))
    width      = 0.75 / n_strats

    for i, strat in enumerate(strategies):
        sub  = regime_pnl[regime_pnl["strategy"] == strat].set_index("regime")
        vals = [float(sub.loc[r, "cumulative_pnl_usd"]) if r in sub.index
                else float("nan") for r in regimes]
        offset = (i - n_strats / 2 + 0.5) * width
        clr  = STRATEGY_COLORS.get(strat, "#aaaaaa")
        lbl  = STRATEGY_LABELS.get(strat, strat)
        ax.bar(x + offset, vals, width=width * 0.92, color=clr, alpha=0.85,
               label=lbl, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").title() for r in regimes], fontsize=11)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda v, _: f"${v:,.0f}" if abs(v) < 1e6 else f"${v/1e3:,.0f}k"
    ))
    _style(ax)
    ax.set_title("Simulated PnL by Regime — $100k Notional",
                 fontsize=14, fontweight="bold", color=C_TEXT, pad=10)
    ax.text(0, 1.01,
            "Cumulative PnL accumulated during each HMM regime  |  Not realised PnL",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.set_ylabel("Cumulative PnL (USD)", fontsize=10, color="#444")
    ax.legend(framealpha=0.92, fontsize=9, edgecolor=C_SPINE,
              loc="upper right", ncol=2)
    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 22 — Transaction cost sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_tc_sensitivity(tc_sensitivity: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.set_facecolor("white")

    non_bench = [s for s in tc_sensitivity["strategy"].unique() if s != "benchmark"]
    tc_levels = sorted(tc_sensitivity["tc_bps"].unique())

    for ax, metric, ylabel, title in [
        (axes[0], "sharpe",  "Sharpe Ratio",      "Sharpe Ratio vs Transaction Cost"),
        (axes[1], "cagr",    "CAGR",              "CAGR vs Transaction Cost"),
    ]:
        for strat in non_bench:
            sub = tc_sensitivity[tc_sensitivity["strategy"] == strat].sort_values("tc_bps")
            clr = STRATEGY_COLORS.get(strat, "#aaaaaa")
            lbl = STRATEGY_LABELS.get(strat, strat)
            ax.plot(sub["tc_bps"], sub[metric], marker="o", color=clr,
                    linewidth=1.8, markersize=5, label=lbl)
        # Benchmark reference (horizontal line at 5bps row = same regardless of TC)
        bench = tc_sensitivity[tc_sensitivity["strategy"] == "benchmark"]
        if len(bench) > 0:
            val = float(bench[metric].iloc[0])
            ax.axhline(val, color=C_BENCH, linewidth=1.2, linestyle="--",
                       alpha=0.7, label="Benchmark")
        _style(ax)
        if metric == "cagr":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_title(title, fontsize=12, fontweight="bold", color=C_TEXT)
        ax.set_xlabel("Transaction Cost (bps)", fontsize=10, color="#444")
        ax.set_ylabel(ylabel, fontsize=10, color="#444")
        ax.set_xticks(tc_levels)
        ax.legend(framealpha=0.9, fontsize=9, edgecolor=C_SPINE)

    fig.suptitle("Robustness Check 1: Transaction Cost Sensitivity",
                 fontsize=14, fontweight="bold", color=C_TEXT)
    fig.text(0.5, -0.01, "How sensitive are Sharpe and CAGR to the TC assumption?",
             ha="center", fontsize=9, color=C_SUB)
    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 23 — Subperiod performance
# ─────────────────────────────────────────────────────────────────────────────

def plot_subperiod_performance(subperiod_df: pd.DataFrame, path: Path) -> None:
    if len(subperiod_df) == 0:
        return

    subperiods = list(subperiod_df["subperiod"].unique())
    strategies = list(subperiod_df["strategy"].unique())
    n_strats   = len(strategies)
    x          = np.arange(len(subperiods))
    width      = 0.75 / n_strats

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.set_facecolor("white")

    for ax, metric, ylabel, fmt_fn in [
        (axes[0], "total_return", "Total Return",
         lambda v: f"{v*100:.1f}%"),
        (axes[1], "sharpe",       "Sharpe Ratio",
         lambda v: f"{v:.2f}"),
    ]:
        for i, strat in enumerate(strategies):
            sub  = subperiod_df[subperiod_df["strategy"] == strat].copy()
            vals = []
            for sp in subperiods:
                row = sub[sub["subperiod"] == sp]
                vals.append(float(row[metric].iloc[0]) if len(row) > 0 else float("nan"))
            offset = (i - n_strats / 2 + 0.5) * width
            clr = STRATEGY_COLORS.get(strat, "#aaaaaa")
            lbl = STRATEGY_LABELS.get(strat, strat)
            valid = [v for v in vals if not np.isnan(v)]
            vvals = [v if not np.isnan(v) else 0 for v in vals]
            ax.bar(x + offset, vvals, width=width * 0.92, color=clr,
                   alpha=0.85, label=lbl, edgecolor="white", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(subperiods, fontsize=9, rotation=10, ha="right")
        ax.axhline(0, color="#555", linewidth=0.8)
        _style(ax)
        if metric == "total_return":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_title(f"Subperiod {ylabel}", fontsize=12, fontweight="bold", color=C_TEXT)
        ax.set_ylabel(ylabel, fontsize=10, color="#444")
        ax.legend(framealpha=0.9, fontsize=8.5, edgecolor=C_SPINE, ncol=2)

    fig.suptitle("Robustness Check 2: Subperiod Performance",
                 fontsize=14, fontweight="bold", color=C_TEXT)
    fig.text(0.5, -0.01,
             "2023 recovery  ·  2024 bull market  ·  2025 tariff correction  ·  2026 latest",
             ha="center", fontsize=9, color=C_SUB)
    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 24 — Strategy ranking heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategy_ranking(ranking_df: pd.DataFrame, path: Path) -> None:
    if len(ranking_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.set_facecolor("white")

    rank_cols = [c for c in ranking_df.columns if c.endswith("_rank") and c != "overall_rank_rank"]
    if "overall_rank" in ranking_df.columns:
        rank_cols = [c for c in rank_cols] + (
            ["overall_rank"] if "overall_rank" in ranking_df.columns and
            "overall_rank" not in rank_cols else []
        )

    strategies = [STRATEGY_LABELS.get(s, s) for s in ranking_df["strategy"]]
    metric_labels = {
        "cagr_rank":                    "CAGR",
        "sharpe_rank":                  "Sharpe",
        "calmar_rank":                  "Calmar",
        "max_drawdown_rank":            "Max DD\n(lower better)",
        "worst_daily_loss_rank":        "Worst Day\n(lower better)",
        "avg_equity_exposure_rank":     "Avg Eq Exp\n(lower=defensive)",
        "total_transaction_cost_rank":  "Total TC\n(lower better)",
        "overall_rank":                 "Overall",
    }

    display_cols = [c for c in rank_cols if c in metric_labels]
    if not display_cols:
        display_cols = rank_cols[:8]

    matrix = np.array([
        [float(ranking_df.loc[ranking_df.index[i], c])
         if c in ranking_df.columns else float("nan")
         for c in display_cols]
        for i in range(len(ranking_df))
    ])
    n_strats, n_metrics = matrix.shape

    n_valid = (~np.isnan(matrix)).sum(axis=0)
    max_rank = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 5
    norm_matrix = matrix / max_rank  # [0, 1], lower = better rank

    im = ax.imshow(norm_matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_strats))
    ax.set_xticklabels([metric_labels.get(c, c.replace("_rank", "")) for c in display_cols],
                       fontsize=9)
    ax.set_yticklabels(strategies, fontsize=10)

    for i in range(n_strats):
        for j in range(n_metrics):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            tc = "white" if (norm_matrix[i, j] > 0.6 or norm_matrix[i, j] < 0.2) else "black"
            ax.text(j, i, f"#{int(val)}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=tc)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Rank (dark green = #1)", fontsize=9, color="#444")
    cb.ax.tick_params(labelsize=8)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["#1 (best)", "Mid", "Last"])

    ax.set_title("Robustness Check 3: Multi-Objective Strategy Ranking",
                 fontsize=13, fontweight="bold", color=C_TEXT, pad=12)
    ax.text(0, 1.03,
            "Dark green = #1 best for that objective  |  1 = best rank  |  Overall = average rank",
            transform=ax.transAxes, fontsize=8.5, color=C_SUB)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_charts(
    panel_path: Path,
    features_path: Path,
    predictions_path: Path,
    daily_returns_path: Path,
    metrics_path: Path,
    charts_dir: Path,
    model_cfg: dict,
    metrics_comparison_path: Path | None = None,
    bull_participation_path: Path | None = None,
    economic_impact_path: Path | None = None,
    tc_sensitivity_path: Path | None = None,
    subperiod_path: Path | None = None,
    strategy_ranking_path: Path | None = None,
) -> list[Path]:
    """Load all inputs and generate all charts (up to 24). Returns list of saved paths."""
    charts_dir.mkdir(parents=True, exist_ok=True)

    panel       = pd.read_csv(panel_path,         index_col=0, parse_dates=True)
    features    = pd.read_csv(features_path,      index_col=0, parse_dates=True)
    predictions = pd.read_csv(predictions_path,   index_col=0, parse_dates=True)
    daily       = pd.read_csv(daily_returns_path,  index_col=0, parse_dates=True)
    metrics_df  = pd.read_csv(metrics_path)

    metrics_cmp = None
    if metrics_comparison_path and metrics_comparison_path.exists():
        metrics_cmp = pd.read_csv(metrics_comparison_path)

    bull_part = pd.DataFrame()
    if bull_participation_path and bull_participation_path.exists():
        bull_part = pd.read_csv(bull_participation_path)

    eco_impact = pd.DataFrame()
    if economic_impact_path and economic_impact_path.exists():
        eco_impact = pd.read_csv(economic_impact_path)

    tc_sens = pd.DataFrame()
    if tc_sensitivity_path and tc_sensitivity_path.exists():
        tc_sens = pd.read_csv(tc_sensitivity_path)

    subperiod = pd.DataFrame()
    if subperiod_path and subperiod_path.exists():
        subperiod = pd.read_csv(subperiod_path)

    ranking = pd.DataFrame()
    if strategy_ranking_path and strategy_ranking_path.exists():
        ranking = pd.read_csv(strategy_ranking_path)

    core_features: list[str] = model_cfg.get("features", {}).get("hmm_core", [])

    def p(name: str) -> Path:
        return charts_dir / name

    specs: list[tuple[str, object]] = [
        ("01_price_regime_overlay.png",
         lambda: plot_price_regime_overlay(panel, predictions, p("01_price_regime_overlay.png"))),
        ("02_regime_probabilities.png",
         lambda: plot_regime_probabilities(predictions, p("02_regime_probabilities.png"))),
        ("03_yield_curve_by_regime.png",
         lambda: plot_yield_curve_by_regime(features, predictions, p("03_yield_curve_by_regime.png"))),
        ("04_slope_10y_2y_by_regime.png",
         lambda: plot_slope_by_regime(features, predictions, p("04_slope_10y_2y_by_regime.png"))),
        ("05_strategy_vs_benchmark.png",
         lambda: plot_strategy_vs_benchmark(daily, p("05_strategy_vs_benchmark.png"))),
        ("06_drawdown_comparison.png",
         lambda: plot_drawdown_comparison(daily, p("06_drawdown_comparison.png"))),
        ("07_regime_feature_heatmap.png",
         lambda: plot_regime_feature_heatmap(features, predictions, core_features,
                                              p("07_regime_feature_heatmap.png"))),
        ("08_latest_dashboard.png",
         lambda: plot_latest_dashboard(panel, features, predictions, daily, metrics_df,
                                        p("08_latest_dashboard.png"))),
    ]

    if "bull_aware_ret" in daily.columns:
        specs += [
            ("09_three_way_performance_comparison.png",
             lambda: plot_three_way_performance(daily,
                                                 p("09_three_way_performance_comparison.png"))),
            ("10_three_way_drawdown_comparison.png",
             lambda: plot_three_way_drawdown(daily,
                                              p("10_three_way_drawdown_comparison.png"))),
            ("11_equity_exposure_comparison.png",
             lambda: plot_equity_exposure_comparison(daily,
                                                      p("11_equity_exposure_comparison.png"))),
        ]
        if metrics_cmp is not None:
            specs.append(
                ("12_bull_aware_dashboard.png",
                 lambda: plot_bull_aware_dashboard(daily, metrics_cmp,
                                                   p("12_bull_aware_dashboard.png")))
            )

    if "regime_momentum_ret" in daily.columns and "vol_managed_ret" in daily.columns:
        specs += [
            ("13_five_strategy_performance.png",
             lambda: plot_five_strategy_performance(
                 daily, p("13_five_strategy_performance.png"))),
            ("14_five_strategy_drawdown.png",
             lambda: plot_five_strategy_drawdown(
                 daily, p("14_five_strategy_drawdown.png"))),
            ("15_alpha_exposure.png",
             lambda: plot_alpha_exposure(daily, p("15_alpha_exposure.png"))),
        ]
        if len(bull_part) > 0:
            specs.append(
                ("16_bull_participation_bars.png",
                 lambda: plot_bull_participation_bars(
                     bull_part, p("16_bull_participation_bars.png")))
            )
        if metrics_cmp is not None and len(bull_part) > 0:
            specs.append(
                ("17_alpha_dashboard.png",
                 lambda: plot_alpha_dashboard(
                     daily, metrics_cmp, bull_part, p("17_alpha_dashboard.png")))
            )

    # PnL charts (18-21) — generated when benchmark_ret is present
    if "benchmark_ret" in daily.columns:
        specs.append(
            ("18_cumulative_pnl_comparison.png",
             lambda: plot_cumulative_pnl(daily, p("18_cumulative_pnl_comparison.png")))
        )
        specs.append(
            ("19_daily_pnl_distribution.png",
             lambda: plot_daily_pnl_distribution(
                 daily, p("19_daily_pnl_distribution.png")))
        )
    if len(eco_impact) > 0:
        specs.append(
            ("20_final_pnl_by_strategy.png",
             lambda: plot_final_pnl_bars(eco_impact, p("20_final_pnl_by_strategy.png")))
        )
    # PnL by regime — derived from daily_df
    if "regime_label" in daily.columns and "benchmark_ret" in daily.columns:
        from src.pnl import compute_regime_pnl
        regime_pnl_df = compute_regime_pnl(daily)
        if len(regime_pnl_df) > 0:
            _rpnl = regime_pnl_df  # capture for lambda
            specs.append(
                ("21_pnl_by_regime.png",
                 lambda: plot_pnl_by_regime(_rpnl, p("21_pnl_by_regime.png")))
            )

    # Robustness charts (22-24)
    if len(tc_sens) > 0:
        specs.append(
            ("22_transaction_cost_sensitivity.png",
             lambda: plot_tc_sensitivity(tc_sens, p("22_transaction_cost_sensitivity.png")))
        )
    if len(subperiod) > 0:
        specs.append(
            ("23_subperiod_performance.png",
             lambda: plot_subperiod_performance(
                 subperiod, p("23_subperiod_performance.png")))
        )
    if len(ranking) > 0:
        specs.append(
            ("24_strategy_ranking.png",
             lambda: plot_strategy_ranking(ranking, p("24_strategy_ranking.png")))
        )

    saved = []
    for name, fn in specs:
        try:
            fn()
            saved.append(charts_dir / name)
        except Exception as exc:
            logger.error("Chart %s failed: %s", name, exc, exc_info=True)

    return saved