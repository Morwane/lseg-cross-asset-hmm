"""Regime risk-overlay backtest engine.

Signal source: output/reports/walk_forward_regime_predictions.csv (OOS only).
All regime signals are lagged by one trading day before being applied to returns,
ensuring no same-day look-ahead bias.

Weight rule (from model.yaml):
  risk_on    → equity_weight = 1.0,  defensive_weight = 0.0
  transition → equity_weight = 0.5,  defensive_weight = 0.5
  stress     → equity_weight = 0.0,  defensive_weight = 1.0

Transaction cost: abs(Δweight) × cost_bps / 10_000 deducted from daily return.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_all_metrics, TRADING_DAYS

logger = logging.getLogger(__name__)

_CASH_LABEL = "cash (return=0)"

# ---------------------------------------------------------------------------
# Bull-aware overlay helpers
# ---------------------------------------------------------------------------


def compute_bull_aware_weights(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    lag: int = 1,
) -> pd.Series:
    """Probability-weighted equity weight with trend and volatility risk controls.

    Base rule (applied to lag-1 probabilities):
        w = 1.00 * P(risk_on) + 0.75 * P(transition) + 0.25 * P(stress)

    Risk controls (all panel inputs lagged by `lag` days, no look-ahead):
        Cap:   if VIX > expanding 75th-pct AND equity drawdown < -5%,
               cap weight at 0.50
        Floor: if equity price > 200-day MA AND P(stress) < 0.70,
               floor weight at 0.50  (cap overrides floor if both fire)

    Final weight clipped to [0, 1].
    """
    prob_ro = predictions["prob_risk_on"].shift(lag)
    prob_tr = predictions["prob_transition"].shift(lag)
    prob_st = predictions["prob_stress"].shift(lag)

    w = (1.00 * prob_ro + 0.75 * prob_tr + 0.25 * prob_st).clip(0, 1)

    equity_price = panel["us_equity"]
    vix = panel["vix"]

    # Running peak-to-trough drawdown on equity
    cum_eq = (1 + equity_price.pct_change().fillna(0)).cumprod()
    drawdown = cum_eq / cum_eq.cummax() - 1

    # 200-day MA
    ma200 = equity_price.rolling(200, min_periods=1).mean()

    # Expanding 75th-pct of VIX — shift(1) so at time t it uses data through t-1
    vix_75th = vix.expanding(min_periods=63).quantile(0.75).shift(1)

    # All panel signals with 1-day lag, aligned to prediction index
    idx = predictions.index
    vix_l   = vix.shift(lag).reindex(idx)
    vix75_l = vix_75th.reindex(idx)
    dd_l    = drawdown.shift(lag).reindex(idx)
    price_l = equity_price.shift(lag).reindex(idx)
    ma200_l = ma200.shift(lag).reindex(idx)
    prob_st_l = prob_st.reindex(idx)

    # Floor: bull trend condition
    floor_mask = (price_l > ma200_l) & (prob_st_l < 0.70)
    w = w.where(~floor_mask, w.clip(lower=0.50))

    # Cap: stress condition (overrides floor)
    cap_mask = (vix_l > vix75_l) & (dd_l < -0.05)
    w = w.where(~cap_mask, w.clip(upper=0.50))

    return w.clip(0, 1).dropna()


def compute_regime_momentum_weights(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    lag: int = 1,
) -> pd.Series:
    """Probability-weighted base weight with a time-series momentum overlay.

    Base rule (lagged probabilities):
        base = 1.00*P(risk_on) + 0.75*P(transition) + 0.25*P(stress)

    Momentum adjustment (lagged 63d and 126d returns + 200d MA filter):
        momentum_score = 0.5*sign(ret_63d) + 0.5*sign(ret_126d)
        if momentum_score > 0 and price > 200d MA:  w = base + 0.20
        elif momentum_score < 0 and price < 200d MA: w = base - 0.20
        else: w = base

    Hard clip to [0.25, 1.00].
    Stress cap: if P(stress) > 0.75 → w = min(w, 0.50).
    All signals lagged by `lag` days.
    """
    prob_ro = predictions["prob_risk_on"].shift(lag)
    prob_tr = predictions["prob_transition"].shift(lag)
    prob_st = predictions["prob_stress"].shift(lag)
    base    = (1.00 * prob_ro + 0.75 * prob_tr + 0.25 * prob_st).clip(0, 1)

    price   = panel["us_equity"]
    ret_63  = price.pct_change(63).shift(lag)
    ret_126 = price.pct_change(126).shift(lag)
    ma200   = price.rolling(200, min_periods=1).mean().shift(lag)
    price_l = price.shift(lag)

    idx       = predictions.index
    base      = base.reindex(idx)
    ret_63_l  = ret_63.reindex(idx)
    ret_126_l = ret_126.reindex(idx)
    ma200_l   = ma200.reindex(idx)
    price_l   = price_l.reindex(idx)
    prob_st_l = prob_st.reindex(idx)

    mom_score  = (0.5 * np.sign(ret_63_l.fillna(0)) +
                  0.5 * np.sign(ret_126_l.fillna(0)))
    bull_trend = (mom_score > 0) & (price_l > ma200_l)
    bear_trend = (mom_score < 0) & (price_l < ma200_l)

    w = base.copy()
    w = w.where(~bull_trend, w + 0.20)
    w = w.where(~bear_trend, w - 0.20)
    w = w.clip(0.25, 1.00)

    # Stress cap (overrides clip)
    stress_cap = prob_st_l > 0.75
    w = w.where(~stress_cap, w.clip(upper=0.50))

    return w.clip(0.25, 1.00).dropna()


def compute_vol_managed_weights(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    lag: int = 1,
    target_vol: float = 0.16,
) -> pd.Series:
    """Volatility-managed probability-weighted equity weight.

    Base rule (lagged probabilities):
        base = 1.00*P(risk_on) + 0.75*P(transition) + 0.25*P(stress)

    Vol scaling (lagged 20-day realised vol):
        vol_scale = clip(target_vol / realised_vol_20d, 0.50, 1.25)
        w = clip(base * vol_scale, 0.25, 1.00)

    Bull floor: if price > 200d MA and P(stress) < 0.70 → w = max(w, 0.50)
    Stress cap: if VIX > expanding-75th-pct and drawdown < -5% → w = min(w, 0.50)
    All signals lagged by `lag` days.
    """
    prob_ro = predictions["prob_risk_on"].shift(lag)
    prob_tr = predictions["prob_transition"].shift(lag)
    prob_st = predictions["prob_stress"].shift(lag)
    base    = (1.00 * prob_ro + 0.75 * prob_tr + 0.25 * prob_st).clip(0, 1)

    price   = panel["us_equity"]
    vix     = panel["vix"]

    # 20-day realised annualised vol (computed from panel prices — no features dependency)
    real_vol_20d = price.pct_change().rolling(20, min_periods=5).std() * np.sqrt(252)
    cum_eq   = (1 + price.pct_change().fillna(0)).cumprod()
    drawdown = cum_eq / cum_eq.cummax() - 1
    ma200    = price.rolling(200, min_periods=1).mean()
    vix_75th = vix.expanding(min_periods=63).quantile(0.75).shift(1)

    idx       = predictions.index
    base      = base.reindex(idx)
    rv_l      = real_vol_20d.shift(lag).reindex(idx)
    price_l   = price.shift(lag).reindex(idx)
    ma200_l   = ma200.shift(lag).reindex(idx)
    vix_l     = vix.shift(lag).reindex(idx)
    vix75_l   = vix_75th.reindex(idx)
    dd_l      = drawdown.shift(lag).reindex(idx)
    prob_st_l = prob_st.reindex(idx)

    vol_scale = (target_vol / rv_l.replace(0, np.nan)).clip(0.50, 1.25)
    w = (base * vol_scale).clip(0.25, 1.00)

    floor_mask = (price_l > ma200_l) & (prob_st_l < 0.70)
    w = w.where(~floor_mask, w.clip(lower=0.50))

    cap_mask = (vix_l > vix75_l) & (dd_l < -0.05)
    w = w.where(~cap_mask, w.clip(upper=0.50))

    return w.clip(0.25, 1.00).dropna()


def compute_bull_market_periods(panel: pd.DataFrame) -> pd.Series:
    """Boolean Series: True on days that qualify as a systematic bull period.

    Rule: 60-day benchmark return > 0  AND  equity price > 200-day moving average.
    This is an evaluation-only label — not used in any trading decision.
    """
    price = panel["us_equity"]
    ret_60d = price.pct_change(60)
    ma200   = price.rolling(200, min_periods=1).mean()
    return (ret_60d > 0) & (price > ma200)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_regime_predictions(path: Path) -> pd.DataFrame:
    """Load walk-forward regime predictions CSV with DatetimeIndex."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_return_series(
    cleaned_path: Path,
    defensive_col: str = "SHY.O",
) -> tuple[pd.Series, pd.Series, str]:
    """Load daily equity and defensive returns from the cleaned panel.

    Returns (equity_ret, defensive_ret, defensive_label).
    If the defensive column is absent, defensive_ret is all zeros (cash proxy).
    """
    panel = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    panel.index.name = "date"

    equity_ret = panel["us_equity"].pct_change(1)

    if defensive_col in panel.columns:
        defensive_ret = panel[defensive_col].pct_change(1)
        defensive_label = defensive_col
        logger.info("Defensive asset: %s (from cleaned panel)", defensive_col)
    else:
        defensive_ret = pd.Series(0.0, index=panel.index, dtype=float)
        defensive_label = _CASH_LABEL
        logger.warning(
            "'%s' not found in cleaned panel — using cash (return=0) as defensive asset.",
            defensive_col,
        )

    return equity_ret, defensive_ret, defensive_label


# ---------------------------------------------------------------------------
# Weight construction (pure functions — testable without I/O)
# ---------------------------------------------------------------------------


def _normalise_label(label: str, regime_mapping: dict[str, float]) -> str:
    """Map variant labels (transition_1, state_N) to a canonical mapping key."""
    if label in regime_mapping:
        return label
    if label.startswith("transition"):
        return "transition"
    # Ambiguous / generic state label — default to neutral
    return "transition"


def apply_weight_map(
    regime_labels: pd.Series,
    regime_mapping: dict[str, float],
    lag: int = 1,
) -> pd.Series:
    """Apply a 1-day lag to regime labels and map to equity weights.

    The lag ensures the weight on day t was decided using the signal from day t-1,
    i.e. no same-day look-ahead bias.

    Returns a Series of equity weights. The first `lag` rows are NaN and should
    be dropped by the caller.
    """
    normalised = regime_labels.map(lambda x: _normalise_label(x, regime_mapping))
    equity_weights = normalised.map(regime_mapping)
    return equity_weights.shift(lag)


def compute_transaction_costs(
    equity_weights: pd.Series,
    transaction_cost_bps: int = 5,
) -> pd.Series:
    """Compute daily transaction costs from changes in equity weight.

    On the first day the position is entered from 0 (no prior holding),
    so the cost on that day equals abs(weight) × cost_rate.
    """
    prior_weights = equity_weights.shift(1).fillna(0.0)
    weight_change = (equity_weights - prior_weights).abs()
    return weight_change * transaction_cost_bps / 10_000


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def compute_portfolio_returns(
    equity_returns: pd.Series,
    defensive_returns: pd.Series,
    equity_weights: pd.Series,     # already lagged, no NaN
    transaction_cost_bps: int = 5,
) -> pd.DataFrame:
    """Build daily overlay and benchmark return series over the backtest window.

    Aligns all inputs to the equity_weights index (the OOS prediction period
    after the lag drop). Returns a DataFrame with columns:
      benchmark_ret   — 100% equity buy-and-hold
      overlay_ret     — regime-weighted portfolio net of transaction costs
      equity_weight   — equity weight applied today (from yesterday's signal)
      defensive_weight
      transaction_cost
    """
    # Align to the weight index (OOS window after lag drop)
    idx = equity_weights.index
    eq_ret = equity_returns.reindex(idx)
    def_ret = defensive_returns.reindex(idx)

    missing_eq = eq_ret.isna().sum()
    missing_def = def_ret.isna().sum()
    if missing_eq > 0:
        logger.warning("%d equity return NaNs in backtest window.", missing_eq)
    if missing_def > 0:
        logger.warning("%d defensive return NaNs in backtest window — filling with 0.", missing_def)
        def_ret = def_ret.fillna(0.0)

    def_weights = 1.0 - equity_weights
    costs = compute_transaction_costs(equity_weights, transaction_cost_bps)
    gross_overlay = equity_weights * eq_ret + def_weights * def_ret
    net_overlay = gross_overlay - costs

    return pd.DataFrame(
        {
            "benchmark_ret": eq_ret,
            "overlay_ret": net_overlay,
            "equity_weight": equity_weights,
            "defensive_weight": def_weights,
            "transaction_cost": costs,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Stress period analysis
# ---------------------------------------------------------------------------


def compute_stress_analysis(
    daily_df: pd.DataFrame,
    regime_predictions: pd.DataFrame,
    stress_periods: list[dict],
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Compute return and drawdown statistics for each named stress period.

    Periods that fall entirely before the backtest window are reported with
    NaN returns and a note in the 'coverage' column.
    """
    backtest_start = daily_df.index[0]
    backtest_end = daily_df.index[-1]

    rows = []
    for sp in stress_periods:
        name = sp["name"]
        start = pd.Timestamp(sp["start"])
        end = pd.Timestamp(sp["end"])

        # Intersect with backtest period
        eff_start = max(start, backtest_start)
        eff_end = min(end, backtest_end)

        if eff_start > eff_end:
            rows.append({
                "period": name,
                "start": sp["start"],
                "end": sp["end"],
                "coverage": "outside backtest window",
                "n_days": 0,
                "benchmark_total_ret": float("nan"),
                "overlay_total_ret": float("nan"),
                "benchmark_max_drawdown": float("nan"),
                "overlay_max_drawdown": float("nan"),
                "dominant_regime": "n/a",
            })
            continue

        window = daily_df.loc[eff_start:eff_end]
        if len(window) == 0:
            continue

        bench_ret = window["benchmark_ret"].dropna()
        ov_ret = window["overlay_ret"].dropna()

        bench_total = float((1 + bench_ret).prod() - 1)
        ov_total = float((1 + ov_ret).prod() - 1)

        bench_mdd = compute_all_metrics(bench_ret)["max_drawdown"]
        ov_mdd = compute_all_metrics(ov_ret)["max_drawdown"]

        # Regime breakdown during this window
        if "regime_label" in daily_df.columns:
            dominant = daily_df.loc[eff_start:eff_end, "regime_label"].mode()
            dominant_label = dominant.iloc[0] if len(dominant) > 0 else "n/a"
        else:
            dominant_label = "n/a"

        partial = eff_start > start or eff_end < end
        coverage = f"partial ({eff_start.date()}→{eff_end.date()})" if partial else "full"

        rows.append({
            "period": name,
            "start": sp["start"],
            "end": sp["end"],
            "coverage": coverage,
            "n_days": len(window),
            "benchmark_total_ret": bench_total,
            "overlay_total_ret": ov_total,
            "benchmark_max_drawdown": bench_mdd,
            "overlay_max_drawdown": ov_mdd,
            "dominant_regime": dominant_label,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


def compute_strategy_exposure_summary(
    daily_df: pd.DataFrame,
    strategy_weight_cols: dict[str, str],
) -> pd.DataFrame:
    """Exposure statistics segmented by bull/non-bull period and regime label.

    Args:
        daily_df: backtest DataFrame with regime_label, bull_period columns.
        strategy_weight_cols: mapping of strategy name → equity_weight column name.

    Returns a DataFrame with rows per (strategy, segment) and avg_equity_exposure.
    """
    rows = []
    for strat, wcol in strategy_weight_cols.items():
        if wcol not in daily_df.columns:
            continue
        w = daily_df[wcol]
        # Overall
        rows.append({"strategy": strat, "segment": "overall",
                     "avg_equity_exposure": float(w.mean())})
        # Bull vs non-bull
        if "bull_period" in daily_df.columns:
            for bull_flag, label in [(True, "bull_market"), (False, "non_bull")]:
                mask = daily_df["bull_period"] == bull_flag
                rows.append({"strategy": strat, "segment": label,
                             "avg_equity_exposure": float(w[mask].mean()) if mask.any() else float("nan")})
        # By regime
        if "regime_label" in daily_df.columns:
            for regime in ["risk_on", "transition", "stress"]:
                mask = daily_df["regime_label"] == regime
                rows.append({"strategy": strat, "segment": f"regime_{regime}",
                             "avg_equity_exposure": float(w[mask].mean()) if mask.any() else float("nan")})
    return pd.DataFrame(rows)


def run_backtest(
    predictions_path: Path,
    cleaned_path: Path,
    model_cfg: dict,
    transaction_cost_bps: int = 5,
    trading_days: int = TRADING_DAYS,
) -> dict[str, pd.DataFrame]:
    """Run the full regime risk-overlay backtest (all 5 strategies).

    Returns a dict with keys:
        'daily'                     — per-day returns and weights for all strategies
        'metrics'                   — benchmark vs defensive overlay (legacy format)
        'metrics_comparison'        — all five strategies side-by-side
        'stress'                    — defensive overlay stress analysis (legacy format)
        'stress_comparison'         — all five strategies
        'bull_market_participation' — returns during systematic bull periods
        'strategy_exposure_summary' — avg exposure by strategy and segment
        'defensive_label'           — defensive asset name
    """
    overlay_cfg = model_cfg["risk_overlay"]
    regime_mapping: dict[str, float] = overlay_cfg["regime_mapping"]
    defensive_col: str = overlay_cfg["defensive_asset"]
    stress_periods: list[dict] = model_cfg.get("stress_periods", [])

    # Load inputs
    predictions = load_regime_predictions(predictions_path)
    panel = pd.read_csv(cleaned_path, index_col=0, parse_dates=True)
    panel.index.name = "date"

    equity_ret, defensive_ret, defensive_label = load_return_series(
        cleaned_path, defensive_col,
    )

    logger.info(
        "Predictions: %d rows  [%s → %s]",
        len(predictions), predictions.index[0].date(), predictions.index[-1].date(),
    )
    logger.info("Defensive asset: %s", defensive_label)

    # ── Defensive overlay (fixed weights per regime) ───────────────────────
    equity_weights = apply_weight_map(predictions["regime_label"], regime_mapping, lag=1)
    equity_weights = equity_weights.dropna()

    n_dropped = len(predictions) - len(equity_weights)
    logger.info(
        "Backtest window after 1-day lag: %d rows  [%s → %s]  (%d leading row dropped)",
        len(equity_weights),
        equity_weights.index[0].date(), equity_weights.index[-1].date(),
        n_dropped,
    )

    daily_df = compute_portfolio_returns(
        equity_ret, defensive_ret, equity_weights, transaction_cost_bps,
    )

    # Attach lagged regime labels and states
    lagged_labels = predictions["regime_label"].shift(1).dropna()
    lagged_states = predictions["regime_state"].shift(1).dropna().astype(int)
    daily_df["regime_label"] = lagged_labels.reindex(daily_df.index)
    daily_df["regime_state"] = lagged_states.reindex(daily_df.index)

    # ── Bull-aware overlay ─────────────────────────────────────────────────
    bull_weights = compute_bull_aware_weights(predictions, panel, lag=1)

    # ── Regime-momentum overlay ────────────────────────────────────────────
    mom_weights = compute_regime_momentum_weights(predictions, panel, lag=1)

    # ── Vol-managed overlay ────────────────────────────────────────────────
    vol_weights = compute_vol_managed_weights(predictions, panel, lag=1)

    # Align all strategies to a common index (most restrictive = latest start)
    common_idx = (
        daily_df.index
        .intersection(bull_weights.dropna().index)
        .intersection(mom_weights.dropna().index)
        .intersection(vol_weights.dropna().index)
    )
    daily_df       = daily_df.loc[common_idx]
    equity_weights = equity_weights.loc[common_idx]
    bull_weights   = bull_weights.reindex(common_idx)
    mom_weights    = mom_weights.reindex(common_idx)
    vol_weights    = vol_weights.reindex(common_idx)
    eq_ret_common  = equity_ret.reindex(common_idx)
    def_ret_common = defensive_ret.reindex(common_idx)

    # Compute portfolio returns for each alpha overlay
    bull_portfolio = compute_portfolio_returns(
        eq_ret_common, def_ret_common, bull_weights, transaction_cost_bps,
    )
    mom_portfolio = compute_portfolio_returns(
        eq_ret_common, def_ret_common, mom_weights, transaction_cost_bps,
    )
    vol_portfolio = compute_portfolio_returns(
        eq_ret_common, def_ret_common, vol_weights, transaction_cost_bps,
    )

    daily_df["bull_aware_equity_weight"]    = bull_portfolio["equity_weight"]
    daily_df["bull_aware_defensive_weight"] = bull_portfolio["defensive_weight"]
    daily_df["bull_aware_transaction_cost"] = bull_portfolio["transaction_cost"]
    daily_df["bull_aware_ret"]              = bull_portfolio["overlay_ret"]

    daily_df["regime_momentum_equity_weight"]    = mom_portfolio["equity_weight"]
    daily_df["regime_momentum_defensive_weight"] = mom_portfolio["defensive_weight"]
    daily_df["regime_momentum_transaction_cost"] = mom_portfolio["transaction_cost"]
    daily_df["regime_momentum_ret"]              = mom_portfolio["overlay_ret"]

    daily_df["vol_managed_equity_weight"]    = vol_portfolio["equity_weight"]
    daily_df["vol_managed_defensive_weight"] = vol_portfolio["defensive_weight"]
    daily_df["vol_managed_transaction_cost"] = vol_portfolio["transaction_cost"]
    daily_df["vol_managed_ret"]              = vol_portfolio["overlay_ret"]

    logger.info(
        "Bull-aware avg equity exposure: %.1f%%  |  "
        "Regime-momentum: %.1f%%  |  Vol-managed: %.1f%%",
        float(daily_df["bull_aware_equity_weight"].mean()) * 100,
        float(daily_df["regime_momentum_equity_weight"].mean()) * 100,
        float(daily_df["vol_managed_equity_weight"].mean()) * 100,
    )

    # ── Metrics for all five strategies ───────────────────────────────────
    def _extra(df: pd.DataFrame, weight_col: str, cost_col: str, label_col: str | None) -> dict:
        d: dict = {}
        d["avg_equity_exposure"] = float(df[weight_col].mean())
        d["total_transaction_cost"] = float(df[cost_col].sum())
        d["n_weight_changes"] = int((df[weight_col].diff().abs() > 1e-9).sum())
        if label_col and label_col in df.columns:
            d["n_regime_switches"] = int(
                (df[label_col] != df[label_col].shift(1)).sum()
            )
        else:
            d["n_regime_switches"] = 0
        return d

    bench_m = compute_all_metrics(daily_df["benchmark_ret"],       "benchmark",        trading_days)
    def_m   = compute_all_metrics(daily_df["overlay_ret"],         "defensive",        trading_days)
    bull_m  = compute_all_metrics(daily_df["bull_aware_ret"],      "bull_aware",       trading_days)
    mom_m   = compute_all_metrics(daily_df["regime_momentum_ret"], "regime_momentum",  trading_days)
    vol_m   = compute_all_metrics(daily_df["vol_managed_ret"],     "vol_managed",      trading_days)

    bench_m.update({"avg_equity_exposure": 1.0, "total_transaction_cost": 0.0,
                    "n_weight_changes": 0, "n_regime_switches": 0})
    def_m.update(_extra(daily_df, "equity_weight", "transaction_cost", "regime_label"))
    bull_m.update(_extra(daily_df, "bull_aware_equity_weight",
                         "bull_aware_transaction_cost", None))
    mom_m.update(_extra(daily_df, "regime_momentum_equity_weight",
                        "regime_momentum_transaction_cost", None))
    vol_m.update(_extra(daily_df, "vol_managed_equity_weight",
                        "vol_managed_transaction_cost", None))

    # Legacy format (benchmark vs defensive overlay)
    all_keys = [k for k in bench_m if k != "label"]
    metrics_df = pd.DataFrame([
        {"metric": k, "benchmark": bench_m[k], "overlay": def_m[k]}
        for k in all_keys
    ])

    # Five-strategy comparison
    metrics_comparison_df = pd.DataFrame([
        {"metric": k,
         "benchmark":             bench_m[k],
         "defensive_overlay":     def_m[k],
         "bull_aware_overlay":    bull_m[k],
         "regime_momentum":       mom_m[k],
         "vol_managed":           vol_m[k]}
        for k in all_keys
    ])

    # ── Stress analysis ────────────────────────────────────────────────────
    def _stress_for(ret_col: str) -> pd.DataFrame:
        tmp = daily_df[["benchmark_ret", "regime_label"]].copy()
        tmp["overlay_ret"] = daily_df[ret_col]
        return compute_stress_analysis(tmp, predictions, stress_periods, trading_days)

    stress_df       = _stress_for("overlay_ret")
    stress_bull_df  = _stress_for("bull_aware_ret")
    stress_mom_df   = _stress_for("regime_momentum_ret")
    stress_vol_df   = _stress_for("vol_managed_ret")

    stress_cols = ["period", "start", "end", "coverage", "n_days",
                   "benchmark_total_ret", "overlay_total_ret", "dominant_regime"]
    stress_comparison = stress_df[stress_cols].copy().rename(
        columns={"overlay_total_ret": "defensive_total_ret"}
    )
    stress_comparison["bull_aware_total_ret"]   = stress_bull_df["overlay_total_ret"].values
    stress_comparison["regime_momentum_total_ret"] = stress_mom_df["overlay_total_ret"].values
    stress_comparison["vol_managed_total_ret"]  = stress_vol_df["overlay_total_ret"].values

    # ── Bull-market participation ──────────────────────────────────────────
    bull_periods = compute_bull_market_periods(panel).reindex(daily_df.index).fillna(False)
    daily_df["bull_period"] = bull_periods
    bull_window  = daily_df[bull_periods]
    bull_pct     = float(bull_periods.sum()) / max(len(bull_periods), 1)

    def _bull_total(col: str) -> float:
        s = bull_window[col].dropna()
        return float((1 + s).prod() - 1) if len(s) > 0 else float("nan")

    bull_participation_df = pd.DataFrame([{
        "rule": "60d_return>0 AND price>200d_MA",
        "n_days_in_bull": int(bull_periods.sum()),
        "pct_time_in_bull": round(bull_pct * 100, 1),
        "benchmark_return":             _bull_total("benchmark_ret"),
        "defensive_overlay_return":     _bull_total("overlay_ret"),
        "bull_aware_overlay_return":    _bull_total("bull_aware_ret"),
        "regime_momentum_return":       _bull_total("regime_momentum_ret"),
        "vol_managed_return":           _bull_total("vol_managed_ret"),
    }])

    # ── Strategy exposure summary ──────────────────────────────────────────
    exposure_summary = compute_strategy_exposure_summary(
        daily_df,
        {
            "benchmark":          "benchmark_equity_weight",   # placeholder — handled below
            "defensive_overlay":  "equity_weight",
            "bull_aware_overlay": "bull_aware_equity_weight",
            "regime_momentum":    "regime_momentum_equity_weight",
            "vol_managed":        "vol_managed_equity_weight",
        },
    )
    # Benchmark is always 1.0 — insert manually
    bench_rows = pd.DataFrame([
        {"strategy": "benchmark", "segment": seg, "avg_equity_exposure": 1.0}
        for seg in ["overall", "bull_market", "non_bull",
                    "regime_risk_on", "regime_transition", "regime_stress"]
    ])
    exposure_summary = pd.concat(
        [bench_rows, exposure_summary[exposure_summary["strategy"] != "benchmark"]],
        ignore_index=True,
    )

    logger.info(
        "Backtest complete: %d rows  bench=%.2f%%  def=%.2f%%  "
        "bull_aware=%.2f%%  mom=%.2f%%  vol=%.2f%%",
        len(daily_df),
        bench_m["cagr"] * 100, def_m["cagr"] * 100,
        bull_m["cagr"] * 100,  mom_m["cagr"] * 100, vol_m["cagr"] * 100,
    )

    return {
        "daily":                     daily_df,
        "metrics":                   metrics_df,
        "metrics_comparison":        metrics_comparison_df,
        "stress":                    stress_df,
        "stress_comparison":         stress_comparison,
        "bull_market_participation": bull_participation_df,
        "strategy_exposure_summary": exposure_summary,
        "defensive_label":           defensive_label,
    }
