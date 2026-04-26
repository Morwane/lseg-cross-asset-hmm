"""Professional risk controls for the regime overlay strategies.

Controls applied in sequence by build_risk_controlled_returns():
  1. Vol targeting   — scale equity weight to hit target annualised portfolio vol.
  2. Equity-curve stop-loss — cap weight at defensive_floor while drawdown > threshold.
  3. Turnover cap    — limit max daily weight change to prevent whipsaw.

Separately:
  compute_cvar_by_regime() — Expected Shortfall per strategy × regime × confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import TRADING_DAYS

# ── Column map: strategy → (equity_weight_col, return_col, tc_col) ────────────
_STRAT_COLS: dict[str, tuple[str, str, str]] = {
    "defensive_overlay":  (
        "equity_weight",                "overlay_ret",         "transaction_cost"),
    "bull_aware_overlay": (
        "bull_aware_equity_weight",     "bull_aware_ret",      "bull_aware_transaction_cost"),
    "regime_momentum":    (
        "regime_momentum_equity_weight","regime_momentum_ret", "regime_momentum_transaction_cost"),
    "vol_managed":        (
        "vol_managed_equity_weight",    "vol_managed_ret",     "vol_managed_transaction_cost"),
}

# Short output-column prefix per strategy
_RC_PREFIX: dict[str, str] = {
    "defensive_overlay":  "rc_def",
    "bull_aware_overlay": "rc_bull",
    "regime_momentum":    "rc_mom",
    "vol_managed":        "rc_vol",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _back_compute_def_ret(
    eq_w: pd.Series,
    ret: pd.Series,
    tc: pd.Series,
    bench_ret: pd.Series,
) -> pd.Series:
    """Recover the defensive-asset (bond) daily return from portfolio composition."""
    gross = ret + tc                    # net return → gross before costs
    denom = 1.0 - eq_w
    def_r = pd.Series(np.nan, index=eq_w.index)
    mask  = denom.abs() > 1e-6
    def_r[mask] = (gross[mask] - eq_w[mask] * bench_ret[mask]) / denom[mask]
    return def_r.ffill().bfill()


# ─────────────────────────────────────────────────────────────────────────────
# Control 1 — Volatility targeting
# ─────────────────────────────────────────────────────────────────────────────

def apply_vol_target(
    equity_weights: pd.Series,
    strategy_returns: pd.Series,
    target_vol: float = 0.08,
    lookback: int = 20,
    min_scale: float = 0.50,
    max_scale: float = 1.00,
) -> pd.Series:
    """Scale equity weights so the portfolio targets *target_vol* annualised vol.

    Uses a trailing lookback-day realised vol (lagged 1 day to avoid look-ahead).
    Where vol history is insufficient the scale defaults to 1.0 (no change).
    """
    target_daily = target_vol / np.sqrt(TRADING_DAYS)
    realized_vol = (
        strategy_returns
        .rolling(lookback, min_periods=max(5, lookback // 2))
        .std()
        .shift(1)
    )
    scale = (target_daily / realized_vol).clip(min_scale, max_scale)
    scale = scale.fillna(1.0)
    return (equity_weights * scale).clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Control 2 — Equity-curve stop-loss
# ─────────────────────────────────────────────────────────────────────────────

def apply_equity_curve_stop(
    equity_weights: pd.Series,
    strategy_returns: pd.Series,
    drawdown_threshold: float = -0.10,
    recovery_threshold: float = -0.05,
    defensive_floor: float = 0.15,
) -> tuple[pd.Series, pd.Series]:
    """Cap equity weight at *defensive_floor* while strategy drawdown > threshold.

    State machine:
      ACTIVE  → STOPPED when drawdown < drawdown_threshold
      STOPPED → ACTIVE  when drawdown > recovery_threshold

    Returns (adjusted_weights, stop_active) where stop_active is a boolean
    Series that is True on every day the stop is engaged.
    """
    equity_curve = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    running_max  = equity_curve.cummax()
    drawdown     = equity_curve / running_max - 1.0

    weights     = equity_weights.copy()
    stop_active = pd.Series(False, index=equity_weights.index)

    stopped = False
    for i in range(len(weights)):
        dd = drawdown.iloc[i]
        if not stopped and dd < drawdown_threshold:
            stopped = True
        elif stopped and dd > recovery_threshold:
            stopped = False
        if stopped:
            weights.iloc[i]     = min(weights.iloc[i], defensive_floor)
            stop_active.iloc[i] = True

    return weights, stop_active


# ─────────────────────────────────────────────────────────────────────────────
# Control 3 — Turnover cap
# ─────────────────────────────────────────────────────────────────────────────

def apply_turnover_cap(
    equity_weights: pd.Series,
    max_daily_change: float = 0.10,
) -> pd.Series:
    """Prevent daily equity-weight changes from exceeding *max_daily_change*."""
    weights = equity_weights.copy()
    for i in range(1, len(weights)):
        delta = weights.iloc[i] - weights.iloc[i - 1]
        if abs(delta) > max_daily_change:
            weights.iloc[i] = weights.iloc[i - 1] + np.sign(delta) * max_daily_change
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# CVaR by regime
# ─────────────────────────────────────────────────────────────────────────────

def compute_cvar_by_regime(
    daily_df: pd.DataFrame,
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Conditional Value-at-Risk (Expected Shortfall) per strategy × regime × confidence.

    Returns a DataFrame with columns:
      strategy, regime, confidence, var, cvar, n_obs
    where var and cvar are expressed as daily returns (negative = loss).
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    regime_col = "regime_label" if "regime_label" in daily_df.columns else None
    rows: list[dict] = []

    for strat, (eq_col, ret_col, _tc) in _STRAT_COLS.items():
        if ret_col not in daily_df.columns:
            continue
        rets = daily_df[ret_col].dropna()

        for cl in confidence_levels:
            if regime_col:
                for regime in sorted(daily_df[regime_col].dropna().unique()):
                    mask        = daily_df[regime_col] == regime
                    regime_rets = rets[mask & rets.notna()]
                    if len(regime_rets) < 10:
                        continue
                    q   = np.percentile(regime_rets, (1.0 - cl) * 100.0)
                    tail = regime_rets[regime_rets <= q]
                    rows.append({
                        "strategy":   strat,
                        "regime":     regime,
                        "confidence": cl,
                        "var":        float(q),
                        "cvar":       float(tail.mean()) if len(tail) else float(q),
                        "n_obs":      int(len(regime_rets)),
                    })
            else:
                q    = np.percentile(rets, (1.0 - cl) * 100.0)
                tail = rets[rets <= q]
                rows.append({
                    "strategy":   strat,
                    "regime":     "all",
                    "confidence": cl,
                    "var":        float(q),
                    "cvar":       float(tail.mean()) if len(tail) else float(q),
                    "n_obs":      int(len(rets)),
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Combined builder
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_controlled_returns(
    daily_df: pd.DataFrame,
    target_vol: float = 0.08,
    lookback: int = 20,
    drawdown_threshold: float = -0.10,
    recovery_threshold: float = -0.05,
    defensive_floor: float = 0.15,
    max_daily_change: float = 0.10,
    tc_per_unit: float = 0.001,
) -> pd.DataFrame:
    """Apply all risk controls and return daily_df with additional rc_ columns.

    New columns per strategy (prefix = rc_def / rc_bull / rc_mom / rc_vol):
      {prefix}_equity_weight  — final equity weight after all controls
      {prefix}_ret            — net return (gross - estimated transaction costs)
      {prefix}_stop_active    — bool, True when stop-loss is engaged

    Controls applied in order: vol targeting → equity-curve stop → turnover cap.
    Transaction costs are estimated as tc_per_unit × absolute daily weight change.
    """
    bench_ret = daily_df["benchmark_ret"]
    result    = daily_df.copy()

    for strat, (eq_col, ret_col, tc_col) in _STRAT_COLS.items():
        if ret_col not in daily_df.columns or eq_col not in daily_df.columns:
            continue

        eq_w  = daily_df[eq_col]
        ret   = daily_df[ret_col]
        tc    = daily_df[tc_col]
        def_r = _back_compute_def_ret(eq_w, ret, tc, bench_ret)

        # 1. Vol targeting on original returns
        vt_w = apply_vol_target(eq_w, ret, target_vol, lookback)

        # 2. Equity-curve stop on vol-targeted portfolio
        vt_ret      = vt_w * bench_ret + (1.0 - vt_w) * def_r
        stop_w, stop_active = apply_equity_curve_stop(
            vt_w, vt_ret, drawdown_threshold, recovery_threshold, defensive_floor
        )

        # 3. Turnover cap
        final_w = apply_turnover_cap(stop_w, max_daily_change)

        # Recompute gross return and approximate transaction costs
        gross_ret = final_w * bench_ret + (1.0 - final_w) * def_r
        turnover  = final_w.diff().abs().fillna(0.0)
        net_ret   = gross_ret - turnover * tc_per_unit

        pfx = _RC_PREFIX[strat]
        result[f"{pfx}_equity_weight"] = final_w.values
        result[f"{pfx}_ret"]           = net_ret.values
        result[f"{pfx}_stop_active"]   = stop_active.values

    return result
