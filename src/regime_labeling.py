"""Regime labelling from realised state statistics."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum spread in equity return means (scaled units) to assign semantic labels.
# Below this threshold labels fall back to generic "state_N" names.
_MIN_EQUITY_RETURN_SPREAD: float = 1e-3


def build_label_map(
    model,
    X_scaled: np.ndarray,
    equity_ret_idx: int,
    min_spread: float = _MIN_EQUITY_RETURN_SPREAD,
) -> dict[int, str]:
    """Map HMM state integers to semantic regime labels.

    States are ranked by their realised mean equity return (in scaled units —
    monotone transformation preserves rank). Never assumes state 0 = risk_on.

    Rank rules:
      - Highest equity return → "risk_on"
      - Lowest equity return  → "stress"
      - Single middle state   → "transition"
      - Multiple middle states → "transition_1", "transition_2", …

    Falls back to "state_N" if the return spread is below min_spread (ambiguous).
    """
    states = model.predict(X_scaled)
    n_components = model.n_components

    state_eq_means: dict[int, float] = {}
    for s in range(n_components):
        mask = states == s
        vals = X_scaled[mask, equity_ret_idx] if mask.sum() > 0 else np.array([0.0])
        state_eq_means[s] = float(vals.mean())

    spread = max(state_eq_means.values()) - min(state_eq_means.values())
    if spread < min_spread:
        logger.warning(
            "Equity return spread across states %.2e < threshold %.2e — "
            "falling back to generic state labels.",
            spread, min_spread,
        )
        return {s: f"state_{s}" for s in range(n_components)}

    ranked = sorted(state_eq_means, key=state_eq_means.__getitem__, reverse=True)

    label_map: dict[int, str] = {}
    label_map[ranked[0]] = "risk_on"
    label_map[ranked[-1]] = "stress"

    middle = ranked[1:-1]
    if len(middle) == 1:
        label_map[middle[0]] = "transition"
    else:
        for i, s in enumerate(middle):
            label_map[s] = f"transition_{i + 1}"

    for s, lbl in sorted(label_map.items()):
        logger.info(
            "  State %d → %-14s  mean_equity_ret_scaled=%.4f",
            s, lbl, state_eq_means[s],
        )

    return label_map


def attach_regime_labels(
    X_df: pd.DataFrame,
    model,
    X_scaled: np.ndarray,
    label_map: dict[int, str],
) -> pd.DataFrame:
    """Return a DataFrame with regime_state (int) and regime_label (str).

    Shares the DatetimeIndex of X_df.
    """
    states = model.predict(X_scaled)
    return pd.DataFrame(
        {
            "regime_state": states,
            "regime_label": [label_map[s] for s in states],
        },
        index=X_df.index,
    )


def summarise_regimes(
    regime_df: pd.DataFrame,
    equity_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Per-regime frequency and (optionally) annualised equity return statistics."""
    rows = []
    for label, grp in regime_df.groupby("regime_label"):
        row: dict = {
            "regime_label": label,
            "n_days": len(grp),
            "freq_pct": round(len(grp) / len(regime_df) * 100, 2),
        }
        if equity_returns is not None:
            aligned = equity_returns.reindex(grp.index).dropna()
            if not aligned.empty:
                row["equity_ret_ann_pct"] = round(float(aligned.mean()) * 252 * 100, 2)
                row["equity_vol_ann_pct"] = round(float(aligned.std()) * np.sqrt(252) * 100, 2)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("regime_label").reset_index(drop=True)
