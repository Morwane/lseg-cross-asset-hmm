"""Expanding-window walk-forward validation for the Gaussian HMM regime model.

Training rule: for each calendar month M after the initial training period,
fit the scaler and HMM on all data up to and including the last day of month M-1,
then predict states and probabilities for all days in month M.

No data from month M (or later) is ever used when predicting month M.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.hmm_model import fit_scaler, fit_hmm
from src.regime_labeling import build_label_map

logger = logging.getLogger(__name__)

# Canonical semantic labels used for probability output columns.
_SEMANTIC_LABELS = ("risk_on", "transition", "stress")


def _month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return sorted month-end timestamps that bound every row in index."""
    periods = index.to_period("M").unique()
    return sorted(periods.to_timestamp("M"))


def walk_forward_predict(
    X_df: pd.DataFrame,
    n_components: int,
    covariance_type: str,
    covariance_type_fallback: str,
    n_iter: int,
    random_state: int,
    initial_train_rows: int,
    equity_ret_idx: int = 0,
) -> pd.DataFrame:
    """Expanding-window walk-forward prediction; returns only out-of-sample rows.

    For each month after the initial training period:
      1. Scaler and HMM are fit on all data up to the prior month-end.
      2. States and posteriors are predicted for the current month.
      3. Label map is derived from the training data (never the prediction data).

    Returns a DataFrame indexed by date with columns:
      regime_state   — integer state (argmax posterior)
      regime_label   — semantic label (risk_on / transition / stress)
      prob_risk_on   — posterior probability for the risk_on regime
      prob_transition— posterior probability for the transition regime
      prob_stress    — posterior probability for the stress regime
      train_end_date — last training date used for this fold
      n_train_rows   — number of rows in the training window
    """
    month_ends = _month_end_dates(X_df.index)

    # Locate the first prediction month: earliest month where the preceding
    # data provides at least initial_train_rows observations.
    first_pred_idx: int | None = None
    for i, me in enumerate(month_ends):
        if int((X_df.index <= me).sum()) >= initial_train_rows:
            first_pred_idx = i + 1
            break

    if first_pred_idx is None or first_pred_idx >= len(month_ends):
        raise ValueError(
            f"Insufficient data for an initial training window of {initial_train_rows} rows "
            f"(dataset has {len(X_df)} rows across {len(month_ends)} months)."
        )

    n_folds = len(month_ends) - first_pred_idx
    logger.info(
        "Walk-forward: %d prediction months  initial_train_rows=%d  "
        "first_pred_month=%s  n_components=%d",
        n_folds, initial_train_rows,
        month_ends[first_pred_idx].strftime("%Y-%m"), n_components,
    )

    fold_frames: list[pd.DataFrame] = []

    for fold in range(n_folds):
        pred_month_end = month_ends[first_pred_idx + fold]
        train_cutoff = month_ends[first_pred_idx + fold - 1]

        train_mask = X_df.index <= train_cutoff
        pred_mask = (X_df.index > train_cutoff) & (X_df.index <= pred_month_end)

        X_train = X_df.loc[train_mask].values
        X_pred_df = X_df.loc[pred_mask]

        if len(X_train) == 0 or len(X_pred_df) == 0:
            continue

        try:
            scaler = fit_scaler(X_train)
            X_train_sc = scaler.transform(X_train)
            X_pred_sc = scaler.transform(X_pred_df.values)

            model, _ = fit_hmm(
                X_train_sc, n_components,
                covariance_type, covariance_type_fallback,
                n_iter, random_state,
            )

            label_map = build_label_map(model, X_train_sc, equity_ret_idx)
            states = model.predict(X_pred_sc)
            probas = model.predict_proba(X_pred_sc)  # (n_pred, n_states)

            # Aggregate state probabilities into fixed semantic columns.
            # "transition_1", "transition_2" etc. all flow into prob_transition.
            prob_cols: dict[str, np.ndarray] = {
                f"prob_{lbl}": np.zeros(len(X_pred_df)) for lbl in _SEMANTIC_LABELS
            }
            for state_idx, lbl in label_map.items():
                target = "transition" if lbl.startswith("transition") else lbl
                col = f"prob_{target}"
                if col in prob_cols:
                    prob_cols[col] = prob_cols[col] + probas[:, state_idx]

            fold_df = pd.DataFrame(
                {
                    "regime_state": states,
                    "regime_label": [label_map[s] for s in states],
                    **prob_cols,
                    "train_end_date": train_cutoff.date().isoformat(),
                    "n_train_rows": len(X_train),
                },
                index=X_pred_df.index,
            )
            fold_frames.append(fold_df)

            if fold == 0 or (fold + 1) % 12 == 0 or fold == n_folds - 1:
                logger.info(
                    "  Fold %3d/%d  train_end=%s  n_pred=%d",
                    fold + 1, n_folds,
                    train_cutoff.strftime("%Y-%m-%d"), len(X_pred_df),
                )

        except Exception as exc:
            logger.warning("  Fold %d (%s) failed — skipping: %s", fold + 1,
                           pred_month_end.strftime("%Y-%m"), exc)
            continue

    if not fold_frames:
        raise RuntimeError("Walk-forward produced no predictions — all folds failed.")

    result = pd.concat(fold_frames).sort_index()
    result.index.name = "date"

    n_oos = len(result)
    logger.info(
        "Walk-forward complete: %d OOS rows (%.1f%% of %d feature rows)",
        n_oos, n_oos / len(X_df) * 100, len(X_df),
    )
    return result
