"""Tests for walk-forward validation — Phase 6."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.walk_forward import walk_forward_predict

# Use a tiny HMM (n_iter=5) so tests run in under a second each.
_WF_KWARGS = dict(
    n_components=2,
    covariance_type="diag",
    covariance_type_fallback="diag",
    n_iter=5,
    random_state=0,
    equity_ret_idx=0,
)


def _make_feature_df(n_rows: int = 200, n_features: int = 4, seed: int = 7) -> pd.DataFrame:
    """Synthetic daily feature DataFrame with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_rows, freq="B")
    data = rng.standard_normal((n_rows, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_train_end_is_strictly_before_prediction_date():
    """For every OOS row, train_end_date must be before the prediction date.

    This is the core no-leakage invariant: the model fitted on data up to T-1
    cannot have seen anything from T onward.
    """
    pytest.importorskip("hmmlearn")
    X_df = _make_feature_df(200)
    initial_train_rows = 80

    preds = walk_forward_predict(X_df, initial_train_rows=initial_train_rows, **_WF_KWARGS)

    train_ends = pd.to_datetime(preds["train_end_date"])
    pred_dates = preds.index

    violations = (train_ends.values >= pred_dates.values).sum()
    assert violations == 0, (
        f"{violations} rows where train_end_date >= prediction date (look-ahead detected)."
    )


def test_output_contains_required_columns():
    """Walk-forward output must have the fixed schema expected by downstream code."""
    pytest.importorskip("hmmlearn")
    X_df = _make_feature_df(160)
    preds = walk_forward_predict(X_df, initial_train_rows=60, **_WF_KWARGS)

    required = {
        "regime_state", "regime_label",
        "prob_risk_on", "prob_transition", "prob_stress",
        "train_end_date", "n_train_rows",
    }
    assert required.issubset(set(preds.columns)), (
        f"Missing columns: {required - set(preds.columns)}"
    )
    assert preds.index.name == "date"


def test_regime_labels_are_valid_strings():
    """Every regime_label must be a recognised string; no NaN or integer labels."""
    pytest.importorskip("hmmlearn")
    X_df = _make_feature_df(160)
    preds = walk_forward_predict(X_df, initial_train_rows=60, **_WF_KWARGS)

    # Labels must be strings
    assert preds["regime_label"].dtype == object
    # Must be non-null
    assert preds["regime_label"].notna().all()
    # Every label must be a non-empty string
    assert (preds["regime_label"].str.len() > 0).all()


def test_oos_rows_cover_post_training_period_only():
    """No OOS prediction date should fall within the initial training window."""
    pytest.importorskip("hmmlearn")
    X_df = _make_feature_df(200)
    initial_train_rows = 80

    preds = walk_forward_predict(X_df, initial_train_rows=initial_train_rows, **_WF_KWARGS)

    # The training window covers at least initial_train_rows rows from the start.
    # All prediction dates must be strictly after the first possible training cutoff.
    # Minimum train_end_date is the date of the initial_train_rows-th observation.
    min_train_end = pd.Timestamp(preds["train_end_date"].min())
    assert (preds.index > min_train_end).all(), (
        "Some OOS rows have dates inside the initial training window."
    )


def test_probabilities_sum_to_one():
    """prob_risk_on + prob_transition + prob_stress must equal 1.0 for every row."""
    pytest.importorskip("hmmlearn")
    X_df = _make_feature_df(160)
    preds = walk_forward_predict(X_df, initial_train_rows=60, **_WF_KWARGS)

    total = (
        preds["prob_risk_on"] + preds["prob_transition"] + preds["prob_stress"]
    )
    assert (total - 1.0).abs().max() < 1e-6, (
        f"Probabilities do not sum to 1 (max deviation: {(total - 1.0).abs().max():.2e})"
    )
