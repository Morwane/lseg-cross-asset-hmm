"""Tests for regime labelling logic — Phase 5 implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime_labeling import build_label_map, attach_regime_labels, summarise_regimes


class _MockHMM:
    """Minimal HMM mock — predict() returns a fixed state array."""

    def __init__(self, n_components: int, states: np.ndarray):
        self.n_components = n_components
        self._states = states

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._states


def _make_X_scaled(n: int, n_features: int, seed: int = 0) -> np.ndarray:
    """Return a small (n, n_features) float array of standardised-ish values."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n_features))


def test_labels_not_hardcoded_by_state_number():
    """State 0 is not assumed to be risk_on; labels derive from equity return rank.

    Setup: state 2 has the highest equity return → state 2 must be labelled 'risk_on'.
            state 0 has the lowest equity return  → state 0 must be labelled 'stress'.
    """
    n, n_features, eq_idx = 90, 3, 0  # feature 0 is equity_ret_1d

    # Allocate 30 rows to each of 3 states
    states = np.array([0] * 30 + [1] * 30 + [2] * 30)

    # Build X_scaled with controlled equity returns per state
    X = np.zeros((n, n_features))
    X[:30, eq_idx] = -0.01   # state 0 → low equity return
    X[30:60, eq_idx] = 0.00  # state 1 → mid
    X[60:, eq_idx] = +0.01   # state 2 → high equity return

    model = _MockHMM(n_components=3, states=states)
    label_map = build_label_map(model, X, equity_ret_idx=eq_idx)

    assert label_map[2] == "risk_on", f"Expected state 2 → 'risk_on', got {label_map}"
    assert label_map[0] == "stress", f"Expected state 0 → 'stress', got {label_map}"
    assert label_map[1] == "transition", f"Expected state 1 → 'transition', got {label_map}"


def test_labels_assigned_from_summary_statistics():
    """Regime labels are derived from per-state equity return means, not state index.

    With n_components=2, the higher-return state is 'risk_on' and the lower is 'stress'
    regardless of which integer index they occupy.
    """
    n, n_features, eq_idx = 60, 4, 0

    # state 1 has higher returns than state 0
    states = np.array([0] * 30 + [1] * 30)
    X = np.zeros((n, n_features))
    X[:30, eq_idx] = -0.005   # state 0 → worse
    X[30:, eq_idx] = +0.005   # state 1 → better

    model = _MockHMM(n_components=2, states=states)
    label_map = build_label_map(model, X, equity_ret_idx=eq_idx)

    assert label_map[1] == "risk_on"
    assert label_map[0] == "stress"
    # No 'transition' should exist for n=2
    assert "transition" not in label_map.values()


def test_ambiguous_regimes_handled_safely():
    """If all states have identical equity return means, fall back to generic labels.

    Generic labels take the form 'state_N' so downstream code does not crash.
    """
    n, n_features, eq_idx = 60, 3, 0

    states = np.array([0] * 20 + [1] * 20 + [2] * 20)
    X = np.zeros((n, n_features))
    # All states get the exact same equity return → spread = 0
    X[:, eq_idx] = 0.001

    model = _MockHMM(n_components=3, states=states)
    label_map = build_label_map(model, X, equity_ret_idx=eq_idx)

    # Must have an entry for every state
    assert set(label_map.keys()) == {0, 1, 2}
    # All labels should be generic (no semantic names)
    for lbl in label_map.values():
        assert lbl.startswith("state_"), (
            f"Expected generic 'state_N' label, got '{lbl}'"
        )
    # Semantic names must be absent
    assert "risk_on" not in label_map.values()
    assert "stress" not in label_map.values()
