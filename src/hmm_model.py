"""Gaussian HMM fitting, prediction, and model selection."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_hmm_features(
    features_path: Path,
    core_features: list[str],
) -> pd.DataFrame:
    """Load features CSV, select hmm_core columns, drop warm-up NaN rows."""
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df.index.name = "date"

    missing = [c for c in core_features if c not in df.columns]
    if missing:
        raise ValueError(f"hmm_core features missing from feature dataset: {missing}")

    X_df = df[core_features].copy()
    n_before = len(X_df)
    X_df = X_df.dropna()
    n_dropped = n_before - len(X_df)
    if n_dropped > 0:
        logger.info("Dropped %d warm-up rows (NaN in hmm_core features).", n_dropped)

    logger.info(
        "HMM feature matrix: %d rows × %d features  [%s → %s]",
        len(X_df), len(core_features),
        X_df.index[0].date(), X_df.index[-1].date(),
    )
    return X_df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on the feature matrix."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def fit_hmm(
    X_scaled: np.ndarray,
    n_components: int,
    covariance_type: str = "full",
    covariance_type_fallback: str = "diag",
    n_iter: int = 1000,
    random_state: int = 42,
) -> tuple:
    """Fit a GaussianHMM; falls back to covariance_type_fallback if primary fails.

    Returns (fitted_model, covariance_type_used).
    """
    from hmmlearn.hmm import GaussianHMM

    for cov_type in [covariance_type, covariance_type_fallback]:
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=cov_type,
                n_iter=n_iter,
                random_state=random_state,
                verbose=False,
            )
            model.fit(X_scaled)
            if not model.monitor_.converged:
                logger.warning(
                    "HMM (n=%d, cov=%s) did not converge in %d iterations.",
                    n_components, cov_type, n_iter,
                )
            logger.info(
                "Fitted HMM  n_components=%d  covariance_type=%s  converged=%s",
                n_components, cov_type, model.monitor_.converged,
            )
            return model, cov_type
        except Exception as exc:
            if cov_type == covariance_type_fallback:
                raise RuntimeError(
                    f"HMM fitting failed with both '{covariance_type}' and "
                    f"'{covariance_type_fallback}': {exc}"
                ) from exc
            logger.warning(
                "HMM fitting failed with covariance_type='%s' (n=%d): %s — trying '%s'.",
                cov_type, n_components, exc, covariance_type_fallback,
            )

    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def compute_bic(model, X_scaled: np.ndarray) -> float:
    """BIC = -2 * log_likelihood + n_params * log(n_samples)."""
    n_samples, n_features = X_scaled.shape
    # model.score() returns mean log-likelihood per sample
    log_likelihood = model.score(X_scaled) * n_samples

    n_components = model.n_components
    cov_type = model.covariance_type

    if cov_type == "full":
        cov_params = n_components * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        cov_params = n_components * n_features
    elif cov_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif cov_type == "spherical":
        cov_params = n_components
    else:
        cov_params = n_components * n_features

    n_params = (
        n_components * n_features       # means
        + cov_params                    # covariances
        + n_components * (n_components - 1)  # transition matrix
        + (n_components - 1)            # initial distribution
    )
    return float(-2 * log_likelihood + n_params * np.log(n_samples))


def select_best_n_components(
    X_scaled: np.ndarray,
    candidates: list[int],
    covariance_type: str = "full",
    covariance_type_fallback: str = "diag",
    n_iter: int = 1000,
    random_state: int = 42,
) -> tuple:
    """Fit each candidate n_components; return the model with the lowest BIC.

    Returns (best_model, best_n_components, covariance_type_used, bic_scores).
    """
    bic_scores: dict[int, float] = {}
    models: dict[int, tuple] = {}

    logger.info("BIC model selection over candidates: %s", candidates)
    for n in candidates:
        try:
            model, cov_used = fit_hmm(
                X_scaled, n, covariance_type, covariance_type_fallback, n_iter, random_state,
            )
            bic = compute_bic(model, X_scaled)
            bic_scores[n] = bic
            models[n] = (model, cov_used)
            logger.info("  n_components=%d  BIC=%.2f  cov=%s", n, bic, cov_used)
        except Exception as exc:
            logger.warning("  n_components=%d  FAILED: %s", n, exc)

    if not bic_scores:
        raise RuntimeError("All n_components candidates failed to fit.")

    best_n = min(bic_scores, key=bic_scores.__getitem__)
    best_model, best_cov = models[best_n]
    logger.info(
        "Best n_components=%d  BIC=%.2f  (candidates tried: %s)",
        best_n, bic_scores[best_n], list(bic_scores),
    )
    return best_model, best_n, best_cov, bic_scores


# ---------------------------------------------------------------------------
# State statistics
# ---------------------------------------------------------------------------


def get_state_statistics(
    model,
    X_scaled: np.ndarray,
    feature_names: list[str],
    scaler: Optional[StandardScaler] = None,
) -> pd.DataFrame:
    """Per-state realised mean, std, and fractional weight.

    If scaler is provided, means and stds are back-transformed to original units.
    """
    states = model.predict(X_scaled)
    rows = []
    for s in range(model.n_components):
        mask = states == s
        row: dict = {"state": s, "n_days": int(mask.sum()), "weight": round(mask.mean(), 4)}
        for i, fname in enumerate(feature_names):
            vals_scaled = X_scaled[mask, i]
            if scaler is not None:
                vals = vals_scaled * scaler.scale_[i] + scaler.mean_[i]
            else:
                vals = vals_scaled
            row[f"{fname}_mean"] = round(float(vals.mean()), 6) if len(vals) else None
            row[f"{fname}_std"] = round(float(vals.std()), 6) if len(vals) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model, scaler: StandardScaler, path: Path) -> None:
    """Save the fitted model and scaler together to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    logger.info("Model bundle saved → %s", path)


def load_model(path: Path) -> tuple:
    """Load a saved (model, scaler) pair from disk."""
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"]
