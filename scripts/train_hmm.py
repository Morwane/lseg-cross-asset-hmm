#!/usr/bin/env python3
"""Train the Gaussian HMM — full-sample or walk-forward mode.

Reads output/data/features_dataset.csv, fits a GaussianHMM using the hmm_core
feature set from config/model.yaml, labels regimes from realised statistics, and
saves results to output/.

Usage:
    python scripts/train_hmm.py                        # full-sample, preferred n
    python scripts/train_hmm.py --n-components 3
    python scripts/train_hmm.py --select-bic
    python scripts/train_hmm.py --mode walk_forward    # expanding-window OOS
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings, load_model_config, PROJECT_ROOT, TRADING_DAYS
from src.hmm_model import (
    load_hmm_features,
    fit_scaler,
    fit_hmm,
    select_best_n_components,
    get_state_statistics,
    save_model,
)
from src.regime_labeling import build_label_map, attach_regime_labels, summarise_regimes
from src.walk_forward import walk_forward_predict
from src.utils import setup_logging, ensure_output_dirs

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the cross-asset Gaussian HMM regime model."
    )
    parser.add_argument(
        "--mode",
        choices=["full_sample", "walk_forward"],
        default="full_sample",
        help="Training mode: full_sample (default) or walk_forward (expanding-window OOS).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of HMM states (full_sample mode only). "
             "Defaults to preferred_n_components in model.yaml.",
    )
    parser.add_argument(
        "--select-bic",
        action="store_true",
        help="Try all n_components_candidates and select lowest BIC (full_sample mode only).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_wf_summary(predictions: pd.DataFrame, out_path: Path) -> None:
    """Print a compact walk-forward results summary to stdout."""
    width = 72
    print("\n" + "=" * width)
    print("  WALK-FORWARD VALIDATION SUMMARY")
    print("=" * width)
    print(f"  OOS rows  : {len(predictions)}")
    print(f"  Date range: {predictions.index[0].date()} → {predictions.index[-1].date()}")
    n_folds = predictions["train_end_date"].nunique()
    print(f"  Folds     : {n_folds} monthly retrains")
    print()
    counts = predictions["regime_label"].value_counts().sort_index()
    print(f"  {'Label':<18}  {'Days':>6}  {'Freq %':>7}")
    print("  " + "-" * 34)
    for lbl, n in counts.items():
        print(f"  {lbl:<18}  {n:>6}  {n / len(predictions) * 100:>6.1f}%")
    print()
    print(f"  Saved → {out_path}")
    print("=" * width + "\n")


def _print_summary(
    regime_df: pd.DataFrame,
    state_stats: pd.DataFrame,
    feature_names: list[str],
    n_components: int,
    cov_type: str,
    bic_scores: dict | None,
    output_paths: dict[str, Path],
) -> None:
    width = 76
    print("\n" + "=" * width)
    print("  HMM TRAINING SUMMARY")
    print("=" * width)
    print(f"  States (n_components) : {n_components}")
    print(f"  Covariance type       : {cov_type}")
    print(f"  Date range            : {regime_df.index[0].date()} → {regime_df.index[-1].date()}")
    print(f"  Observations          : {len(regime_df)}")

    if bic_scores:
        print()
        print("  BIC scores:")
        for n, bic in sorted(bic_scores.items()):
            marker = " ← selected" if n == n_components else ""
            print(f"    n={n}  BIC={bic:.1f}{marker}")

    print()
    print("  REGIME FREQUENCIES")
    print(f"  {'Label':<18}  {'State':>5}  {'Days':>6}  {'Freq %':>7}")
    print("  " + "-" * 44)
    label_to_state = dict(zip(
        regime_df["regime_label"].values, regime_df["regime_state"].values
    ))
    counts = regime_df.groupby("regime_label").size()
    for lbl in sorted(counts.index):
        s = label_to_state.get(lbl, "?")
        n = counts[lbl]
        pct = n / len(regime_df) * 100
        print(f"  {lbl:<18}  {str(s):>5}  {n:>6}  {pct:>6.1f}%")

    print()
    print("  STATE MEAN EQUITY RETURN (annualised)")
    eq_col = next(
        (c for c in state_stats.columns if c.endswith("equity_ret_1d_mean")), None
    )
    if eq_col:
        print(f"  {'State':>5}  {'Ann. Ret %':>12}  {'Regime':>14}")
        print("  " + "-" * 36)
        state_to_label = {
            row["regime_state"]: row["regime_label"]
            for _, row in regime_df.drop_duplicates("regime_state").iterrows()
        }
        for _, row in state_stats.iterrows():
            s = int(row["state"])
            ann_ret = row[eq_col] * TRADING_DAYS * 100
            lbl = state_to_label.get(s, "?")
            print(f"  {s:>5}  {ann_ret:>11.2f}%  {lbl:>14}")

    print()
    print("  OUTPUT FILES")
    for key, path in output_paths.items():
        print(f"  {key:<28} → {path}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    settings = load_settings()
    model_cfg = load_model_config()

    data_dir = PROJECT_ROOT / settings.outputs.data_dir
    audit_dir = PROJECT_ROOT / settings.outputs.audit_dir
    models_dir = PROJECT_ROOT / "output" / "models"
    ensure_output_dirs(data_dir, audit_dir, models_dir)

    features_path = data_dir / "features_dataset.csv"
    if not features_path.exists():
        logger.error(
            "Feature dataset not found at %s.\n"
            "Run 'python scripts/build_features.py' first.",
            features_path,
        )
        sys.exit(1)

    # Load configuration
    core_features: list[str] = model_cfg["features"]["hmm_core"]
    hmm_cfg = model_cfg["hmm"]
    cov_type: str = hmm_cfg["covariance_type"]
    cov_fallback: str = hmm_cfg["covariance_type_fallback"]
    n_iter: int = hmm_cfg["n_iter"]
    random_state: int = hmm_cfg["random_state"]
    candidates: list[int] = hmm_cfg["n_components_candidates"]
    preferred_n: int = hmm_cfg["preferred_n_components"]
    initial_train_years: int = hmm_cfg["initial_train_years"]

    # Load features (drops warm-up NaN rows)
    X_df = load_hmm_features(features_path, core_features)

    try:
        eq_ret_idx = core_features.index("equity_ret_1d")
    except ValueError:
        logger.warning("'equity_ret_1d' not in hmm_core — using index 0 for regime labelling.")
        eq_ret_idx = 0

    # ------------------------------------------------------------------ #
    # Walk-forward mode                                                    #
    # ------------------------------------------------------------------ #
    if args.mode == "walk_forward":
        report_dir = PROJECT_ROOT / settings.outputs.report_dir
        ensure_output_dirs(report_dir)

        initial_train_rows = initial_train_years * TRADING_DAYS
        logger.info(
            "Walk-forward mode  n_components=%d  initial_train_rows=%d (%d years)",
            preferred_n, initial_train_rows, initial_train_years,
        )

        predictions = walk_forward_predict(
            X_df=X_df,
            n_components=preferred_n,
            covariance_type=cov_type,
            covariance_type_fallback=cov_fallback,
            n_iter=n_iter,
            random_state=random_state,
            initial_train_rows=initial_train_rows,
            equity_ret_idx=eq_ret_idx,
        )

        out_path = report_dir / "walk_forward_regime_predictions.csv"
        predictions.to_csv(out_path)
        logger.info("Walk-forward predictions saved → %s  shape=%s", out_path, predictions.shape)

        _print_wf_summary(predictions, out_path)
        return

    # ------------------------------------------------------------------ #
    # Full-sample mode (existing logic)                                    #
    # ------------------------------------------------------------------ #
    X = X_df.values
    scaler = fit_scaler(X)
    X_scaled = scaler.transform(X)

    # Resolve n_components
    if args.select_bic:
        n_components_target = None  # determined by BIC
    elif args.n_components is not None:
        n_components_target = args.n_components
    else:
        n_components_target = preferred_n

    # Fit model
    bic_scores: dict | None = None
    if args.select_bic:
        logger.info("Running BIC model selection over candidates: %s", candidates)
        model, n_components, cov_used, bic_scores = select_best_n_components(
            X_scaled, candidates, cov_type, cov_fallback, n_iter, random_state,
        )
    else:
        logger.info("Fitting HMM with n_components=%d", n_components_target)
        model, cov_used = fit_hmm(
            X_scaled, n_components_target, cov_type, cov_fallback, n_iter, random_state,
        )
        n_components = n_components_target

    # Label regimes
    logger.info("Assigning regime labels from realised equity return statistics.")
    label_map = build_label_map(model, X_scaled, eq_ret_idx)
    regime_df = attach_regime_labels(X_df, model, X_scaled, label_map)

    # State statistics (original units via scaler)
    state_stats = get_state_statistics(model, X_scaled, core_features, scaler=scaler)

    # Merge state → label into state_stats
    state_to_label = {s: lbl for s, lbl in label_map.items()}
    state_stats["regime_label"] = state_stats["state"].map(state_to_label)

    # Equity return series for regime summary
    if "equity_ret_1d" in X_df.columns:
        eq_returns = X_df["equity_ret_1d"]
    else:
        eq_returns = None

    regime_summary = summarise_regimes(regime_df, equity_returns=eq_returns)

    # Save outputs
    model_path = models_dir / "hmm_model.pkl"
    labels_path = data_dir / "regime_labels.csv"
    stats_path = audit_dir / "hmm_state_statistics.csv"
    summary_path = audit_dir / "hmm_regime_summary.csv"

    save_model(model, scaler, model_path)
    regime_df.to_csv(labels_path)
    logger.info("Regime labels saved → %s  shape=%s", labels_path, regime_df.shape)
    state_stats.to_csv(stats_path, index=False)
    logger.info("State statistics saved → %s", stats_path)
    regime_summary.to_csv(summary_path, index=False)
    logger.info("Regime summary saved → %s", summary_path)

    if bic_scores is not None:
        bic_df = pd.DataFrame(
            [{"n_components": n, "bic": bic} for n, bic in sorted(bic_scores.items())]
        )
        bic_path = audit_dir / "hmm_bic_scores.csv"
        bic_df.to_csv(bic_path, index=False)
        logger.info("BIC scores saved → %s", bic_path)
    else:
        bic_path = None

    output_paths = {
        "regime_labels.csv": labels_path,
        "hmm_state_statistics.csv": stats_path,
        "hmm_regime_summary.csv": summary_path,
        "hmm_model.pkl": model_path,
    }
    if bic_path:
        output_paths["hmm_bic_scores.csv"] = bic_path

    _print_summary(regime_df, state_stats, core_features, n_components, cov_used, bic_scores, output_paths)


if __name__ == "__main__":
    main()
