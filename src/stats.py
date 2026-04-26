"""Statistical robustness measures for backtest evaluation.

Two checks:
  1. Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014).
     Adjusts the observed Sharpe for non-normality of returns and multiple
     testing across strategies. Returns the probability (0–1) that the true
     Sharpe exceeds the expected maximum from N independent trials.

  2. Bootstrap confidence intervals — IID percentile bootstrap (10 000 resamples).
     Reports 5th / 25th / 50th / 75th / 95th percentiles for Sharpe, CAGR,
     and max drawdown.

Note: the IID bootstrap does not preserve autocorrelation or path-dependency in
drawdown paths. Results should be interpreted as indicative rather than exact CIs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import TRADING_DAYS

_STRATEGIES: dict[str, str] = {
    "benchmark":          "benchmark_ret",
    "defensive_overlay":  "overlay_ret",
    "bull_aware_overlay": "bull_aware_ret",
    "regime_momentum":    "regime_momentum_ret",
    "vol_managed":        "vol_managed_ret",
}

_EULER_GAMMA = 0.5772156649015329
_CI_LEVELS   = (5, 25, 50, 75, 95)


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────────────

def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    sr_benchmark: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Accounts for non-normality (skewness, excess kurtosis) and multiple testing.
    When n_trials > 1 the benchmark SR* is replaced with the expected maximum
    Sharpe from n_trials independent IID strategies under the null.

    Returns a dict with:
      dsr              — probability in [0, 1]
      sr_annualized    — annualised Sharpe from the input series
      sr_daily         — per-period (daily) Sharpe
      sr_star_daily    — benchmark threshold (per-period) used in the test
      sr_std           — standard deviation of the SR estimator
      skewness         — sample skewness
      excess_kurtosis  — sample excess kurtosis (0 for normal)
      n_obs            — number of observations used
      n_trials         — number of strategies tested
    """
    clean = returns.dropna()
    T = len(clean)

    if T < 30:
        nan = float("nan")
        return {"dsr": nan, "sr_annualized": nan, "sr_daily": nan,
                "sr_star_daily": nan, "sr_std": nan,
                "skewness": nan, "excess_kurtosis": nan,
                "n_obs": T, "n_trials": n_trials}

    mean_r = float(clean.mean())
    std_r  = float(clean.std())

    if std_r < 1e-12:
        nan = float("nan")
        return {"dsr": nan, "sr_annualized": nan, "sr_daily": nan,
                "sr_star_daily": nan, "sr_std": nan,
                "skewness": float(clean.skew()), "excess_kurtosis": float(clean.kurtosis()),
                "n_obs": T, "n_trials": n_trials}

    sr_n       = mean_r / std_r                         # daily Sharpe
    skewness   = float(clean.skew())
    excess_kurt = float(clean.kurtosis())               # pandas: excess kurtosis
    kurtosis   = excess_kurt + 3.0                      # actual (4th moment)

    # Asymptotic variance of the SR estimator (Bailey & LdP 2014, eq. 7)
    sr_var = (1.0 - skewness * sr_n + (kurtosis - 1.0) / 4.0 * sr_n ** 2) / (T - 1)
    sr_var = max(sr_var, 1e-15)
    sr_std = float(np.sqrt(sr_var))

    # SR* — expected maximum from n_trials independent strategies
    if n_trials > 1:
        z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
        z_max  = (1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2
        sr_star = sr_std * z_max          # scale to per-period SR units
    else:
        sr_star = float(sr_benchmark)

    dsr = float(stats.norm.cdf((sr_n - sr_star) / sr_std))

    return {
        "dsr":             dsr,
        "sr_annualized":   sr_n * float(np.sqrt(trading_days)),
        "sr_daily":        sr_n,
        "sr_star_daily":   sr_star,
        "sr_std":          sr_std,
        "skewness":        skewness,
        "excess_kurtosis": excess_kurt,
        "n_obs":           T,
        "n_trials":        n_trials,
    }


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_confidence_intervals(
    returns: pd.Series,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """IID percentile bootstrap CIs for Sharpe, CAGR, and max drawdown.

    Returns a flat dict with keys like sharpe_p5, cagr_p50, mdd_p95, etc.
    Percentiles computed: 5, 25, 50, 75, 95.
    """
    clean = returns.dropna().values
    T = len(clean)
    if T < 10:
        return {f"{m}_p{p}": float("nan")
                for m in ("sharpe", "cagr", "mdd") for p in _CI_LEVELS}

    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_bootstrap)
    cagrs   = np.empty(n_bootstrap)
    mdds    = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(clean, size=T, replace=True)
        vol = sample.std()
        sharpes[i] = (sample.mean() / vol * np.sqrt(trading_days)
                      if vol > 1e-12 else np.nan)
        cagrs[i]   = float((1.0 + sample).prod() ** (trading_days / T) - 1.0)
        cum        = np.cumprod(1.0 + sample)
        running_max = np.maximum.accumulate(cum)
        mdds[i]    = float(np.min(cum / running_max - 1.0))

    result: dict = {}
    for pct in _CI_LEVELS:
        result[f"sharpe_p{pct}"] = float(np.nanpercentile(sharpes, pct))
        result[f"cagr_p{pct}"]   = float(np.nanpercentile(cagrs,   pct))
        result[f"mdd_p{pct}"]    = float(np.nanpercentile(mdds,    pct))
    return result


# ── Strategy-level aggregation ────────────────────────────────────────────────

def compute_all_stats(
    daily_df: pd.DataFrame,
    n_trials: int = 5,
    n_bootstrap: int = 10_000,
    trading_days: int = TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute DSR and bootstrap CIs for all five strategies.

    Returns (dsr_df, bootstrap_df).
    n_trials should equal the number of strategies being compared (default 5).
    """
    dsr_rows  = []
    boot_rows = []

    for strat, col in _STRATEGIES.items():
        if col not in daily_df.columns:
            continue
        rets = daily_df[col].dropna()
        dsr_rows.append({"strategy": strat,
                         **deflated_sharpe_ratio(rets, n_trials=n_trials,
                                                 trading_days=trading_days)})
        boot_rows.append({"strategy": strat,
                          **bootstrap_confidence_intervals(rets, n_bootstrap=n_bootstrap,
                                                           trading_days=trading_days)})

    return pd.DataFrame(dsr_rows), pd.DataFrame(boot_rows)
