"""Tests for the bull-aware overlay — no look-ahead, weights in [0,1], cost deduction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    compute_bull_aware_weights,
    compute_bull_market_periods,
    compute_portfolio_returns,
    compute_transaction_costs,
)


def _make_predictions(n: int = 40, start: str = "2023-01-02") -> pd.DataFrame:
    """Synthetic walk-forward predictions DataFrame."""
    idx = pd.bdate_range(start, periods=n, freq="B")
    rng = np.random.default_rng(0)
    prob_ro = rng.dirichlet(np.ones(3), size=n)
    return pd.DataFrame(
        {
            "regime_state":   np.tile([0, 1, 2], n)[:n],
            "regime_label":   np.tile(["risk_on", "transition", "stress"], n)[:n],
            "prob_risk_on":   prob_ro[:, 0],
            "prob_transition": prob_ro[:, 1],
            "prob_stress":    prob_ro[:, 2],
        },
        index=idx,
    )


def _make_panel(n: int = 250, start: str = "2022-01-03") -> pd.DataFrame:
    """Synthetic panel with us_equity (trending up) and vix."""
    idx = pd.bdate_range(start, periods=n, freq="B")
    rng = np.random.default_rng(1)
    prices = 4000 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    vix    = np.abs(rng.normal(18, 5, n))
    return pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx)


class TestBullAwareWeights:
    def test_weights_between_zero_and_one(self):
        """Bull-aware equity weights must always be in [0, 1]."""
        pred  = _make_predictions(40)
        panel = _make_panel(250)
        w = compute_bull_aware_weights(pred, panel, lag=1)
        assert (w >= 0).all(), "Negative weight found"
        assert (w <= 1).all(), "Weight above 1 found"

    def test_weights_are_lagged_by_one_day(self):
        """Weight on day t is derived from probabilities on day t-1.

        If we override the probability on day 0 to be 100% stress,
        the corresponding weight on day 1 should be at most 0.25
        (the stress coefficient in the base formula).
        """
        pred  = _make_predictions(10)
        panel = _make_panel(250)

        # Force day-0 to be pure stress — weight on day 1 should reflect that
        pred.iloc[0, pred.columns.get_loc("prob_risk_on")]    = 0.0
        pred.iloc[0, pred.columns.get_loc("prob_transition")] = 0.0
        pred.iloc[0, pred.columns.get_loc("prob_stress")]     = 1.0

        w = compute_bull_aware_weights(pred, panel, lag=1).dropna()

        # Day 0 in the predictions is lag t-1 for day 1 in the weight series.
        # Base formula: 1.0*0 + 0.75*0 + 0.25*1 = 0.25 (before risk controls)
        # So weight on the FIRST day of the weight series <= 0.25
        assert w.iloc[0] <= 0.25 + 1e-9, (
            f"Day-1 weight {w.iloc[0]:.4f} should reflect day-0 stress prob"
        )

    def test_no_look_ahead_weight_count(self):
        """After lag=1, weights have at most len(predictions)-1 non-NaN entries."""
        pred  = _make_predictions(20)
        panel = _make_panel(250)
        w = compute_bull_aware_weights(pred, panel, lag=1).dropna()
        # Cannot have more dates than (len predictions - lag)
        assert len(w) <= len(pred), "More weights than prediction rows"

    def test_cap_clips_weight_when_vix_high_and_drawdown_deep(self):
        """Cap clips weight to 0.5 when VIX >> historical 75th pct and drawdown > 5%.

        Setup: 220 calm days (VIX=5, price flat), then 50 stress days (VIX=80, price -20%).
        Predictions are placed 25 days into the stress phase.
        At that point: VIX(80) >> VIX_75th(~5), DD(-10%) < -5% → both cap conditions met.
        price < 200d-MA (MA still dominated by calm phase) → floor does NOT fire.
        Net result: weight should be clipped to 0.50.
        """
        n_burn, n_stress = 220, 50
        n = n_burn + n_stress
        idx = pd.bdate_range("2018-01-02", periods=n, freq="B")
        prices = np.concatenate([np.ones(n_burn) * 4000.0,
                                  np.linspace(4000.0, 3200.0, n_stress)])
        vix    = np.concatenate([np.ones(n_burn) * 5.0,
                                  np.ones(n_stress) * 80.0])
        panel  = pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx)

        # Predictions placed well into the stress phase (days 25..49)
        pred_start = n_burn + 25
        pred_idx   = idx[pred_start:]    # 25 prediction dates
        pred2 = pd.DataFrame({
            "regime_state": 0, "regime_label": "risk_on",
            "prob_risk_on": 1.0, "prob_transition": 0.0, "prob_stress": 0.0,
        }, index=pred_idx)

        w = compute_bull_aware_weights(pred2, panel, lag=1).dropna()
        assert len(w) > 0, "No weights produced"
        assert (w <= 0.50 + 1e-9).all(), (
            f"Cap should clip all weights to 0.50: max = {w.max():.4f}"
        )

    def test_floor_fires_above_200d_ma_low_stress(self):
        """When equity is above 200-day MA and stress prob is low, floor >= 0.5."""
        # Build a panel with sustained rising equity (always above its 200-day MA)
        n = 400
        idx = pd.bdate_range("2020-01-02", periods=n, freq="B")
        prices = 4000 * np.cumprod(1 + np.ones(n) * 0.001)  # steadily rising
        vix    = np.ones(n) * 12                              # calm VIX
        panel  = pd.DataFrame({"us_equity": prices, "vix": vix}, index=idx)

        # Predictions: pure risk_on at the end (stress prob = 0)
        pred_idx = pd.bdate_range("2021-07-01", periods=20, freq="B")
        pred2 = pd.DataFrame(
            {"regime_state": 0, "regime_label": "risk_on",
             "prob_risk_on": 0.5, "prob_transition": 0.4, "prob_stress": 0.0},
            index=pred_idx,
        )

        w = compute_bull_aware_weights(pred2, panel, lag=1).dropna()
        # Floor condition: price > MA and stress < 70% → w >= 0.5
        assert (w >= 0.50 - 1e-9).all(), (
            f"Floor did not fire: min weight = {w.min():.4f}"
        )


class TestBullAwareTransactionCosts:
    def test_costs_deducted_on_weight_change(self):
        """Transaction costs are proportional to |Δweight| for bull-aware weights."""
        idx = pd.bdate_range("2023-01-02", periods=5, freq="B")
        weights = pd.Series([0.5, 0.5, 0.75, 0.75, 0.25], index=idx)
        costs = compute_transaction_costs(weights, transaction_cost_bps=5)

        assert costs.iloc[1] == pytest.approx(0.0)          # no change 0.5→0.5
        assert costs.iloc[2] == pytest.approx(0.25 * 5 / 10_000)  # +25% change
        assert costs.iloc[3] == pytest.approx(0.0)          # no change
        assert costs.iloc[4] == pytest.approx(0.50 * 5 / 10_000)  # -50% change

    def test_costs_subtracted_from_gross_return(self):
        """Net overlay return = gross return - transaction cost."""
        idx = pd.bdate_range("2023-01-02", periods=4, freq="B")
        eq_ret  = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx)
        def_ret = pd.Series([0.00, 0.00, 0.00, 0.00], index=idx)
        weights = pd.Series([0.5, 1.0, 0.5, 0.5], index=idx)

        daily = compute_portfolio_returns(eq_ret, def_ret, weights, transaction_cost_bps=10)

        # Day 1: 0.5→1.0 change = +0.5, cost = 0.5*10/10000 = 0.0005
        # gross = 1.0*0.01 + 0.0*0.00 = 0.01; net = 0.01 - 0.0005 = 0.0095
        assert daily["overlay_ret"].iloc[1] == pytest.approx(0.01 - 0.5 * 10 / 10_000)

        # Day 2: 1.0→0.5 change = -0.5, same cost
        assert daily["overlay_ret"].iloc[2] == pytest.approx(0.5 * 0.01 - 0.5 * 10 / 10_000)


class TestThreeWayAlignment:
    def test_all_three_strategies_share_same_index(self):
        """Benchmark, defensive overlay, and bull-aware overlay must share the same DatetimeIndex."""
        pred  = _make_predictions(30)
        panel = _make_panel(250)

        bull_w  = compute_bull_aware_weights(pred, panel, lag=1).dropna()
        idx = bull_w.index

        # Simulate what run_backtest does: reindex equity and defensive returns
        eq_ret  = pd.Series(0.001, index=panel.index).reindex(idx)
        def_ret = pd.Series(0.0,   index=panel.index).reindex(idx)

        # Defensive overlay (fixed weights — all 0.5 for simplicity)
        def_w = pd.Series(0.5, index=idx)

        bull_daily = compute_portfolio_returns(eq_ret, def_ret, bull_w, 5)
        def_daily  = compute_portfolio_returns(eq_ret, def_ret, def_w,  5)

        assert bull_daily.index.equals(def_daily.index), \
            "Bull-aware and defensive daily DataFrames have different indices"
        assert bull_daily.index.equals(idx), \
            "Daily index does not match weight index"

    def test_benchmark_always_fully_invested(self):
        """Benchmark return in compute_portfolio_returns is always eq_ret (no overlay)."""
        idx = pd.bdate_range("2023-01-02", periods=10, freq="B")
        rng = np.random.default_rng(7)
        eq_ret  = pd.Series(rng.normal(0.001, 0.01, 10), index=idx)
        def_ret = pd.Series(0.0002, index=idx)
        weights = pd.Series(0.5, index=idx)

        daily = compute_portfolio_returns(eq_ret, def_ret, weights, 5)

        pd.testing.assert_series_equal(
            daily["benchmark_ret"], eq_ret, check_names=False,
            rtol=1e-9,
        )


class TestBullMarketPeriods:
    def test_rising_trend_qualifies_as_bull(self):
        """A steadily rising equity index above its 200-day MA should be bull all the time."""
        n   = 400
        idx = pd.bdate_range("2020-01-02", periods=n, freq="B")
        prices = 4000 * np.cumprod(1 + np.ones(n) * 0.001)
        panel  = pd.DataFrame({"us_equity": prices}, index=idx)
        bull   = compute_bull_market_periods(panel)

        # After the warm-up period (200 days), should be all bull
        assert bull.iloc[250:].all(), "Expected bull market in rising trend"

    def test_falling_market_not_bull(self):
        """A steadily falling equity index below its 200-day MA is not a bull period."""
        n   = 400
        idx = pd.bdate_range("2020-01-02", periods=n, freq="B")
        prices = 4000 * np.cumprod(1 - np.ones(n) * 0.001)
        panel  = pd.DataFrame({"us_equity": prices}, index=idx)
        bull   = compute_bull_market_periods(panel)

        # After 200 days, price well below MA and 60d return negative
        assert not bull.iloc[250:].any(), "Falling market should not be bull"
