"""Tests for config loading — no LSEG session required."""
from __future__ import annotations

from src.config import (
    TRADING_DAYS,
    load_instruments_config,
    load_model_config,
    load_settings,
)


def test_load_settings_returns_settings_object():
    """Settings object is returned without error."""
    settings = load_settings()
    assert settings is not None


def test_project_name_is_set():
    """Project name matches the expected repository name."""
    settings = load_settings()
    assert settings.project.name == "lseg-cross-asset-hmm-regime-risk-overlay"


def test_lseg_session_name():
    """LSEG session name is the desktop.workspace entry."""
    settings = load_settings()
    assert settings.lseg.session_name == "desktop.workspace"


def test_backtest_start_date_format():
    """Backtest start date is a non-empty ISO date string."""
    settings = load_settings()
    assert settings.backtest.start_date
    # Must be parseable as ISO date
    from datetime import date
    date.fromisoformat(settings.backtest.start_date)


def test_backtest_end_date_is_none_by_default():
    """end_date defaults to None (open-ended backtest)."""
    settings = load_settings()
    assert settings.backtest.end_date is None


def test_trading_days_constant():
    """TRADING_DAYS canonical value is 252."""
    assert TRADING_DAYS == 252


def test_backtest_trading_days_matches_constant():
    """settings.backtest.trading_days equals the TRADING_DAYS constant."""
    settings = load_settings()
    assert settings.backtest.trading_days == TRADING_DAYS


def test_signal_lag_enabled():
    """One-day signal lag is enabled by default."""
    settings = load_settings()
    assert settings.backtest.use_one_day_signal_lag is True


def test_output_dirs_end_with_expected_names():
    """Output directory paths contain their logical names."""
    settings = load_settings()
    assert "audit" in settings.outputs.audit_dir
    assert "data" in settings.outputs.data_dir
    assert "charts" in settings.outputs.chart_dir
    assert "reports" in settings.outputs.report_dir


def test_load_instruments_config_has_equity():
    """Instruments config contains equity block."""
    cfg = load_instruments_config()
    assert "equity" in cfg
    assert "candidates" in cfg["equity"]
    assert len(cfg["equity"]["candidates"]) > 0


def test_load_instruments_config_has_true_yield_curve():
    """Instruments config contains all four true yield curve RICs."""
    cfg = load_instruments_config()
    assert "rates_true_yield_curve" in cfg
    yield_cfg = cfg["rates_true_yield_curve"]
    for tenor in ("us_2y", "us_5y", "us_10y", "us_30y"):
        assert tenor in yield_cfg, f"Missing {tenor} in rates_true_yield_curve"
        assert yield_cfg[tenor]["field"] == "TR.MIDYIELD"


def test_load_model_config_has_hmm_section():
    """Model config contains an hmm section with required keys."""
    cfg = load_model_config()
    assert "hmm" in cfg
    hmm = cfg["hmm"]
    assert "n_components_candidates" in hmm
    assert "covariance_type" in hmm
    assert "covariance_type_fallback" in hmm


def test_model_config_covariance_fallback_is_diag():
    """HMM covariance fallback is diag (dimensionality safeguard)."""
    cfg = load_model_config()
    assert cfg["hmm"]["covariance_type_fallback"] == "diag"


def test_model_config_hmm_core_has_eight_features():
    """HMM core feature list contains exactly 8 features."""
    cfg = load_model_config()
    core = cfg["features"]["hmm_core"]
    assert len(core) == 8, f"Expected 8 HMM core features, got {len(core)}: {core}"


def test_model_config_risk_overlay_has_three_regimes():
    """Risk overlay defines weights for risk_on, transition, and stress."""
    cfg = load_model_config()
    mapping = cfg["risk_overlay"]["regime_mapping"]
    assert set(mapping.keys()) == {"risk_on", "transition", "stress"}
