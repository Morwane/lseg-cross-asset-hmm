"""Configuration loading from YAML files — no LSEG dependency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT: Path = Path(__file__).parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"

TRADING_DAYS: int = 252


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project identity fields."""

    name: str
    version: str


@dataclass(frozen=True)
class LsegConfig:
    """LSEG session settings."""

    session_name: str


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest date range and cost assumptions."""

    start_date: str
    end_date: Optional[str]
    trading_days: int
    transaction_cost_bps: int
    use_one_day_signal_lag: bool


@dataclass(frozen=True)
class OutputConfig:
    """Relative paths for each output category."""

    audit_dir: str
    data_dir: str
    chart_dir: str
    report_dir: str


@dataclass(frozen=True)
class Settings:
    """Root settings object parsed from config/settings.yaml."""

    project: ProjectConfig
    lseg: LsegConfig
    backtest: BacktestConfig
    outputs: OutputConfig


def load_settings() -> Settings:
    """Load and return project settings from config/settings.yaml."""
    path = CONFIG_DIR / "settings.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Settings(
        project=ProjectConfig(**raw["project"]),
        lseg=LsegConfig(**raw["lseg"]),
        backtest=BacktestConfig(**raw["backtest"]),
        outputs=OutputConfig(**raw["outputs"]),
    )


def load_instruments_config() -> dict:
    """Load instrument universe definition from config/instruments.yaml."""
    path = CONFIG_DIR / "instruments.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config() -> dict:
    """Load HMM and feature configuration from config/model.yaml."""
    path = CONFIG_DIR / "model.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_output_path(settings: Settings, key: str) -> Path:
    """Resolve a named output directory path (e.g. 'audit_dir') relative to project root."""
    value: str = getattr(settings.outputs, key)
    return PROJECT_ROOT / value
