"""Shared utilities: logging setup and filesystem helpers."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging with a consistent format and return the root logger."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()


def ensure_output_dirs(*dirs: Path) -> None:
    """Create all provided directories (and any missing parents) if they do not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
