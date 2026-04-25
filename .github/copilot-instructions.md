# Copilot Instructions

This repository is a research-grade cross-asset HMM regime detection and risk overlay engine using LSEG market data.

See `CLAUDE_TASK.md` for the full implementation specification.

## Key conventions

- Python 3.9+, type hints on every function, one-line docstrings
- No hardcoded dates or RICs in `src/` — load exclusively from `config/`
- LSEG audit-first: test every RIC before using it; never silently substitute a broken RIC
- No `inplace=True` in pandas operations
- All time-series DataFrames must use `DatetimeIndex` with `index.name = "date"`
- `TRADING_DAYS = 252` is the canonical constant (defined in `src/config.py`)
- Save charts at 300 dpi to `output/charts/`
- Save tables to `output/reports/` or `output/audit/`
- `output/audit/validated_universe.csv` is the official path for the validated universe

## Architecture

```
config/   ← YAML configuration (instruments, model, settings)
scripts/  ← Thin entrypoints — business logic lives in src/
src/      ← All business logic
tests/    ← Unit tests (no LSEG session required)
output/   ← Generated artefacts (data not committed)
```

## HMM notes

- Use 8 core features in the HMM (see `config/model.yaml:features.hmm_core`)
- Full feature set is for diagnostics and reporting only
- Never assume HMM state 0 is risk_on — label from realised statistics
- Walk-forward validation only; no full-sample backtest presented as tradable
