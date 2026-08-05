# Contributing to LightGBM Time Series Projects

Thank you for contributing! Please follow these guidelines to keep the codebase clean and reproducible.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/motagfr/lightgbm-time-series.git
   cd lightgbm-time-series
   ```

2. Sync virtual environment with `uv`:
   ```bash
   uv sync
   ```

3. Run verification / demo script:
   ```bash
   uv run python src/forecast_demo.py
   ```

## Adding Dependencies

Always add third-party packages using `uv` to keep `pyproject.toml` and `uv.lock` synchronized:

```bash
uv add <package_name>
```

## Commit Guidelines

Use clear, descriptive commit messages adhering to standard conventions:
- `feat:` New forecasting feature, model module, or script.
- `fix:` Bug fixes or data processing patch.
- `docs:` Documentation improvements.
- `refactor:` Code reorganization without functional changes.
