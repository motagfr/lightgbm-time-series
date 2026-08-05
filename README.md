# LightGBM Time Series Forecasting Suite

A high-performance machine learning repository for Data Science and Time Series Forecasting powered by **LightGBM**, **uv**, and modern Python tooling.

## 🚀 Overview

This repository provides an enterprise-ready environment and framework for building, training, evaluating, and deploying gradient-boosted decision tree models (LightGBM) on temporal data.

### Features
- ⚡ **Fast & Lightweight Dependency Management**: Powered by `uv` for reproducible environments and sub-second dependency locking (`uv.lock`).
- 📈 **Advanced Feature Engineering**: Built-in pipelines for calendar attributes, multi-horizon lags, and rolling window aggregations.
- 🎯 **Temporal Validation**: Out-of-time train/test splits to eliminate lookahead bias and data leakage.
- 📊 **Comprehensive Evaluation**: Automated reporting of RMSE, MAE, MAPE, and feature importances.
- 🐙 **GitHub Integration Ready**: Direct sync support with GitHub user account [`@motagfr`](https://github.com/motagfr).

---

## 🛠️ Quick Start

### 1. Environment Setup with `uv`

Ensure `uv` is installed, then run:

```bash
# Sync dependencies and build virtual environment (.venv)
uv sync

# Run the time-series forecasting demo
uv run python src/forecast_demo.py
```

### 2. Installed Libraries

The environment includes the following data science & ML stack:

| Library | Category | Description |
| :--- | :--- | :--- |
| **`lightgbm`** | Model Engine | Fast, distributed, high-performance gradient boosting framework. |
| **`pandas`** | Data Wrangling | Time series indexing, resamplings, and dataframe manipulations. |
| **`numpy`** | Numerical Computation | High-dimensional array operations. |
| **`scikit-learn`** | Machine Learning | Preprocessing, evaluation metrics, and time series cross-validation (`TimeSeriesSplit`). |
| **`statsmodels`** | Time Series Analysis | Stationarity tests, seasonal decomposition, ACF/PACF analysis. |
| **`optuna`** | Hyperparameter Tuning | Automated Bayesian optimization for LightGBM parameters. |
| **`matplotlib` & `seaborn`** | Visualization | Static plots for forecasts and residual diagnostics. |
| **`plotly`** | Interactive Visuals | Interactive time series graphs and feature analysis charts. |
| **`pyarrow`** | Data Format | High-performance Parquet format storage and retrieval. |
| **`ipykernel`** | Interactive Environment | Jupyter notebook kernel support. |

---

## 📁 Repository Structure

```
LightGBM Projects/
├── .venv/                   # Managed Python virtual environment
├── data/
│   ├── raw/                 # Raw datasets (ignored by git)
│   └── processed/           # Feature-engineered Parquet/CSV data
├── src/
│   └── forecast_demo.py     # Complete LightGBM time series workflow script
├── .gitignore               # Comprehensive Python & ML git ignore
├── pyproject.toml           # uv project configuration
├── uv.lock                  # Exact dependency lockfile
├── README.md                # Main repository documentation
├── GUIDE.md                 # Time-series forecasting best practices guide
└── CONTRIBUTING.md          # Contribution guidelines
```

---

## 🐙 Connecting & Publishing to GitHub (`@motagfr`)

### 1. GitHub CLI Authentication
To connect this repository to your GitHub account (`@motagfr`), authenticate via GitHub CLI:

```bash
gh auth login
```
*Select `GitHub.com` -> `HTTPS` -> `Log in with a web browser` or paste your Personal Access Token (PAT).*

### 2. Creating and Pushing the Remote Repository
Once authenticated, create and push the repository to GitHub in a single command:

```bash
gh repo create motagfr/lightgbm-time-series --public --source=. --remote=origin --push
```

---

## 📖 Related Guides
- Refer to [GUIDE.md](file:///c:/Users/mtg/Desktop/LightGBM%20Projects/GUIDE.md) for feature engineering, time series CV strategies, and hyperparameter tuning for LightGBM.
- Refer to [CONTRIBUTING.md](file:///c:/Users/mtg/Desktop/LightGBM%20Projects/CONTRIBUTING.md) for setup and pull request guidelines.
