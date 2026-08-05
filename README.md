# LightGBM Time Series Forecasting

A Python project for data science and time series forecasting built with **LightGBM**, managed with **`uv`**, and integrated with **Google Cloud Platform (GCP)** and **Microsoft Azure**.

---

## 📌 Project Overview

This repository implements a time series forecasting workflow using gradient boosted decision trees (LightGBM):
- **Data & Feature Engineering**: Creates calendar dynamics, lag features, and rolling window statistics from temporal data.
- **Model Training & Evaluation**: Fits a LightGBM regressor using out-of-time train/test splits and evaluates metrics (RMSE, MAE, MAPE).
- **Cloud Pipelines**: Includes export utilities for saving model checkpoints and predictions to Google Cloud Platform (Cloud Storage, BigQuery) and Microsoft Azure (Blob Storage).
- **Infrastructure as Code**: Terraform configurations for cloud resource setup.

---

## 🛠️ Quick Start

### 1. Installation
Ensure `uv` is installed, then build the virtual environment:
```bash
uv sync
```

### 2. Run Forecasting Demo
Execute the time series forecasting script:
```bash
uv run python src/forecast_demo.py
```

### 3. Run Cloud Export Pipeline
```bash
uv run python src/cloud_pipeline.py --provider both
```

---

## 🤖 Disclosure

AI agents were utilized for parts of the project setup, code structuring, dependency configuration, and pipeline generation.
