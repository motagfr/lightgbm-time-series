# LightGBM Time Series Forecasting

Hi there! 👋 This is a project where I explore time series forecasting using LightGBM. It includes feature engineering (lags, rolling averages, calendar dynamics), model training, and simple pipelines for saving artifacts over to Google Cloud (GCS & BigQuery) and Microsoft Azure (Blob Storage).

I used `uv` to keep package management fast and reliable.

---

## What's in this repo

- **`src/forecast_demo.py`**: Demo script that generates sample time series data, builds lag and rolling window features, trains a LightGBM regressor, and outputs standard evaluation metrics (RMSE, MAE, MAPE).
- **`src/cloud_pipeline.py`**: A helper script to save metrics and data artifacts to GCP and Azure storage.
- **`infrastructure/main.tf`**: A simple Terraform script to set up cloud buckets and datasets.

---

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run the forecasting model**:
   ```bash
   uv run python src/forecast_demo.py
   ```

3. **Run the cloud export script**:
   ```bash
   uv run python src/cloud_pipeline.py --provider both
   ```

---

## Note

I put this project together to work on time series forecasting and multi-cloud workflows. AI coding tools were used to help scaffold parts of the project setup and code structure.
