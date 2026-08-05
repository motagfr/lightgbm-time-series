"""
LightGBM Time Series Forecasting Demo
-------------------------------------
Demonstrates time-series feature engineering (lags, rolling stats, calendar features),
temporal train/test splitting, model training with LightGBM, and evaluation metrics (RMSE, MAE, MAPE).
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def generate_synthetic_time_series(n_days: int = 730) -> pd.DataFrame:
    """Generate daily time-series data with trend, seasonality, and noise."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
    
    # Trend + Weekly Seasonality + Annual Seasonality + Noise
    t = np.arange(n_days)
    trend = 0.05 * t
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)
    annual_seasonality = 25 * np.cos(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, 5, size=n_days)
    
    y = 100 + trend + weekly_seasonality + annual_seasonality + noise
    df = pd.DataFrame({"date": dates, "sales": y})
    return df


def create_time_series_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    """Engineer temporal, lag, and rolling statistics features."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    
    # 1. Calendar / Temporal Features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    
    # 2. Lag Features
    lags = [1, 7, 14, 28]
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
        
    # 3. Rolling Window Features
    windows = [7, 14, 28]
    for w in windows:
        df[f"rolling_mean_{w}"] = df[target_col].shift(1).rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df[target_col].shift(1).rolling(window=w).std()
        df[f"rolling_max_{w}"] = df[target_col].shift(1).rolling(window=w).max()
        df[f"rolling_min_{w}"] = df[target_col].shift(1).rolling(window=w).min()

    # Drop rows with NaN due to lag/rolling shifts
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    print("Generating synthetic time series dataset...")
    raw_df = generate_synthetic_time_series(n_days=730)
    
    print("Engineering features (lags, rolling statistics, calendar dynamics)...")
    featured_df = create_time_series_features(raw_df)
    
    feature_cols = [c for c in featured_df.columns if c not in ["date", "sales"]]
    target_col = "sales"
    
    # 4. Out-of-time Temporal Split (80% Train, 20% Test)
    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx]
    test_df = featured_df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # 5. LightGBM Model Configuration
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # 6. Fit Model with Early Stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # 7. Predictions & Evaluation
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    
    print("\n--- Evaluation Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
    
    # 8. Feature Importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
