"""
LightGBM Cloud Run Service API
------------------------------
Exposes the time series forecasting pipeline as a REST API on Google Cloud Run.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, jsonify
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.forecast_demo import generate_synthetic_time_series, create_time_series_features

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "LightGBM Time Series Forecasting Engine",
        "status": "online",
        "cloud_provider": "Google Cloud Run",
        "endpoints": {
            "/forecast": "Run time-series model training & evaluation",
            "/health": "Service health check"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/forecast", methods=["GET"])
def forecast():
    raw_df = generate_synthetic_time_series(n_days=730)
    featured_df = create_time_series_features(raw_df)
    
    feature_cols = [c for c in featured_df.columns if c not in ["date", "sales"]]
    target_col = "sales"
    
    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx]
    test_df = featured_df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(30, verbose=False)])
    
    preds = model.predict(X_test)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    mape = float(mean_absolute_percentage_error(y_test, preds) * 100)
    
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    top_features = importance_df.head(5).to_dict(orient="records")
    
    return jsonify({
        "status": "success",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape_percent": round(mape, 2)
        },
        "top_features": top_features
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
