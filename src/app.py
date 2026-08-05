"""
LightGBM Cloud Run Service API
------------------------------
Exposes time series forecasting & Optuna hyperparameter tuning as REST APIs on Google Cloud Run.
"""

import os
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, jsonify, request
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.forecast_demo import generate_synthetic_time_series, create_time_series_features

optuna.logging.set_verbosity(optuna.logging.WARNING)
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "LightGBM Time Series Forecasting Engine",
        "status": "online",
        "cloud_provider": "Google Cloud Run",
        "endpoints": {
            "/forecast": "Run baseline time-series model training & evaluation",
            "/optuna": "Run Optuna automated hyperparameter tuning",
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

@app.route("/optuna", methods=["GET"])
def tune_optuna():
    n_trials = int(request.args.get("trials", 15))
    raw_df = generate_synthetic_time_series(n_days=730)
    featured_df = create_time_series_features(raw_df)
    
    feature_cols = [c for c in featured_df.columns if c not in ["date", "sales"]]
    target_col = "sales"
    
    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx]
    test_df = featured_df.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    def objective(trial):
        p = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1
        }
        m = lgb.LGBMRegressor(**p)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(30, verbose=False)])
        pr = m.predict(X_test)
        return float(np.sqrt(mean_squared_error(y_test, pr)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    best_params = {"objective": "regression", "metric": "rmse", "boosting_type": "gbdt", "random_state": 42, "verbose": -1, **study.best_params}
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(30, verbose=False)])
    
    final_preds = best_model.predict(X_test)
    final_rmse = float(np.sqrt(mean_squared_error(y_test, final_preds)))
    final_mape = float(mean_absolute_percentage_error(y_test, final_preds) * 100)
    
    return jsonify({
        "status": "success",
        "optimization": "Optuna Bayesian TPE",
        "trials_completed": n_trials,
        "best_rmse": round(final_rmse, 4),
        "best_mape_percent": round(final_mape, 2),
        "best_hyperparameters": study.best_params
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
