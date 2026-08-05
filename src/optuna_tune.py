"""
Optuna Automated Hyperparameter Optimization for LightGBM Time Series
----------------------------------------------------------------------
Uses Bayesian Optimization (TPE sampler) to find the best LightGBM hyperparameters.
"""

import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.forecast_demo import generate_synthetic_time_series, create_time_series_features

# Suppress Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train, y_train, X_val, y_val):
    """Objective function for Optuna hyperparameter optimization trial."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "random_state": 42,
        "verbose": -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


def run_tuning(n_trials: int = 25):
    print("Generating dataset & engineering features...")
    raw_df = generate_synthetic_time_series(n_days=730)
    featured_df = create_time_series_features(raw_df)

    feature_cols = [c for c in featured_df.columns if c not in ["date", "sales"]]
    target_col = "sales"

    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:split_idx]
    test_df = featured_df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"Starting Optuna Hyperparameter Optimization ({n_trials} trials)...")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)

    print("\n--- Optuna Optimization Completed ---")
    print(f"Best Trial RMSE: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # Retrain model with best parameters
    best_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbose": -1,
        **study.best_params
    }

    tuned_model = lgb.LGBMRegressor(**best_params)
    tuned_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    tuned_preds = tuned_model.predict(X_test)
    tuned_mape = mean_absolute_percentage_error(y_test, tuned_preds) * 100

    print(f"\nFinal Tuned Model MAPE: {tuned_mape:.2f}%")
    return study, tuned_model


if __name__ == "__main__":
    run_tuning(n_trials=25)
