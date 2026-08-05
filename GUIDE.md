# Time Series Forecasting with LightGBM: Best Practices Guide

This guide outlines core principles, feature engineering techniques, cross-validation strategies, and model tuning patterns for time-series forecasting using **LightGBM**.

---

## 1. Feature Engineering for Time Series

Unlike traditional autoregressive statistical models (e.g. ARIMA), GBDT models like LightGBM do not natively understand temporal order unless explicit time-dependent features are constructed.

### A. Calendar & Seasonal Features
- **Cyclical Encoding**: Transform periodic features (e.g. month, dayofweek, hour) using sine and cosine transformations:
  $$\text{sin\_month} = \sin\left(\frac{2\pi \times \text{month}}{12}\right)$$
  $$\text{cos\_month} = \cos\left(\frac{2\pi \times \text{month}}{12}\right)$$
- **Holiday & Event Flags**: Binary indicators for public holidays, sales events, or promotional dates.

### B. Lag Features
Lag features capture direct autoregressive dependency.
- For a forecast horizon of $h$ steps ahead, ensure all lag features used during training are $\ge h$ steps back to avoid leakage during multi-step forecasting.

### C. Rolling & Expanding Statistics
- **Rolling Aggregations**: Rolling mean, standard deviation, min, max, and quantiles over windows (e.g., 7, 14, 28 days).
- **Shifted Windows**: Always apply `.shift(1)` before calculating rolling statistics on historical values to prevent including the target value of the prediction timestamp.

---

## 2. Validation Strategy (Preventing Data Leakage)

> [!IMPORTANT]
> **Never use standard K-Fold Cross Validation for time series data!** Standard random K-Fold shuffles future observations into training folds, causing severe lookahead bias.

### Recommended Approaches:
1. **Out-of-Time Temporal Split**: Split data chronologically (e.g., first 80% train, remaining 20% test).
2. **TimeSeriesSplit / Expanding Window CV**: Train on expanding temporal windows:
   - Fold 1: Train [T1 .. T100] -> Valid [T101 .. T120]
   - Fold 2: Train [T1 .. T120] -> Valid [T121 .. T140]
   - Fold 3: Train [T1 .. T140] -> Valid [T141 .. T160]

---

## 3. LightGBM Hyperparameter Tuning Strategy

Key LightGBM parameters for time series optimization:

| Parameter | Default | Recommended Search Range | Notes |
| :--- | :--- | :--- | :--- |
| `n_estimators` | 100 | 500 – 3000 | Use with `early_stopping_rounds`. |
| `learning_rate` | 0.1 | 0.01 – 0.05 | Smaller values improve generalization on noisy data. |
| `num_leaves` | 31 | 15 – 63 | Lower values prevent overfitting on smaller temporal datasets. |
| `max_depth` | -1 | 4 – 10 | Constrains tree depth. |
| `subsample` (`bagging_fraction`) | 1.0 | 0.7 – 0.9 | Stochastic sub-sampling of rows. |
| `colsample_bytree` (`feature_fraction`) | 1.0 | 0.6 – 0.9 | Sub-sampling of features. |
| `min_child_samples` | 20 | 20 – 100 | Minimum data in one leaf. |

---

## 4. Example Optuna Integration

```python
import optuna
import lightgbm as lgb

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "verbose": -1
    }
    
    # Fit model with early stopping on validation set
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)

# Run optimization study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best Params:", study.best_params)
```
