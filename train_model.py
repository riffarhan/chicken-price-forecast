"""
Train an XGBoost model for chicken price forecasting.

Loads the generated dataset, engineers features, trains an XGBoost regressor
with a linear regression baseline for comparison, and saves the model + metrics.

Usage:
    python train_model.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_PATH = "data/chicken_price_dataset.csv"
MODEL_PATH = "models/chicken_price_xgboost_model.json"
FEATURE_IMPORTANCE_PATH = "outputs/feature_importance.csv"
TEST_PREDICTIONS_PATH = "outputs/test_predictions.csv"

FEATURE_COLUMNS = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_4",
    "feed_lag_1",
    "feed_lag_2",
    "feed_price",
    "supply_index",
    "weather_severity",
    "week_of_year",
    "month",
    "quarter",
    "week_sin",
    "week_cos",
    "is_holiday",
    "price_rolling_mean_4",
    "price_rolling_std_4",
    "price_change_1w",
    "feed_change_1w",
    "feed_to_price_ratio",
    "feed_x_holiday",
    "supply_x_weather",
    "province_encoded",
    "logistics_cost",
]

TARGET = "chicken_price"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.3,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and cyclical features used for training."""
    le = LabelEncoder()
    df["province_encoded"] = le.fit_transform(df["province"])
    df["feed_x_holiday"] = df["feed_price"] * df["is_holiday"]
    df["supply_x_weather"] = df["supply_index"] * df["weather_severity"]
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


def main() -> None:
    print("=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)

    # --- Load & clean ---
    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    df = df.dropna().copy()
    df = add_derived_features(df)
    print(f"  After cleaning: {len(df)} rows")

    # --- Time-based train/test split (85/15) ---
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * 0.85)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")

    # --- Baseline: Linear Regression ---
    print("\nTraining baseline (Linear Regression)...")
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
    print(f"  RMSE: {rmse_lr:.2f} IDR/kg")
    print(f"  MAPE: {mape_lr:.2f}%")

    # --- XGBoost ---
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    print("  Training complete.")

    # --- Evaluate ---
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    improvement = (rmse_lr - rmse_test) / rmse_lr * 100
    overfitting_gap = (rmse_test - rmse_train) / rmse_train * 100

    print(f"\nXGBoost Results:")
    print(f"  Train RMSE: {rmse_train:.2f} IDR/kg")
    print(f"  Test RMSE:  {rmse_test:.2f} IDR/kg")
    print(f"  Test MAE:   {mae_test:.2f} IDR/kg")
    print(f"  Test R2:    {r2_test:.4f}")
    print(f"  Test MAPE:  {mape_test:.2f}%")
    print(f"\n  Improvement over LR: {improvement:.1f}%")
    print(f"  Train-test gap:      {overfitting_gap:.1f}%", end="")
    print("  (OK)" if overfitting_gap < 10 else "  (possible overfitting)")

    # --- Feature importance ---
    importance = (
        pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\nTop 10 features:\n{importance.head(10).to_string(index=False)}")

    # --- Save artifacts ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    model.save_model(MODEL_PATH)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    predictions = test[["date", "province", TARGET]].copy()
    predictions["predicted_price"] = y_pred_test
    predictions["error"] = predictions[TARGET] - predictions["predicted_price"]
    predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    print(f"\nSaved:")
    print(f"  Model:              {MODEL_PATH}")
    print(f"  Feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"  Test predictions:   {TEST_PREDICTIONS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
