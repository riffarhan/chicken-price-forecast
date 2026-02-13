"""
Train an XGBoost model for chicken price forecasting.

Loads the generated dataset, engineers features via the shared pipeline,
trains an XGBoost regressor with a linear regression baseline for
comparison, and saves the model and metrics.

Usage:
    python train_model.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    TARGET,
    TEST_PREDICTIONS_PATH,
    XGB_PARAMS,
)
from features import engineer_features, validate_features

warnings.filterwarnings("ignore")


def main() -> None:
    print("=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)

    # --- Load ---
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run  python generate_dataset.py  first.")
        sys.exit(1)

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    # --- Feature engineering (shared pipeline) ---
    print("\nEngineering features...")
    df = engineer_features(df)
    validate_features(df)
    print(f"  Ready: {len(df)} rows, {len(FEATURE_COLUMNS)} features")

    # --- Time-based train/test split (85 / 15) ---
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
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)

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
