"""
Production prediction pipeline for chicken price forecasting.

Loads the latest data, applies feature engineering, generates predictions
with SHAP explanations, validates performance, and saves outputs.

Usage:
    python predict.py
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "models/chicken_price_xgboost_model.json"
DATA_PATH = "data/chicken_price_dataset.csv"
OUTPUT_PATH = "outputs/predictions_latest.csv"
METRICS_PATH = "outputs/model_metrics.csv"
RMSE_THRESHOLD = 1200  # IDR/kg — retrain if exceeded

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the feature engineering pipeline used during training."""
    df = df.sort_values(["province", "date"]).reset_index(drop=True)

    for lag in [1, 2, 4]:
        df[f"price_lag_{lag}"] = df.groupby("province")["chicken_price"].shift(lag)
        df[f"feed_lag_{lag}"] = df.groupby("province")["feed_price"].shift(lag)

    df["price_rolling_mean_4"] = df.groupby("province")["chicken_price"].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    df["price_rolling_std_4"] = df.groupby("province")["chicken_price"].transform(
        lambda x: x.rolling(window=4, min_periods=1).std()
    )

    df["price_change_1w"] = df.groupby("province")["chicken_price"].diff(1)
    df["feed_change_1w"] = df.groupby("province")["feed_price"].diff(1)
    df["feed_to_price_ratio"] = df["feed_price"] / df["chicken_price"]

    le = LabelEncoder()
    df["province_encoded"] = le.fit_transform(df["province"])

    df["feed_x_holiday"] = df["feed_price"] * df["is_holiday"]
    df["supply_x_weather"] = df["supply_index"] * df["weather_severity"]
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df.dropna()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> int:
    """Run the full prediction pipeline. Returns 0 on success, 1 on failure."""
    try:
        logger.info("=" * 60)
        logger.info("STARTING PREDICTION PIPELINE  —  %s", datetime.now())
        logger.info("=" * 60)

        # 1. Load data
        logger.info("Step 1: Loading data...")
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        logger.info("  Loaded %d records", len(df))

        # 2. Feature engineering
        logger.info("Step 2: Engineering features...")
        df = engineer_features(df)
        logger.info("  Ready: %d rows after cleaning", len(df))

        # 3. Load model & predict
        logger.info("Step 3: Loading model and predicting...")
        if not os.path.exists(MODEL_PATH):
            logger.error("Model not found at %s — run train_model.py first", MODEL_PATH)
            return 1

        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)

        predictions = model.predict(df[FEATURE_COLUMNS])
        df["predicted_price"] = predictions
        df["prediction_error"] = df["chicken_price"] - df["predicted_price"]
        df["prediction_error_pct"] = (
            df["prediction_error"] / df["chicken_price"]
        ) * 100
        logger.info("  Generated %d predictions", len(predictions))

        # 4. SHAP explanations (latest record per province)
        logger.info("Step 4: Calculating SHAP explanations...")
        latest = df.groupby("province").tail(1)
        X_latest = latest[FEATURE_COLUMNS]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_latest)

        shap_df = pd.DataFrame(
            {
                "province": latest["province"].values,
                "date": latest["date"].values,
                "predicted_price": latest["predicted_price"].values,
                "top_driver": [
                    FEATURE_COLUMNS[np.argmax(np.abs(shap_values[i]))]
                    for i in range(len(shap_values))
                ],
                "top_driver_impact": [
                    shap_values[i][np.argmax(np.abs(shap_values[i]))]
                    for i in range(len(shap_values))
                ],
            }
        )
        logger.info("  SHAP values calculated for %d provinces", len(shap_df))

        # 5. Validate
        logger.info("Step 5: Validating predictions...")
        rmse = np.sqrt(np.mean(df["prediction_error"] ** 2))
        mae = np.mean(np.abs(df["prediction_error"]))
        mape = np.mean(np.abs(df["prediction_error_pct"]))
        bias = np.mean(df["prediction_error"])

        logger.info("  RMSE: %.2f IDR/kg", rmse)
        logger.info("  MAE:  %.2f IDR/kg", mae)
        logger.info("  MAPE: %.2f%%", mape)
        logger.info("  Bias: %+.2f IDR/kg", bias)

        status = "HEALTHY"
        if rmse > RMSE_THRESHOLD:
            logger.warning(
                "RMSE (%.0f) exceeds threshold (%d) — consider retraining",
                rmse,
                RMSE_THRESHOLD,
            )
            status = "DEGRADED"

        # 6. Save outputs
        logger.info("Step 6: Saving outputs...")

        df.to_csv(OUTPUT_PATH, index=False)
        logger.info("  Predictions: %s", OUTPUT_PATH)

        shap_path = f"outputs/shap_summary_{datetime.now().strftime('%Y%m%d')}.csv"
        shap_df.to_csv(shap_path, index=False)
        logger.info("  SHAP summary: %s", shap_path)

        new_row = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "bias": bias,
                    "status": status,
                    "n_predictions": len(df),
                }
            ]
        )
        if os.path.exists(METRICS_PATH):
            history = pd.read_csv(METRICS_PATH)
            history = pd.concat([history, new_row], ignore_index=True)
            history.to_csv(METRICS_PATH, index=False)
        else:
            new_row.to_csv(METRICS_PATH, index=False)
        logger.info("  Metrics: %s", METRICS_PATH)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED — Status: %s", status)
        logger.info("=" * 60)
        return 0

    except Exception:
        logger.exception("PIPELINE FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_pipeline())
