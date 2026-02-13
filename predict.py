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

from config import (
    BASE_DIR,
    DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUT_PATH,
    RMSE_THRESHOLD,
    TARGET,
)
from features import engineer_features, validate_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                BASE_DIR,
                "logs",
                f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


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
        if not os.path.exists(DATA_PATH):
            logger.error("Dataset not found at %s — run generate_dataset.py first.", DATA_PATH)
            return 1
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        logger.info("  Loaded %d records (%d provinces)", len(df), df["province"].nunique())

        # 2. Feature engineering (shared pipeline)
        logger.info("Step 2: Engineering features...")
        df = engineer_features(df)
        validate_features(df)
        logger.info("  Ready: %d rows", len(df))

        # 3. Load model & predict
        logger.info("Step 3: Loading model and predicting...")
        if not os.path.exists(MODEL_PATH):
            logger.error("Model not found at %s — run train_model.py first.", MODEL_PATH)
            return 1

        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)

        predictions = model.predict(df[FEATURE_COLUMNS])
        df["predicted_price"] = predictions
        df["prediction_error"] = df[TARGET] - df["predicted_price"]
        df["prediction_error_pct"] = (df["prediction_error"] / df[TARGET]) * 100
        logger.info("  Generated %d predictions", len(predictions))

        # Sanity check — flag unreasonable predictions
        neg = (predictions < 0).sum()
        if neg:
            logger.warning("  %d negative predictions detected — check input data", neg)

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
                    float(shap_values[i][np.argmax(np.abs(shap_values[i]))])
                    for i in range(len(shap_values))
                ],
            }
        )
        logger.info("  SHAP values calculated for %d provinces", len(shap_df))

        # 5. Validate
        logger.info("Step 5: Validating predictions...")
        rmse = float(np.sqrt(np.mean(df["prediction_error"] ** 2)))
        mae = float(np.mean(np.abs(df["prediction_error"])))
        mape = float(np.mean(np.abs(df["prediction_error_pct"])))
        bias = float(np.mean(df["prediction_error"]))

        logger.info("  RMSE: %.2f IDR/kg", rmse)
        logger.info("  MAE:  %.2f IDR/kg", mae)
        logger.info("  MAPE: %.2f%%", mape)
        logger.info("  Bias: %+.2f IDR/kg", bias)

        status = "HEALTHY"
        if rmse > RMSE_THRESHOLD:
            logger.warning(
                "RMSE (%.0f) exceeds threshold (%d) — consider retraining.",
                rmse,
                RMSE_THRESHOLD,
            )
            status = "DEGRADED"

        # 6. Save outputs
        logger.info("Step 6: Saving outputs...")

        df.to_csv(OUTPUT_PATH, index=False)
        logger.info("  Predictions: %s", OUTPUT_PATH)

        shap_path = os.path.join(
            BASE_DIR,
            "outputs",
            f"shap_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        )
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
