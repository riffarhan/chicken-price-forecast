"""Shared configuration for the chicken price forecasting project."""

import os

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so scripts work from any cwd)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "chicken_price_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "chicken_price_xgboost_model.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "predictions_latest.csv")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "model_metrics.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "outputs", "feature_importance.csv")
TEST_PREDICTIONS_PATH = os.path.join(BASE_DIR, "outputs", "test_predictions.csv")

# ---------------------------------------------------------------------------
# Provinces — fixed ordering guarantees consistent encoding across all scripts
# ---------------------------------------------------------------------------
PROVINCES = [
    "Aceh",
    "Bali",
    "Banten",
    "Central Java",
    "East Java",
    "East Kalimantan",
    "Jakarta",
    "Lampung",
    "North Sumatra",
    "Riau",
    "South Sulawesi",
    "South Sumatra",
    "West Java",
    "West Kalimantan",
    "Yogyakarta",
]

PROVINCE_TO_INT: dict[str, int] = {p: i for i, p in enumerate(PROVINCES)}

# ---------------------------------------------------------------------------
# Feature columns expected by the model
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Monitoring thresholds
# ---------------------------------------------------------------------------
RMSE_THRESHOLD = 1200  # IDR/kg — pipeline flags DEGRADED above this
