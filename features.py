"""Shared feature engineering for training and prediction.

All scripts import from here so the transformation logic stays in one place.
"""

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, PROVINCE_TO_INT


def encode_province(df: pd.DataFrame) -> pd.DataFrame:
    """Map province names to integers using a fixed, deterministic mapping.

    Unlike sklearn's LabelEncoder, this always produces the same encoding
    regardless of which provinces are present in *df*.
    """
    unknown = set(df["province"].unique()) - set(PROVINCE_TO_INT)
    if unknown:
        raise ValueError(f"Unknown province(s): {unknown}")
    df["province_encoded"] = df["province"].map(PROVINCE_TO_INT)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline.

    Expects at minimum the raw columns produced by ``generate_dataset.py``:
    ``date, province, chicken_price, feed_price, supply_index,
    weather_severity, is_holiday, week_of_year, month, quarter, year,
    logistics_cost``.

    Returns a DataFrame with all ``FEATURE_COLUMNS`` populated and NaN rows
    (from lag/rolling operations) removed.
    """
    required = [
        "date",
        "province",
        "chicken_price",
        "feed_price",
        "supply_index",
        "weather_severity",
        "is_holiday",
        "week_of_year",
        "month",
        "quarter",
        "logistics_cost",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["province", "date"]).reset_index(drop=True)

    # Lag features
    for lag in [1, 2, 4]:
        df[f"price_lag_{lag}"] = df.groupby("province")["chicken_price"].shift(lag)
        df[f"feed_lag_{lag}"] = df.groupby("province")["feed_price"].shift(lag)

    # Rolling statistics
    df["price_rolling_mean_4"] = df.groupby("province")["chicken_price"].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )
    df["price_rolling_std_4"] = df.groupby("province")["chicken_price"].transform(
        lambda x: x.rolling(window=4, min_periods=1).std()
    )

    # Change features
    df["price_change_1w"] = df.groupby("province")["chicken_price"].diff(1)
    df["feed_change_1w"] = df.groupby("province")["feed_price"].diff(1)

    # Ratio
    df["feed_to_price_ratio"] = df["feed_price"] / df["chicken_price"]

    # Province encoding (deterministic)
    df = encode_province(df)

    # Interaction features
    df["feed_x_holiday"] = df["feed_price"] * df["is_holiday"]
    df["supply_x_weather"] = df["supply_index"] * df["weather_severity"]

    # Cyclical encodings
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with NaN (from lag/rolling features)")

    return df


def validate_features(df: pd.DataFrame) -> None:
    """Raise if any expected feature column is missing from *df*."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
