"""
Generate a realistic synthetic chicken price dataset.

Creates weekly price data across 15 Indonesian provinces with realistic
seasonality, holiday effects, feed cost correlations, and supply shocks.

Usage:
    python generate_dataset.py
"""

import os

import numpy as np
import pandas as pd

PROVINCES = [
    "Jakarta",
    "West Java",
    "Central Java",
    "East Java",
    "Banten",
    "Yogyakarta",
    "Bali",
    "South Sumatra",
    "North Sumatra",
    "Lampung",
    "West Kalimantan",
    "South Sulawesi",
    "East Kalimantan",
    "Riau",
    "Aceh",
]

OUTPUT_PATH = "data/chicken_price_dataset.csv"


def generate_dataset(
    start_year: int = 2021, end_year: int = 2022, seed: int = 42
) -> pd.DataFrame:
    """Generate a synthetic chicken price dataset.

    Args:
        start_year: First year of data.
        end_year: Last year of data.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with weekly chicken price data for all provinces.
    """
    np.random.seed(seed)

    date_range = pd.date_range(
        start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="W"
    )

    rows: list[dict] = []

    for province in PROVINCES:
        base_price = np.random.uniform(24_000, 28_000)
        price_volatility = np.random.uniform(0.05, 0.15)
        logistics_cost = np.random.uniform(0, 2_000)

        for i, date in enumerate(date_range):
            week_of_year = date.isocalendar()[1]
            month = date.month
            quarter = (month - 1) // 3 + 1
            year = date.year

            # Seasonality — Q4 prices tend to be higher
            seasonal_effect = 0.0
            if quarter == 4:
                seasonal_effect = np.random.uniform(1_000, 2_000)
            elif quarter == 1:
                seasonal_effect = np.random.uniform(-500, 500)

            # Holiday effects (Idul Fitri, Christmas/New Year, Independence Day)
            holiday_effect = 0.0
            is_holiday = 0
            if (year == 2021 and week_of_year in [19, 20]) or (
                year == 2022 and week_of_year in [17, 18]
            ):
                holiday_effect = np.random.uniform(3_000, 5_000)
                is_holiday = 1
            elif week_of_year in [51, 52, 1]:
                holiday_effect = np.random.uniform(1_500, 3_000)
                is_holiday = 1
            elif week_of_year in [32, 33]:
                holiday_effect = np.random.uniform(800, 1_500)
                is_holiday = 1

            # Feed price — correlated with chicken price, with trend
            feed_base = 14_000 + i * 20
            feed_noise = np.random.normal(0, 800)
            feed_seasonality = 500 * np.sin(2 * np.pi * week_of_year / 52)
            feed_price = feed_base + feed_noise + feed_seasonality

            # Supply shocks (5% chance each week)
            supply_shock = 0.0
            if np.random.random() < 0.05:
                supply_shock = np.random.uniform(-4_000, -2_000)

            supply_index = np.random.uniform(80, 120)
            supply_effect = (100 - supply_index) * 30

            # Weather severity (0=normal, 1=moderate, 2=severe)
            weather_severity = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            weather_effect = weather_severity * np.random.uniform(200, 800)

            trend = i * 15
            noise = np.random.normal(0, 500)

            chicken_price = (
                base_price
                + logistics_cost
                + trend
                + seasonal_effect
                + holiday_effect
                + (feed_price - 14_000) * 0.6
                + supply_effect
                + supply_shock
                + weather_effect
                + noise
            )

            # Feed impact is amplified during holidays
            if is_holiday:
                chicken_price += (feed_price - 14_000) * 0.4

            chicken_price = max(chicken_price, 15_000)

            rows.append(
                {
                    "date": date,
                    "province": province,
                    "chicken_price": round(chicken_price, 2),
                    "feed_price": round(feed_price, 2),
                    "supply_index": round(supply_index, 2),
                    "weather_severity": weather_severity,
                    "is_holiday": is_holiday,
                    "week_of_year": week_of_year,
                    "month": month,
                    "quarter": quarter,
                    "year": year,
                    "logistics_cost": round(logistics_cost, 2),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["province", "date"]).reset_index(drop=True)

    # Lagged features
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

    return df


def main() -> None:
    print("=" * 60)
    print("GENERATING CHICKEN PRICE DATASET")
    print("=" * 60)

    df = generate_dataset()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Provinces: {df['province'].nunique()}")
    print(f"\nPrice statistics:\n{df['chicken_price'].describe()}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
