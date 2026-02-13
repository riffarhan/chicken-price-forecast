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

from config import DATA_PATH, PROVINCES


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

            # Feed price — correlated with chicken price, with upward trend
            feed_base = 14_000 + i * 20
            feed_noise = np.random.normal(0, 800)
            feed_seasonality = 500 * np.sin(2 * np.pi * week_of_year / 52)
            feed_price = feed_base + feed_noise + feed_seasonality

            # Supply shocks (5 % chance each week)
            supply_shock = 0.0
            if np.random.random() < 0.05:
                supply_shock = np.random.uniform(-4_000, -2_000)

            supply_index = np.random.uniform(80, 120)
            supply_effect = (100 - supply_index) * 30

            # Weather severity (0 = normal, 1 = moderate, 2 = severe)
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

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("GENERATING CHICKEN PRICE DATASET")
    print("=" * 60)

    df = generate_dataset()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"\nShape:      {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Provinces:  {df['province'].nunique()}")
    print(f"\nPrice statistics:\n{df['chicken_price'].describe()}")
    print(f"\nSaved to: {DATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
