"""Tests for parsing, calendar, and leakage-free feature engineering."""

import numpy as np
import pandas as pd
import pytest

from chicken_forecast.data.calendar import calendar_features
from chicken_forecast.data.pihps import parse_grid
from chicken_forecast.features import FEATURES, make_supervised, origin_features


def _toy_panel(n_weeks=80, provinces=("Aceh", "Bali")):
    weeks = pd.date_range("2022-01-02", periods=n_weeks, freq="7D")
    rows = []
    rng = np.random.default_rng(0)
    for i, p in enumerate(provinces):
        price = 35000 + 1000 * i + np.cumsum(rng.normal(0, 300, n_weeks))
        for w, v in zip(weeks, price):
            rows.append(
                {
                    "province": p,
                    "week": w,
                    "chicken_retail": v,
                    "chicken_wholesale": v * 0.9,
                    "chicken_producer": v * 0.75,
                    "egg_retail": 27000.0,
                    "national_retail": 36000.0,
                    "maize_idr_kg": 3500.0,
                    "soymeal_idr_kg": 7000.0,
                    "usd_idr": 15500.0,
                    "rain_mm": float(rng.gamma(2, 20)),
                    "tmax_c": 31.0,
                    "hot_days": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_parse_grid_handles_missing_and_thousands():
    rows = [
        {"no": "I", "name": "Aceh", "level": 1, "01/01/2024": "36,950", "02/01/2024": "-"},
        {"no": "II", "name": " Bali ", "level": 1, "01/01/2024": "40,100", "02/01/2024": "40,200"},
    ]
    df = parse_grid(rows)
    assert len(df) == 3
    assert df["province"].tolist() == ["Aceh", "Bali", "Bali"]
    assert df.loc[df["province"] == "Aceh", "price"].iloc[0] == 36950
    assert df["date"].min() == pd.Timestamp("2024-01-01")


def test_calendar_marks_ramadan_and_fitri():
    # Uses the committed Aladhan/Nager snapshot in data/raw/calendar.
    # Idul Fitri 2024 = 10 April; the week ending Sun 14 April contains it
    cal = calendar_features(pd.to_datetime(["2024-03-24", "2024-04-14", "2024-08-04"]))
    assert cal["ramadan_frac"].iloc[0] == 1.0
    assert cal["fitri_week"].iloc[1] == 1
    assert cal["ramadan_frac"].iloc[2] == 0.0
    assert cal["weeks_to_fitri"].iloc[0] < 0  # Lebaran still ahead
    assert cal["holiday_days"].iloc[1] >= 2  # Idul Fitri day 1 + 2


def test_aladhan_matches_official_calendar_within_a_day():
    from chicken_forecast.data.calendar import validate

    v = validate()
    assert len(v) >= 10
    assert v["diff_days"].abs().max() <= 1


def test_features_only_use_the_past():
    panel = _toy_panel()
    feats = origin_features(panel, ["Aceh", "Bali"])
    # Corrupt everything after a cutoff; features up to the cutoff must not change
    cutoff = panel["week"].iloc[50]
    corrupted = panel.copy()
    corrupted.loc[corrupted["week"] > cutoff, "chicken_retail"] *= 3
    feats2 = origin_features(corrupted, ["Aceh", "Bali"])
    cols = [c for c in FEATURES if c in feats.columns]
    a = feats[feats["week"] <= cutoff][cols].reset_index(drop=True)
    b = feats2[feats2["week"] <= cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_supervised_target_matches_future_price():
    panel = _toy_panel()
    feats = origin_features(panel, ["Aceh", "Bali"])
    sup = make_supervised(feats, horizon=4)
    row = sup[(sup["province"] == "Aceh")].iloc[10]
    future = panel[(panel["province"] == "Aceh") & (panel["week"] == row["week"] + pd.Timedelta(weeks=4))]
    assert row["target_price"] == pytest.approx(future["chicken_retail"].iloc[0])
    assert row["y"] == pytest.approx(np.log(row["target_price"] / row["chicken_retail"]))


def test_gaps_do_not_misalign_lags():
    panel = _toy_panel()
    # Remove 5 weeks for Aceh: the lag must still mean "1 calendar week ago"
    gap = panel["week"].iloc[30:35]
    panel = panel[~((panel["province"] == "Aceh") & panel["week"].isin(gap))]
    feats = origin_features(panel, ["Aceh", "Bali"])
    after_gap = feats[(feats["province"] == "Aceh") & (feats["week"] == gap.iloc[-1] + pd.Timedelta(weeks=1))]
    assert np.isnan(after_gap["ret_1"].iloc[0])
