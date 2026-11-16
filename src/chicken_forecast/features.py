"""Feature engineering shared by training, backtesting and forecasting.

Design rules (the v1 model broke these, which inflated its R² to 0.97):

* Every feature for forecast origin week *t* uses only data observed up to
  and including week *t*. Nothing from the target week leaks in, except
  calendar features, which are known in advance.
* The model predicts the **log change** ``log(p[t+h] / p[t])`` rather than the
  price level. That puts all 34 provinces on a common scale (Papua at
  ~Rp55k/kg and Lampung at ~Rp33k/kg) and lets one global model learn from
  all of them at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chicken_forecast.config import TARGET_SERIES
from chicken_forecast.data.calendar import calendar_features

# Features computed at the forecast origin (week t)
ORIGIN_FEATURES = [
    "ret_1",
    "ret_2",
    "ret_4",
    "ret_8",
    "ret_13",
    "ret_26",
    "ret_52",
    "dev_ma4",
    "dev_ma13",
    "dev_ma52",
    "vol_13",
    "spread_national",
    "spread_national_chg4",
    "national_ret_1",
    "national_ret_4",
    "margin_producer",
    "margin_producer_chg4",
    "producer_ret_1",
    "producer_ret_4",
    "margin_wholesale",
    "wholesale_ret_1",
    "egg_ret_4",
    "maize_chg_3m",
    "maize_dev_2y",
    "soymeal_chg_3m",
    "soymeal_dev_2y",
    "usd_idr_chg_3m",
    "rain_1w",
    "rain_4w_anom",
    "hot_days_4w",
    "tmax_4w_anom",
    "province_code",
    "price_level_rel",
]

# Calendar features evaluated at the target week (t + h) and at the origin
CAL_COLS = [
    "weeks_to_fitri",
    "weeks_to_adha",
    "ramadan_frac",
    "fitri_week",
    "holiday_days",
    "xmas_ny",
    "doy_sin",
    "doy_cos",
]
TARGET_CAL_FEATURES = [f"tgt_{c}" for c in CAL_COLS]
ORIGIN_CAL_FEATURES = ["weeks_to_fitri", "ramadan_frac"]

ALL_FEATURES = ORIGIN_FEATURES + ORIGIN_CAL_FEATURES + TARGET_CAL_FEATURES

# Feature groups, each a hypothesis from the literature (see docs/literature.md).
# `chicken-forecast ablation` measures what each group adds out-of-sample.
FEATURE_GROUPS: dict[str, list[str]] = {
    # Own-price dynamics, volatility clustering (GARCH-type persistence),
    # spatial link to the national market. Always included as the core.
    "core_price": [
        "ret_1", "ret_2", "ret_4", "ret_8", "ret_13", "ret_26", "ret_52",
        "dev_ma4", "dev_ma13", "dev_ma52", "vol_13",
        "spread_national", "spread_national_chg4", "national_ret_1", "national_ret_4",
        "province_code", "price_level_rel",
    ],
    # Ramadan / Lebaran / Idul Adha / holidays / time of year
    "calendar": ORIGIN_CAL_FEATURES + TARGET_CAL_FEATURES,
    # Vertical price transmission farmer -> wholesale -> retail
    "supply_chain": [
        "margin_producer", "margin_producer_chg4", "producer_ret_1", "producer_ret_4",
        "margin_wholesale", "wholesale_ret_1",
    ],
    # Related commodity (same feed base, substitute protein)
    "egg": ["egg_ret_4"],
    # Feed cost: maize, soybean meal, exchange rate
    "feed": ["maize_chg_3m", "maize_dev_2y", "soymeal_chg_3m", "soymeal_dev_2y", "usd_idr_chg_3m"],
    # Heat stress (strong biological evidence) and rainfall (weak evidence)
    "weather_heat": ["hot_days_4w", "tmax_4w_anom"],
    "weather_rain": ["rain_1w", "rain_4w_anom"],
}

# Groups kept in the production model, chosen from the ablation results
# (outputs/ablation.csv, docs/methodology.md):
#   kept:    calendar, supply_chain, weather_heat (each lowers backtest error)
#   dropped: feed (raises error ~8-10%), weather_rain (no effect),
#            egg (mixed: worse at 1w, better at 4w; weak literature support)
SELECTED_GROUPS = ["core_price", "calendar", "supply_chain", "weather_heat"]


def features_for(groups: list[str]) -> list[str]:
    return [f for g in groups for f in FEATURE_GROUPS[g]]


FEATURES = features_for(SELECTED_GROUPS)

# Human-readable names for the dashboard / SHAP explanations
FEATURE_LABELS = {
    "ret_1": "Price change, last 1 week",
    "ret_2": "Price change, last 2 weeks",
    "ret_4": "Price change, last 4 weeks",
    "ret_8": "Price change, last 8 weeks",
    "ret_13": "Price change, last quarter",
    "ret_26": "Price change, last 6 months",
    "ret_52": "Price change, year on year",
    "dev_ma4": "Gap to 4-week average",
    "dev_ma13": "Gap to 13-week average",
    "dev_ma52": "Gap to 52-week average",
    "vol_13": "Recent volatility",
    "spread_national": "Gap to national price",
    "spread_national_chg4": "Change in gap to national price",
    "national_ret_1": "National price change, 1 week",
    "national_ret_4": "National price change, 4 weeks",
    "margin_producer": "Retail-to-producer margin",
    "margin_producer_chg4": "Change in producer margin",
    "producer_ret_1": "Producer price change, 1 week",
    "producer_ret_4": "Producer price change, 4 weeks",
    "margin_wholesale": "Retail-to-wholesale margin",
    "wholesale_ret_1": "Wholesale price change, 1 week",
    "egg_ret_4": "Egg price change, 4 weeks",
    "maize_chg_3m": "Maize (feed) price change, 3 months",
    "maize_dev_2y": "Maize price vs 2-year average",
    "soymeal_chg_3m": "Soybean meal price change, 3 months",
    "soymeal_dev_2y": "Soybean meal vs 2-year average",
    "usd_idr_chg_3m": "Rupiah/USD change, 3 months",
    "rain_1w": "Rainfall last week (mm)",
    "rain_4w_anom": "Rainfall anomaly, last 4 weeks",
    "hot_days_4w": "Hot days (>=33°C), last 4 weeks",
    "tmax_4w_anom": "Temperature anomaly, last 4 weeks",
    "tgt_holiday_days": "Public holidays in target week",
    "province_code": "Province",
    "price_level_rel": "Price level vs province history",
    "weeks_to_fitri": "Weeks to Idul Fitri (now)",
    "ramadan_frac": "Ramadan (now)",
    "tgt_weeks_to_fitri": "Weeks to Idul Fitri (target week)",
    "tgt_weeks_to_adha": "Weeks to Idul Adha (target week)",
    "tgt_ramadan_frac": "Ramadan (target week)",
    "tgt_fitri_week": "Idul Fitri week (target)",
    "tgt_xmas_ny": "Christmas / New Year (target)",
    "tgt_doy_sin": "Time of year (sin)",
    "tgt_doy_cos": "Time of year (cos)",
}


def _log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def origin_features(panel: pd.DataFrame, provinces: list[str]) -> pd.DataFrame:
    """Compute origin-week features for every (province, week) row of *panel*."""
    # Reindex each province onto a complete weekly grid so that row shifts
    # below always mean "k weeks ago", even when the source has gaps.
    grids = []
    for prov, sub in panel.groupby("province", sort=False):
        full = pd.date_range(sub["week"].min(), sub["week"].max(), freq="7D")
        grids.append(sub.set_index("week").reindex(full).rename_axis("week").assign(province=prov))
    df = pd.concat(grids).reset_index().sort_values(["province", "week"])
    # National/feed columns are shared across provinces: refill them by week
    shared = panel.drop_duplicates("week").set_index("week")
    for col in ["national_retail", "maize_idr_kg", "soymeal_idr_kg", "usd_idr"]:
        df[col] = df["week"].map(shared[col])
    df = df.reset_index(drop=True)
    lp = _log(df[TARGET_SERIES])
    df["_lp"] = lp

    def by_prov(col):
        return df.groupby("province", sort=False)[col]

    for k in [1, 2, 4, 8, 13, 26, 52]:
        df[f"ret_{k}"] = lp - by_prov("_lp").shift(k)

    for w in [4, 13, 52]:
        ma = by_prov("_lp").transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 2)).mean())
        df[f"dev_ma{w}"] = lp - ma

    df["vol_13"] = by_prov("ret_1").transform(lambda s: s.rolling(13, min_periods=6).std())

    lnat = _log(df["national_retail"])
    df["spread_national"] = lp - lnat
    df["spread_national_chg4"] = df["spread_national"] - by_prov("spread_national").shift(4)
    df["_lnat"] = lnat
    df["national_ret_1"] = lnat - by_prov("_lnat").shift(1)
    df["national_ret_4"] = lnat - by_prov("_lnat").shift(4)

    lprod = _log(df["chicken_producer"])
    df["_lprod"] = lprod
    df["margin_producer"] = lp - lprod
    df["margin_producer_chg4"] = df["margin_producer"] - by_prov("margin_producer").shift(4)
    df["producer_ret_1"] = lprod - by_prov("_lprod").shift(1)
    df["producer_ret_4"] = lprod - by_prov("_lprod").shift(4)

    lwh = _log(df["chicken_wholesale"])
    df["_lwh"] = lwh
    df["margin_wholesale"] = lp - lwh
    df["wholesale_ret_1"] = lwh - by_prov("_lwh").shift(1)

    df["_legg"] = _log(df["egg_retail"])
    df["egg_ret_4"] = df["_legg"] - by_prov("_legg").shift(4)

    # Feed drivers are national (same for every province), so compute on weeks
    feed = (
        df[["week", "maize_idr_kg", "soymeal_idr_kg", "usd_idr"]]
        .drop_duplicates("week")
        .set_index("week")
        .sort_index()
    )
    feed = np.log(feed)
    f = pd.DataFrame(index=feed.index)
    f["maize_chg_3m"] = feed["maize_idr_kg"] - feed["maize_idr_kg"].shift(13)
    f["maize_dev_2y"] = feed["maize_idr_kg"] - feed["maize_idr_kg"].rolling(104, min_periods=26).mean()
    f["soymeal_chg_3m"] = feed["soymeal_idr_kg"] - feed["soymeal_idr_kg"].shift(13)
    f["soymeal_dev_2y"] = (
        feed["soymeal_idr_kg"] - feed["soymeal_idr_kg"].rolling(104, min_periods=26).mean()
    )
    f["usd_idr_chg_3m"] = feed["usd_idr"] - feed["usd_idr"].shift(13)
    df = df.merge(f.reset_index(), on="week", how="left")

    # Weather (province-specific). Anomalies are measured against the
    # province's own expanding history so they never use future data.
    def expanding_mean(col):
        return by_prov(col).transform(lambda s: s.expanding(min_periods=26).mean())

    df["rain_1w"] = df["rain_mm"]
    df["_rain4"] = by_prov("rain_mm").transform(lambda s: s.rolling(4, min_periods=3).sum())
    df["rain_4w_anom"] = np.log1p(df["_rain4"]) - np.log1p(expanding_mean("_rain4"))
    df["hot_days_4w"] = by_prov("hot_days").transform(lambda s: s.rolling(4, min_periods=3).sum())
    df["_tmax4"] = by_prov("tmax_c").transform(lambda s: s.rolling(4, min_periods=3).mean())
    df["tmax_4w_anom"] = df["_tmax4"] - expanding_mean("_tmax4")

    # Expanding mean so the level feature never peeks at the future
    df["price_level_rel"] = df["_lp"] - df.groupby("province", sort=False)["_lp"].transform(
        lambda s: s.expanding(min_periods=4).mean()
    )
    code = {p: i for i, p in enumerate(sorted(provinces))}
    df["province_code"] = df["province"].map(code).astype("int32")

    cal = calendar_features(df["week"])
    for c in ORIGIN_CAL_FEATURES:
        df[c] = cal[c].values

    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def add_target_calendar(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach calendar features of the target week (origin + horizon weeks)."""
    target_week = df["week"] + pd.to_timedelta(7 * horizon, unit="D")
    cal = calendar_features(target_week)
    out = df.copy()
    for c in CAL_COLS:
        out[f"tgt_{c}"] = cal[c].values
    out["target_week"] = target_week.values
    return out


def make_supervised(feats: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rows usable for training at *horizon*: features at t, target log change to t+h.

    Uses an explicit week join (not a row shift) so gaps in a province's
    history can never pair a feature row with the wrong target week.
    """
    df = add_target_calendar(feats, horizon)
    future = feats[["province", "week", TARGET_SERIES]].rename(
        columns={"week": "target_week", TARGET_SERIES: "target_price"}
    )
    df = df.merge(future, on=["province", "target_week"], how="left")
    df["y"] = np.log(df["target_price"]) - np.log(df[TARGET_SERIES])
    df["horizon"] = horizon
    return df
