"""Turn raw daily PIHPS + monthly FRED data into a clean weekly province panel."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from chicken_forecast.config import (
    NATIONAL,
    PIHPS_SERIES,
    PROCESSED_DIR,
    TARGET_SERIES,
    WEEK_RULE,
    WEEKLY_PANEL_PATH,
)
from chicken_forecast.data import fred, pihps, weather

log = logging.getLogger(__name__)

# A daily price further than this (in log terms, ~±25%) from its centred
# 15-day rolling median is treated as a data-entry error and dropped.
OUTLIER_LOG_DEV = 0.22
MAX_GAP_WEEKS = 3


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Drop isolated spikes/typos in a long daily price table."""
    df = df.sort_values(["province", "date"]).copy()
    logp = np.log(df["price"])
    med = logp.groupby(df["province"]).transform(
        lambda s: s.rolling(15, center=True, min_periods=5).median()
    )
    bad = (logp - med).abs() > OUTLIER_LOG_DEV
    if bad.any():
        log.info("  removed %d daily outliers", int(bad.sum()))
    return df[~bad]


def to_weekly(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Average daily prices into Sunday-ending weeks; fill short gaps."""
    weekly = (
        df.set_index("date")
        .groupby("province")["price"]
        .resample(WEEK_RULE)
        .mean()
        .rename(name)
        .reset_index()
        .rename(columns={"date": "week"})
    )
    weekly[name] = weekly.groupby("province")[name].transform(
        lambda s: s.interpolate(limit=MAX_GAP_WEEKS, limit_area="inside")
    )
    return weekly


def feed_drivers_weekly(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly feed-cost drivers in IDR, aligned to weeks *as they were known*.

    FRED publishes month M early in month M+1, so the value for month M is
    only used from the first week of month M+2 onwards (a conservative
    one-month publication lag) to avoid look-ahead bias.
    """
    m = fred.load().set_index("date").sort_index()
    out = pd.DataFrame(index=m.index)
    out["maize_idr_kg"] = m["maize_usd_mt"] * m["usd_idr"] / 1000
    out["soymeal_idr_kg"] = m["soymeal_usd_mt"] * m["usd_idr"] / 1000
    out["usd_idr"] = m["usd_idr"]
    out.index = out.index + pd.DateOffset(months=2)  # availability date
    out = out.dropna(how="all")
    aligned = out.reindex(out.index.union(weeks)).sort_index().ffill().reindex(weeks)
    aligned.index.name = "week"
    return aligned.reset_index()


def weather_weekly() -> pd.DataFrame:
    """Weekly rainfall total, mean max temperature and hot days per province."""
    w = weather.load_daily()
    w["hot_day"] = (w["tmax_c"] >= 33).astype(float)
    weekly = (
        w.set_index("date")
        .groupby("province")
        .resample(WEEK_RULE)
        .agg({"precip_mm": "sum", "tmax_c": "mean", "hot_day": "sum"})
        .reset_index()
        .rename(columns={"date": "week", "precip_mm": "rain_mm", "hot_day": "hot_days"})
    )
    return weekly


def build_weekly_panel(save: bool = True) -> pd.DataFrame:
    """Build the modelling panel: one row per (province, week)."""
    panel = None
    for name in PIHPS_SERIES:
        log.info("Processing %s", name)
        daily = clean_daily(pihps.load_series(name))
        weekly = to_weekly(daily, name)
        panel = weekly if panel is None else panel.merge(weekly, on=["province", "week"], how="outer")

    # Drop the current, incomplete week (PIHPS publishes on working days).
    last_full_week = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    last_full_week = last_full_week - pd.Timedelta(days=(last_full_week.dayofweek + 1) % 7)
    panel = panel[panel["week"] <= last_full_week]

    # National average of the target series as a separate column
    nat = panel.loc[panel["province"] == NATIONAL, ["week", TARGET_SERIES]].rename(
        columns={TARGET_SERIES: "national_retail"}
    )
    panel = panel[panel["province"] != NATIONAL].merge(nat, on="week", how="left")

    weeks = pd.DatetimeIndex(sorted(panel["week"].unique()))
    panel = panel.merge(feed_drivers_weekly(weeks), on="week", how="left")
    panel = panel.merge(weather_weekly(), on=["province", "week"], how="left")

    panel = panel.dropna(subset=[TARGET_SERIES])
    panel = panel.sort_values(["province", "week"]).reset_index(drop=True)
    log.info(
        "Weekly panel: %d rows, %d provinces, %s to %s",
        len(panel),
        panel["province"].nunique(),
        panel["week"].min().date(),
        panel["week"].max().date(),
    )
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_csv(WEEKLY_PANEL_PATH, index=False, float_format="%.2f")
    return panel


def load_weekly_panel() -> pd.DataFrame:
    if not WEEKLY_PANEL_PATH.exists():
        raise FileNotFoundError("No weekly panel. Run `chicken-forecast build` first.")
    return pd.read_csv(WEEKLY_PANEL_PATH, parse_dates=["week"])
