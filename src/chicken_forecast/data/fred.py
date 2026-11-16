"""Monthly feed-cost drivers from FRED (no API key needed for CSV export).

Broiler feed in Indonesia is roughly 50-60% maize and 20-30% soybean meal, and
soybean meal is almost entirely imported, so global grain prices converted to
rupiah are a reasonable proxy for feed-cost pressure on farmers.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from chicken_forecast.config import FRED_RAW_DIR, FRED_SERIES

log = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def download_all() -> pd.DataFrame:
    """Download every configured FRED series and cache them as one CSV."""
    frames = []
    for name, series_id in FRED_SERIES.items():
        resp = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"])
        df[name] = pd.to_numeric(df[name], errors="coerce")
        frames.append(df.set_index("date"))
        log.info("FRED %s (%s): %d months", name, series_id, len(df))
    out = pd.concat(frames, axis=1).sort_index()
    out = out[out.index >= "2016-01-01"]
    FRED_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FRED_RAW_DIR / "feed_drivers_monthly.csv")
    return out.reset_index()


def load() -> pd.DataFrame:
    path = FRED_RAW_DIR / "feed_drivers_monthly.csv"
    if not path.exists():
        raise FileNotFoundError("No cached FRED data. Run `chicken-forecast fetch` first.")
    return pd.read_csv(path, parse_dates=["date"])
