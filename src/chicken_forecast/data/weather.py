"""Daily weather per province from the Open-Meteo historical archive (ERA5).

Why weather matters for broiler prices:

* **Heat stress**: broilers eat less and grow slower above ~32°C, which
  cuts supply weeks later.
* **Heavy rain / floods**: disrupt transport from farms to markets and raise
  disease pressure, especially outside Java where supply chains are long.

Weather is taken at each province's capital city. Coordinates come from the
Open-Meteo geocoding API (we only supply the city name) and are cached in
``data/raw/weather/locations.csv`` so the lookup is auditable.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from chicken_forecast.config import RAW_DIR

log = logging.getLogger(__name__)

WEATHER_DIR = RAW_DIR / "weather"
LOCATIONS_PATH = WEATHER_DIR / "locations.csv"
DAILY_PATH = WEATHER_DIR / "weather_daily.csv.gz"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2017-10-01"

# PIHPS province name -> capital / main market city searched in the geocoder.
# Kalimantan Selatan uses Banjarmasin (main market; capital moved to Banjarbaru
# in 2022), Maluku Utara uses Ternate (main market; capital is Sofifi).
PROVINCE_CITY = {
    "Aceh": "Banda Aceh",
    "Sumatera Utara": "Medan",
    "Sumatera Barat": "Padang",
    "Riau": "Pekanbaru",
    "Kepulauan Riau": "Tanjung Pinang",
    "Jambi": "Jambi",
    "Bengkulu": "Bengkulu",
    "Sumatera Selatan": "Palembang",
    "Kepulauan Bangka Belitung": "Pangkalpinang",
    "Lampung": "Bandar Lampung",
    "Banten": "Serang",
    "DKI Jakarta": "Jakarta",
    "Jawa Barat": "Bandung",
    "Jawa Tengah": "Semarang",
    "DI Yogyakarta": "Yogyakarta",
    "Jawa Timur": "Surabaya",
    "Bali": "Denpasar",
    "Nusa Tenggara Barat": "Mataram",
    "Nusa Tenggara Timur": "Kupang",
    "Kalimantan Barat": "Pontianak",
    "Kalimantan Tengah": "Palangkaraya",
    "Kalimantan Selatan": "Banjarmasin",
    "Kalimantan Timur": "Samarinda",
    "Kalimantan Utara": "Tanjung Selor",
    "Sulawesi Utara": "Manado",
    "Gorontalo": "Gorontalo",
    "Sulawesi Tengah": "Palu",
    "Sulawesi Barat": "Mamuju",
    "Sulawesi Selatan": "Makassar",
    "Sulawesi Tenggara": "Kendari",
    "Maluku": "Ambon",
    "Maluku Utara": "Ternate",
    "Papua": "Jayapura",
    "Papua Barat": "Manokwari",
}


def geocode_all() -> pd.DataFrame:
    rows = []
    for prov, city in PROVINCE_CITY.items():
        r = requests.get(
            GEOCODE_URL, params={"name": city, "count": 10, "country_code": "ID"}, timeout=30
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        # Prefer populated places (PPL*) over airports etc., biggest first
        places = [x for x in results if str(x.get("feature_code", "")).startswith("PPL")]
        if not places:
            raise RuntimeError(f"Geocoder found no populated place for {city} ({prov})")
        best = max(places, key=lambda x: (x.get("feature_code") in ("PPLA", "PPLC"), x.get("population") or 0))
        rows.append(
            {
                "province": prov,
                "city": city,
                "geocoded_name": best["name"],
                "admin1": best.get("admin1", ""),
                "latitude": best["latitude"],
                "longitude": best["longitude"],
                "feature_code": best.get("feature_code"),
            }
        )
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCATIONS_PATH, index=False)
    log.info("  geocoded %d province capitals", len(df))
    return df


def download_all() -> pd.DataFrame:
    locs = geocode_all()
    end = (date.today() - timedelta(days=2)).isoformat()  # archive lags a few days
    frames = []
    for _, loc in locs.iterrows():
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "start_date": START_DATE,
            "end_date": end,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_mean",
            "timezone": "Asia/Jakarta",
        }
        for attempt in range(5):
            r = requests.get(ARCHIVE_URL, params=params, timeout=120)
            if r.status_code == 429:  # rate limited: back off
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        d = r.json()["daily"]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(d["time"]),
                "province": loc.province,
                "precip_mm": d["precipitation_sum"],
                "tmax_c": d["temperature_2m_max"],
                "tmean_c": d["temperature_2m_mean"],
            }
        )
        frames.append(df)
        log.info("  weather %s (%s): %d days", loc.province, loc.city, len(df))
        time.sleep(1.0)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DAILY_PATH, index=False, compression="gzip", float_format="%.1f")
    return out


def load_daily() -> pd.DataFrame:
    if not DAILY_PATH.exists():
        raise FileNotFoundError("No weather data. Run `chicken-forecast fetch` first.")
    return pd.read_csv(DAILY_PATH, parse_dates=["date"])
