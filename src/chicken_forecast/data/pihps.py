"""Download daily province-level prices from Bank Indonesia's PIHPS panel.

PIHPS (Pusat Informasi Harga Pangan Strategis Nasional) publishes daily prices
for strategic food commodities across all 34 provinces, at four market levels.
The public web table is backed by a JSON endpoint that returns one row per
province and one column per day; this module turns that into a tidy long table.

Raw downloads are cached as one gzipped CSV per (series, year) under
``data/raw/pihps``. Past years are immutable and only fetched once; the
current year is re-downloaded on every refresh.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import requests

from chicken_forecast.config import (
    PIHPS_BASE_URL,
    PIHPS_RAW_DIR,
    PIHPS_SERIES,
    PIHPS_START_YEAR,
)

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (chicken-price-forecast research client)",
    "Accept": "application/json",
}


def fetch_range(
    price_type_id: int,
    commodity_id: str,
    start: date,
    end: date,
    session: requests.Session | None = None,
    retries: int = 4,
) -> pd.DataFrame:
    """Fetch daily prices for every province between *start* and *end* (inclusive).

    Returns a long DataFrame with columns ``date, province, price``.
    Missing observations ("-" in the source) are dropped.
    """
    params = {
        "price_type_id": price_type_id,
        "comcat_id": commodity_id,
        "province_id": "",
        "regency_id": "",
        "market_id": "",
        "tipe_laporan": 1,  # 1 = daily, 2 = weekly, 3 = monthly
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    sess = session or requests.Session()
    url = f"{PIHPS_BASE_URL}/GetGridDataKomoditas"
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, params=params, headers=_HEADERS, timeout=300)
            resp.raise_for_status()
            rows = resp.json()["data"]
            break
        except (requests.RequestException, ValueError, KeyError) as exc:
            if attempt == retries:
                raise
            wait = 5 * attempt
            log.warning("PIHPS request failed (%s), retry %d in %ds", exc, attempt, wait)
            time.sleep(wait)

    return parse_grid(rows)


def parse_grid(rows: list[dict]) -> pd.DataFrame:
    """Convert the PIHPS grid payload (wide, one column per day) to long format."""
    if not rows:
        return pd.DataFrame(columns=["date", "province", "price"])
    meta = {"no", "name", "level"}
    records = []
    for row in rows:
        province = row["name"].strip()
        for key, value in row.items():
            if key in meta or value in (None, "-", ""):
                continue
            records.append((key, province, value))
    df = pd.DataFrame(records, columns=["date", "province", "price"])
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["price"] = pd.to_numeric(df["price"].str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    return df.sort_values(["province", "date"]).reset_index(drop=True)


def _cache_path(series: str, year: int):
    return PIHPS_RAW_DIR / f"{series}_{year}.csv.gz"


def download_series_year(series: str, year: int, force: bool = False) -> pd.DataFrame:
    """Download one calendar year for one series, using the on-disk cache."""
    path = _cache_path(series, year)
    today = date.today()
    is_current = year >= today.year
    if path.exists() and not force and not is_current:
        return pd.read_csv(path, parse_dates=["date"])

    price_type_id, commodity_id = PIHPS_SERIES[series]
    session = requests.Session()
    # Fetch quarter by quarter: smaller responses are faster and less fragile.
    frames = []
    for q in range(4):
        q_start = date(year, 3 * q + 1, 1)
        if q_start > today:
            break
        q_end = date(year, 3 * q + 3, 31 if q in (0, 3) else 30)
        q_end = min(q_end, today)
        t0 = time.time()
        part = fetch_range(price_type_id, commodity_id, q_start, q_end, session=session)
        log.info(
            "  %s %d Q%d: %d rows (%.0fs)", series, year, q + 1, len(part), time.time() - t0
        )
        frames.append(part)

    df = pd.concat(frames, ignore_index=True) if frames else parse_grid([])
    df = df.drop_duplicates(["date", "province"])
    PIHPS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return df


def download_all(
    series: list[str] | None = None,
    start_year: int = PIHPS_START_YEAR,
    end_year: int | None = None,
    force: bool = False,
    max_workers: int = 4,
) -> None:
    """Download (or refresh) every configured series. Runs series in parallel."""
    series = series or list(PIHPS_SERIES)
    end_year = end_year or date.today().year
    jobs = [(s, y) for s in series for y in range(start_year, end_year + 1)]

    def _run(job):
        s, y = job
        df = download_series_year(s, y, force=force)
        log.info("Saved %s %d (%d rows)", s, y, len(df))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_run, jobs))


def load_series(series: str) -> pd.DataFrame:
    """Load every cached year for *series* as one long DataFrame."""
    files = sorted(PIHPS_RAW_DIR.glob(f"{series}_*.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"No cached PIHPS data for '{series}'. Run `chicken-forecast fetch` first."
        )
    df = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    return df.drop_duplicates(["date", "province"]).sort_values(["province", "date"])
