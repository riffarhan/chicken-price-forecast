"""Indonesian holiday calendar, downloaded from public sources.

Chicken demand in Indonesia peaks before Idul Fitri (Lebaran) and, to a lesser
degree, Idul Adha, Christmas and New Year. Islamic holidays follow the Hijri
calendar and move ~11 days earlier every year, so month/quarter dummies can't
capture them. Nothing here is typed in by hand:

* **Aladhan API** (api.aladhan.com): Hijri date for every Gregorian day,
  2017-2028. Gives Ramadan (month 9), Idul Fitri (1 Syawal) and Idul Adha
  (10 Dzulhijjah) consistently across the whole history.
* **Nager.Date API** (date.nager.at): Indonesian fixed-date public holidays
  (New Year, Christmas, Independence Day, ...).
* **Google Calendar "Holidays in Indonesia"** (public ICS feed): the official
  holiday + cuti bersama list for 2021 onward. Used to *validate* the Aladhan
  dates. Indonesia's government sets Islamic holidays by moon sighting, so
  an astronomical calendar can be off by a day; ``validate()`` reports this.

Every feature produced here is known in advance, so it can be computed for
the *target* week of a forecast without leaking future information.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
import requests

from chicken_forecast.config import RAW_DIR

log = logging.getLogger(__name__)

CAL_DIR = RAW_DIR / "calendar"
HIJRI_PATH = CAL_DIR / "hijri_daily.csv"
NAGER_PATH = CAL_DIR / "nager_public_holidays_id.csv"
GOOGLE_PATH = CAL_DIR / "google_holidays_id.csv"

CAL_START_YEAR, CAL_END_YEAR = 2017, 2028

ALADHAN_URL = "https://api.aladhan.com/v1/gToHCalendar/{month}/{year}"
NAGER_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/ID"
GOOGLE_ICS_URL = (
    "https://calendar.google.com/calendar/ical/"
    "en.indonesian%23holiday%40group.v.calendar.google.com/public/basic.ics"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_hijri() -> pd.DataFrame:
    rows = []
    sess = requests.Session()
    for year in range(CAL_START_YEAR, CAL_END_YEAR + 1):
        for month in range(1, 13):
            for attempt in range(4):
                try:
                    r = sess.get(ALADHAN_URL.format(month=month, year=year), timeout=30)
                    r.raise_for_status()
                    days = r.json()["data"]
                    break
                except (requests.RequestException, ValueError, KeyError):
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))
            for d in days:
                g, h = d["gregorian"], d["hijri"]
                rows.append(
                    {
                        "date": pd.to_datetime(g["date"], format="%d-%m-%Y"),
                        "hijri_day": int(h["day"]),
                        "hijri_month": int(h["month"]["number"]),
                        "hijri_year": int(h["year"]),
                    }
                )
            time.sleep(0.15)  # be polite to a free API
        log.info("  Aladhan Hijri calendar %d done", year)
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HIJRI_PATH, index=False)
    return df


def download_nager() -> pd.DataFrame:
    rows = []
    for year in range(CAL_START_YEAR, CAL_END_YEAR + 1):
        r = requests.get(NAGER_URL.format(year=year), timeout=30)
        r.raise_for_status()
        for h in r.json():
            rows.append({"date": h["date"], "name": h["name"], "local_name": h["localName"]})
    df = pd.DataFrame(rows)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(NAGER_PATH, index=False)
    log.info("  Nager.Date: %d Indonesian public holidays", len(df))
    return df


def download_google() -> pd.DataFrame:
    r = requests.get(GOOGLE_ICS_URL, timeout=60)
    r.raise_for_status()
    rows = []
    for block in r.text.split("BEGIN:VEVENT")[1:]:
        start = re.search(r"DTSTART;VALUE=DATE:(\d{8})", block)
        summary = re.search(r"SUMMARY:(.+)", block)
        desc = re.search(r"DESCRIPTION:(.+)", block)
        if start and summary:
            rows.append(
                {
                    "date": pd.to_datetime(start.group(1), format="%Y%m%d"),
                    "name": summary.group(1).strip(),
                    "type": desc.group(1).strip() if desc else "",
                }
            )
    df = pd.DataFrame(rows).sort_values("date")
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOOGLE_PATH, index=False)
    log.info("  Google Calendar: %d Indonesian holiday entries", len(df))
    return df


def download_all() -> None:
    download_nager()
    download_google()
    download_hijri()


# ---------------------------------------------------------------------------
# Derived event dates
# ---------------------------------------------------------------------------
def _hijri() -> pd.DataFrame:
    if not HIJRI_PATH.exists():
        raise FileNotFoundError("No calendar data. Run `chicken-forecast fetch` first.")
    return pd.read_csv(HIJRI_PATH, parse_dates=["date"])


def _official(pattern: str) -> pd.DatetimeIndex:
    if not GOOGLE_PATH.exists():
        return pd.DatetimeIndex([])
    g = pd.read_csv(GOOGLE_PATH, parse_dates=["date"])
    return pd.DatetimeIndex(g.loc[g["name"].str.match(pattern), "date"])


def _prefer_official(computed: pd.DatetimeIndex, official: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Replace each computed date with the official one when one exists within 3 days."""
    out = []
    for d in computed:
        near = official[np.abs((official - d).days) <= 3]
        out.append(near[0] if len(near) else d)
    return pd.DatetimeIndex(out)


def islamic_events() -> dict[str, pd.DatetimeIndex]:
    """Idul Fitri, Idul Adha and Ramadan days.

    Uses the official Indonesian date (Google Calendar, 2021+) where available,
    and the Aladhan astronomical date otherwise. Ramadan is shifted with Idul
    Fitri so it always ends the day before the (official) Lebaran.
    """
    h = _hijri()
    fitri_calc = pd.DatetimeIndex(h.loc[(h.hijri_month == 10) & (h.hijri_day == 1), "date"])
    adha_calc = pd.DatetimeIndex(h.loc[(h.hijri_month == 12) & (h.hijri_day == 10), "date"])
    fitri = _prefer_official(fitri_calc, _official(r"^Idul Fitri$"))
    adha = _prefer_official(adha_calc, _official(r"^Idul Adha$"))

    ramadan = []
    for calc, final in zip(fitri_calc, fitri):
        shift = final - calc
        days = h.loc[(h.hijri_month == 9) & (h.date < calc) & (h.date > calc - pd.Timedelta(days=35)), "date"]
        ramadan.extend(days + shift)
    return {"idul_fitri": fitri, "idul_adha": adha, "ramadan_days": pd.DatetimeIndex(ramadan)}


def islamic_events_aladhan_only() -> dict[str, pd.DatetimeIndex]:
    h = _hijri()
    return {
        "idul_fitri": pd.DatetimeIndex(h.loc[(h.hijri_month == 10) & (h.hijri_day == 1), "date"]),
        "idul_adha": pd.DatetimeIndex(h.loc[(h.hijri_month == 12) & (h.hijri_day == 10), "date"]),
    }


def public_holiday_days() -> pd.DatetimeIndex:
    """All national holiday dates: Nager fixed holidays + Aladhan Eid days."""
    nager = pd.to_datetime(pd.read_csv(NAGER_PATH)["date"])
    ev = islamic_events()
    eid = ev["idul_fitri"].append(ev["idul_fitri"] + pd.Timedelta(days=1)).append(ev["idul_adha"])
    return pd.DatetimeIndex(sorted(set(nager).union(eid)))


def validate() -> pd.DataFrame:
    """Compare Aladhan Eid dates with Google's official Indonesian calendar."""
    g = pd.read_csv(GOOGLE_PATH, parse_dates=["date"])
    ev = islamic_events_aladhan_only()
    out = []
    for name, pattern in [("idul_fitri", r"^Idul Fitri$"), ("idul_adha", r"^Idul Adha$")]:
        official = g.loc[g["name"].str.match(pattern), "date"]
        for d in official:
            nearest = ev[name][np.abs((ev[name] - d).days).argmin()]
            out.append({"event": name, "official": d.date(), "aladhan": nearest.date(),
                        "diff_days": (nearest - d).days})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Weekly features
# ---------------------------------------------------------------------------
def _signed_weeks_to_nearest(dates: pd.DatetimeIndex, events: pd.DatetimeIndex) -> np.ndarray:
    """Weeks from each date to the nearest event (negative = event still ahead)."""
    d = dates.values.astype("datetime64[D]").astype(np.int64)[:, None]
    e = events.values.astype("datetime64[D]").astype(np.int64)[None, :]
    diff = d - e
    idx = np.abs(diff).argmin(axis=1)
    return diff[np.arange(len(d)), idx] / 7.0


def _days_in_week(week_end: pd.DatetimeIndex, days: pd.DatetimeIndex) -> np.ndarray:
    """How many of *days* fall in each Monday-Sunday week ending on week_end."""
    s = pd.Series(1, index=pd.DatetimeIndex(days))
    counts = s.resample("W-SUN").sum()
    return counts.reindex(week_end, fill_value=0).to_numpy()


def calendar_features(week_end: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Holiday/seasonal features for weeks labelled by their (Sunday) end date."""
    week_end = pd.DatetimeIndex(week_end)
    uniq = pd.DatetimeIndex(week_end.unique())
    ev = islamic_events()

    weeks_to_fitri = _signed_weeks_to_nearest(uniq, ev["idul_fitri"])
    weeks_to_adha = _signed_weeks_to_nearest(uniq, ev["idul_adha"])
    doy = uniq.dayofyear.values
    # Christmas / New Year window: Dec 18 - Jan 7
    xmas_ny = ((uniq.month == 12) & (uniq.day >= 18)) | ((uniq.month == 1) & (uniq.day <= 7))

    feats = pd.DataFrame(
        {
            "weeks_to_fitri": np.clip(weeks_to_fitri, -12, 12),
            "weeks_to_adha": np.clip(weeks_to_adha, -8, 8),
            "ramadan_frac": _days_in_week(uniq, ev["ramadan_days"]) / 7.0,
            "fitri_week": _days_in_week(uniq, ev["idul_fitri"]),
            "holiday_days": _days_in_week(uniq, public_holiday_days()),
            "xmas_ny": xmas_ny.astype(int),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=uniq,
    )
    return feats.reindex(week_end)
