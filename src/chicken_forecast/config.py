"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved from the repo root so commands work from any cwd)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PIHPS_RAW_DIR = RAW_DIR / "pihps"
FRED_RAW_DIR = RAW_DIR / "fred"
PROCESSED_DIR = DATA_DIR / "processed"
WEEKLY_PANEL_PATH = PROCESSED_DIR / "weekly_panel.csv"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data source: Bank Indonesia PIHPS (Pusat Informasi Harga Pangan Strategis)
# https://www.bi.go.id/hargapangan
# ---------------------------------------------------------------------------
PIHPS_BASE_URL = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga"
PIHPS_START_YEAR = 2018  # earliest year with data on PIHPS

# (series name, price_type_id, commodity id)
#   price_type_id: 1 = traditional market (consumer), 2 = modern market,
#                  3 = wholesaler (pedagang besar), 4 = producer (farm-gate)
PIHPS_SERIES: dict[str, tuple[int, str]] = {
    "chicken_retail": (1, "com_7"),     # Daging Ayam Ras Segar, pasar tradisional  <- TARGET
    "chicken_wholesale": (3, "com_7"),  # Daging Ayam Ras Segar, pedagang besar
    "chicken_producer": (4, "com_7"),   # Daging Ayam Ras Segar, produsen
    "egg_retail": (1, "com_10"),        # Telur Ayam Ras Segar, pasar tradisional
}

# ---------------------------------------------------------------------------
# Data source: FRED (IMF primary commodity prices + exchange rate), monthly
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, str] = {
    "maize_usd_mt": "PMAIZMTUSDM",        # Global maize price, USD/metric ton
    "soymeal_usd_mt": "PSMEAUSDM",        # Global soybean meal price, USD/metric ton
    "usd_idr": "CCUSMA02IDM618N",         # IDR per USD, monthly average
}

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGET_SERIES = "chicken_retail"
NATIONAL = "Semua Provinsi"  # PIHPS label for the national average
HORIZONS = [1, 2, 4, 8]  # forecast horizons in weeks
WEEK_RULE = "W-SUN"  # weeks end on Sunday, labelled by the Sunday date

BACKTEST_WEEKS = 360  # rolling-origin evaluation window: ~7 years of weekly origins (from Dec 2019)
SELECTION_WEEKS = 52  # most recent origins used for feature selection; older ones = untouched hold-out
INTERVAL_COVERAGE = 0.80  # nominal coverage of the prediction intervals
QUANTILES = (0.1, 0.5, 0.9)  # quantile-regression levels (10th/90th = the 80% interval)

XGB_PARAMS = {
    # Pseudo-Huber on log changes: squared loss for normal weeks, linear for
    # the occasional shock, so a few extreme weeks don't dominate the fit.
    "objective": "reg:pseudohubererror",
    "huber_slope": 0.05,
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}
