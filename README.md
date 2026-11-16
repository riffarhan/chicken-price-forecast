# Chicken Price Forecast · Indonesia

Weekly forecasts of **retail broiler chicken prices for all 34 Indonesian provinces**, built entirely on **real, public data**, with features chosen from **peer-reviewed literature** and kept only if they **improve out-of-sample accuracy**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Data: Bank Indonesia PIHPS](https://img.shields.io/badge/data-BI%20PIHPS-orange)

> **v2.0** replaces the original synthetic-data prototype. The old model's R² of 0.97 came from synthetic data and features that leaked the target. This version uses ~78,000 daily price observations per series from Bank Indonesia, a leak-free feature pipeline, and a rolling-origin backtest against honest baselines.

## Results at a glance

Rolling-origin backtest: **156 weekly forecast origins × 34 provinces** (origins Oct 2023 – Nov 2026), retrained every 13 weeks using only data available at the time.

| Horizon | Model MAPE | Naive MAPE | Error vs naive | DM p-value | Direction correct | 80% interval coverage |
|---|---|---|---|---|---|---|
| 1 week | **2.29%** | 2.42% | **−4.9%** | **< 0.001** | 59% | 78% |
| 2 weeks | 3.68% | 3.82% | −3.1% | 0.14 | 59% | 77% |
| 4 weeks | 5.07% | 5.34% | −4.4% | 0.31 | 62% | 76% |
| 8 weeks | 6.65% | 7.00% | −4.4% | 0.54 | 63% | 73% |

**Untouched hold-out** (Oct 2023 – Nov 2025, never used for any feature or parameter decision):

| Horizon | Error vs naive | DM p-value |
|---|---|---|
| 1 week | **−4.7%** | **0.003** |
| 2 weeks | **−3.9%** | **0.046** |
| 4 weeks | −3.8% | 0.25 |
| 8 weeks | +2.5% | 0.72 |

What this means:

- Weekly chicken prices are **close to a random walk**. "Price stays the same" is a hard baseline to beat, and the old v1 claim of R² = 0.97 was not realistic.
- The model **reliably beats naive at 1–2 weeks** (significant on the untouched hold-out). At 4 weeks it is better but not significantly so, and at 8 weeks it is no better than naive on older data. The dashboard shows a reliability notice for each horizon.
- Beyond the point forecast, the model gives **calibrated 80% intervals** (76–78% realised coverage up to 4 weeks), **59–63% direction accuracy**, and **SHAP explanations** of what drives each province's forecast.

Full details are in [docs/methodology.md](docs/methodology.md).

## Data (all downloaded from public sources)

| Data | Source | Used for |
|---|---|---|
| Daily chicken **retail, wholesale and producer** prices; egg prices; 34 provinces, 2018 → | [Bank Indonesia PIHPS](https://www.bi.go.id/hargapangan) | Target and supply-chain signals |
| Global **maize & soybean meal** prices, **IDR/USD** | [FRED](https://fred.stlouisfed.org/series/PMAIZMTUSDM) / IMF | Feed-cost hypothesis (tested, dropped) |
| **Ramadan, Idul Fitri, Idul Adha** | [Aladhan Hijri calendar API](https://aladhan.com/islamic-calendar-api), with official dates from Google's public Indonesian holiday calendar | Demand seasonality |
| National **public holidays** | [Nager.Date](https://date.nager.at) | Holiday weeks |
| Daily **rainfall & temperature** at each province capital | [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) ERA5 reanalysis | Heat-stress hypothesis |

Nothing is synthetic or hand-typed. Raw snapshots are committed under `data/raw/` for reproducibility.

## Why these features? Literature + ablation

Each feature group is a hypothesis from published research ([docs/literature.md](docs/literature.md)) and was then tested by leave-one-group-out ablation with Diebold–Mariano significance tests:

| Feature group | Literature | Kept? |
|---|---|---|
| Own-price dynamics & volatility | GARCH volatility clustering in weekly PIHPS broiler prices (Meilany et al., 2025) | ✅ core |
| Ramadan / Lebaran / holidays | Price spikes around Ramadan & Idul Fitri (Meilany et al., 2025; Permatasari et al., 2023) | ✅ |
| Producer → wholesale → retail margins | Asymmetric price transmission, NARDL (Jojo et al., 2021) | ✅ |
| Heat stress | Feed intake & growth loss in heat-stressed broilers (Liu et al., 2020; Teyssier et al., 2022) | ✅ |
| Global feed prices | Feed ≈ 66% of farm cost (Sehabudin et al., 2022); feed price significant in 34-province panel (Fitrimawati et al., 2024) | ❌ hurt weekly accuracy |
| Rainfall | Evidence only for rice prices (Putra et al., 2021), not chicken | ❌ no effect |
| Egg prices | Partial | ❌ mixed |

## Quick start

```bash
git clone https://github.com/riffarhan/chicken-price-forecast.git
cd chicken-price-forecast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# The repo already contains data + outputs, so the dashboard works right away:
streamlit run app/dashboard.py
```

Refresh everything from the live sources:

```bash
chicken-forecast fetch      # download PIHPS, FRED, calendars, weather (≈5 min)
chicken-forecast build      # daily -> clean weekly panel
chicken-forecast train      # backtest, train, forecast, SHAP explanations
chicken-forecast ablation   # re-run the feature-group experiment (≈15 min)
# or: chicken-forecast all  (fetch + build + train)
```

Run it weekly with cron (Monday 06:00):

```bash
0 6 * * 1 /path/to/chicken-price-forecast/scripts/run_weekly.sh >> /path/to/chicken-price-forecast/logs/cron.log 2>&1
```

## Dashboard

`streamlit run app/dashboard.py` opens five tabs:

- **Forecast**: price history, backtest track record and the 1–8 week forecast with an 80% interval for any province; retail vs wholesale vs farm-gate prices
- **All provinces**: map and ranking of expected price changes
- **Why?**: SHAP drivers behind each province's forecast, plus global feature importance
- **Accuracy**: backtest vs naive baselines by horizon, significance tests, per-province error
- **Data**: sources, feed and rainfall series, CSV downloads

## Project structure

```
chicken-price-forecast/
├── app/
│   └── dashboard.py              # Streamlit dashboard
├── src/chicken_forecast/
│   ├── config.py                 # paths, sources, horizons, model params
│   ├── data/
│   │   ├── pihps.py              # Bank Indonesia PIHPS downloader
│   │   ├── fred.py               # FRED/IMF feed-cost drivers
│   │   ├── calendar.py           # Hijri + official holiday calendars
│   │   ├── weather.py            # Open-Meteo geocoding + ERA5 archive
│   │   └── build.py              # cleaning + weekly panel
│   ├── features.py               # leak-free features, feature groups
│   ├── models.py                 # XGBoost, baselines, backtest, DM test, intervals
│   ├── ablation.py               # feature-group experiment
│   ├── pipeline.py               # train -> forecast -> SHAP -> outputs
│   └── cli.py                    # `chicken-forecast` command
├── data/
│   ├── raw/                      # cached source snapshots (committed)
│   └── processed/weekly_panel.csv
├── outputs/                      # forecasts, backtest, metrics, SHAP, ablation
├── models/                       # trained models (rebuilt by `train`, not committed)
├── docs/
│   ├── literature.md             # verified citations behind every feature
│   └── methodology.md            # data cleaning, leakage controls, evaluation, results
├── scripts/
│   ├── run_weekly.sh             # cron entry point
│   └── tune.py                   # small, pre-declared hyper-parameter comparison
├── tests/                        # parsing, calendar, leakage tests
└── pyproject.toml
```

## Limitations

- The model's edge over the naive baseline is modest, and only statistically significant at 1–2 weeks ahead. See the p-values above.
- Day-old-chick (DOC) prices and supply volumes matter in the literature but have no public weekly source.
- Weather is measured at each province capital, not at the farms.
- PIHPS may revise past values; the committed snapshot is what was served on the download date.

## License

MIT. Data remain subject to the terms of their respective providers (Bank Indonesia, FRED/IMF, Aladhan, Nager.Date, Google, Open-Meteo, which requires attribution: weather data by [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0).
