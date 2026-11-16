# Chicken Price Forecast · Indonesia

Weekly forecasts of **retail broiler chicken prices for all 34 Indonesian provinces**, built entirely on **real, public data**, with features chosen from **peer-reviewed literature** and kept only if they **improve out-of-sample accuracy**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Data: Bank Indonesia PIHPS](https://img.shields.io/badge/data-BI%20PIHPS-orange)

> **v2.0** replaces the original synthetic-data prototype. The old model's R² of 0.97 came from synthetic data and features that leaked the target. This version uses ~78,000 daily price observations per series from Bank Indonesia, a leak-free feature pipeline, and a rolling-origin backtest against honest baselines.

## Results at a glance

Rolling-origin backtest: **360 weekly forecast origins × 34 provinces** (Dec 2019 – Nov 2026, 12,240 forecasts per horizon), retrained every 13 weeks using only data available at the time.

| Horizon | Model MAPE | Naive MAPE | Error vs naive | DM p-value | Direction correct | 80% interval coverage (CQR) |
|---|---|---|---|---|---|---|
| 1 week | **2.68%** | 2.79% | **−3.7%** | **0.006** | 58% | 83% |
| 2 weeks | 4.26% | 4.39% | −2.4% | 0.31 | 59% | 83% |
| 4 weeks | **5.72%** | 6.22% | **−6.6%** | **0.048** | 62% | 84% |
| 8 weeks | **7.15%** | 8.32% | **−12.4%** | **0.030** | 66% | 83% |

**Untouched hold-out** (Dec 2019 – Nov 2025, never used for any feature or parameter decision): −3.6% (p = 0.019), −2.4% (p = 0.35), −6.8% (p = 0.060), −12.1% (p = 0.059) at 1, 2, 4 and 8 weeks.

**By year**, the edge varies a lot: at 8 weeks the model beat naive by 30% in 2020 and 17% in 2023, but was 15% worse in 2025. A one-year evaluation can point either way, which is why the backtest now covers seven years.

**National average:** the PIHPS national price is almost exactly a weighted sum of the 34 provinces (0.05% fit error). Reconciling the national and province forecasts gives the best national forecast: **−11.8% error vs naive at 1 week (p = 0.011)**.

What this means:

- Weekly chicken prices are **close to a random walk**. "Price stays the same" is a hard baseline to beat, and the old v1 claim of R² = 0.97 was not realistic.
- Over seven years the model **significantly beats naive at 1, 4 and 8 weeks**. The gains are modest (4–12%) and vary by year, and the 2-week horizon is not significant.
- Prediction intervals use **conformalized quantile regression** (CQR): 83–84% realised coverage for a nominal 80%, with the best interval score of the three methods tested. The model also gives 58–66% direction accuracy and **SHAP explanations** for each province's forecast.

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

Each feature group is a hypothesis from published research ([docs/literature.md](docs/literature.md)). Groups were selected by ablation with Diebold–Mariano tests on the last 52 weeks, then checked again on all seven years:

| Feature group | Literature | Kept? | Seven-year check |
|---|---|---|---|
| Own-price dynamics & volatility | GARCH volatility clustering in weekly PIHPS broiler prices (Meilany et al., 2025) | ✅ core | n/a |
| Ramadan / Lebaran / holidays | Price spikes around Ramadan & Idul Fitri (Meilany et al., 2025; Permatasari et al., 2023) | ✅ | confirmed (p = 0.029 at 4w) |
| Producer → wholesale → retail margins | Asymmetric price transmission, NARDL (Jojo et al., 2021) | ✅ | confirmed (p = 0.035 at 1w) |
| Heat stress | Feed intake & growth loss in heat-stressed broilers (Liu et al., 2020; Teyssier et al., 2022) | ✅ | not confirmed (mixed) |
| Global feed prices | Feed ≈ 66% of farm cost (Sehabudin et al., 2022); feed price significant in 34-province panel (Fitrimawati et al., 2024) | ❌ hurt weekly accuracy | confirmed harmful (p = 0.013) |
| Rainfall | Evidence only for rice prices (Putra et al., 2021), not chicken | ❌ no effect | confirmed no effect |
| Egg prices | Partial | ❌ mixed | confirmed no effect |

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
chicken-forecast train      # 7-year backtest, train, forecast, SHAP (≈15 min)
chicken-forecast ablation   # re-run the feature-group experiment (≈30 min)
# or: chicken-forecast all  (fetch + build + train)
```

Run it weekly with cron (Monday 06:00):

```bash
0 6 * * 1 /path/to/chicken-price-forecast/scripts/run_weekly.sh >> /path/to/chicken-price-forecast/logs/cron.log 2>&1
```

## Dashboard

`streamlit run app/dashboard.py` opens five tabs:

- **Forecast**: price history, backtest track record and the 1–8 week forecast with an 80% interval for any province; retail vs wholesale vs farm-gate prices
- **All provinces**: national reconciled forecast, map and ranking of expected price changes
- **Why?**: SHAP drivers behind each province's forecast, plus global feature importance
- **Accuracy**: backtest vs naive baselines by horizon, significance tests, accuracy by year, per-province error
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
│   ├── models.py                 # XGBoost point + quantile models, baselines, backtest, DM test, CQR intervals
│   ├── hierarchy.py              # national series, NNLS province weights, forecast reconciliation
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

- The model's edge over the naive baseline is modest (4–12%), varies from year to year, and is not significant at 2 weeks. See the p-values above.
- Day-old-chick (DOC) prices and supply volumes matter in the literature but have no public weekly source.
- Weather is measured at each province capital, not at the farms.
- PIHPS may revise past values; the committed snapshot is what was served on the download date.

## License

MIT. Data remain subject to the terms of their respective providers (Bank Indonesia, FRED/IMF, Aladhan, Nager.Date, Google, Open-Meteo, which requires attribution: weather data by [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0).
