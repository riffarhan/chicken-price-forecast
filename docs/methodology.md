# Methodology

## 1. Problem

Forecast the weekly average **retail price of fresh broiler chicken meat** (*daging ayam ras segar*, traditional markets, Rp/kg) for each of Indonesia's 34 provinces (PIHPS coverage), at horizons of **1, 2, 4 and 8 weeks**.

## 2. Data

| Source | What | Raw frequency | Coverage |
|---|---|---|---|
| Bank Indonesia **PIHPS** | Chicken retail, wholesale and producer prices; egg retail price | Daily (working days) | 34 provinces + national, Jan 2018 → |
| **FRED / IMF** Primary Commodity Prices | Global maize and soybean-meal prices (USD/t); IDR per USD | Monthly | 2016 → |
| **Aladhan** Hijri calendar API | Hijri date of every day (Ramadan, Syawal, Dzulhijjah) | Daily | 2017–2028 |
| **Google** "Holidays in Indonesia" (public ICS) | Official holiday and *cuti bersama* dates | Event | 2021–2027 |
| **Nager.Date** | Indonesian fixed-date public holidays | Event | 2017–2028 |
| **Open-Meteo** archive (ERA5 / ERA5-Land reanalysis) | Daily precipitation and max/mean 2 m temperature | Daily | Oct 2017 → |
| **Open-Meteo** geocoding | Coordinates of each province capital | n/a | n/a |

All raw downloads are cached in `data/raw/` and committed, so results can be reproduced exactly even if a source changes later.

### Cleaning

1. **Daily outliers:** a daily price more than 0.22 log-points (~25%) away from its centred 15-day rolling median is treated as a data-entry error and dropped. The current snapshot drops 28 retail, 82 wholesale, 4 producer and 35 egg observations out of ~300k.
2. **Weekly aggregation:** daily prices are averaged into Monday–Sunday weeks, labelled by the Sunday.
3. **Gaps:** gaps of up to 3 weeks inside a province's series are linearly interpolated. Longer gaps stay missing.
4. **Incomplete week:** the current (partial) week is dropped.
5. **Publication lag:** monthly FRED values for month *M* are only used from the start of month *M+2*, a conservative allowance for release delays.

## 3. Target and model

- **Target:** the log price change over *h* weeks, `y = log(p[t+h]) − log(p[t])`. Modelling changes rather than levels puts provinces with very different price levels on one scale, and makes the naive forecast correspond exactly to `ŷ = 0`.
- **Model:** one global XGBoost regressor per horizon (direct multi-step strategy), trained on all 34 provinces together, with a pseudo-Huber loss (robust to occasional shocks) and recency weighting (half-life 3 years).
- **Forecast:** `p̂[t+h] = p[t] · exp(ŷ)`.

## 4. Leakage controls

The v1 model reported R² ≈ 0.97 because several features (rolling means, ratios, week-over-week change) included the price being predicted. v2 enforces:

- Features at origin week *t* use only data up to and including *t*.
- Each province is placed on a complete weekly grid before any lag/rolling operation, so "1 week ago" always means one calendar week, even across data gaps.
- Calendar features are computed for the *target* week. This is legitimate because holidays are known in advance.
- Anomaly features (weather, price level) are measured against **expanding** means, never full-sample means.
- `tests/test_features.py::test_features_only_use_the_past` corrupts all future prices and asserts that past features do not change.

## 5. Evaluation design

- **Rolling-origin backtest** (Tashman, 2000): the most recent 52 weekly origins, plus 13 earlier warm-up origins used only to calibrate intervals. The model is retrained every 13 weeks using only rows whose target week is on or before the retraining date.
- **Baselines** (Hyndman & Athanasopoulos, *FPP3*, Ch. 5):
  - *Naive:* next price = last observed price.
  - *Seasonal naive:* apply last year's change over the same calendar window.
- **Metrics:** MAE, RMSE, MAPE, MAE reduction vs naive, direction accuracy, 80% interval coverage.
- **Significance:** Diebold–Mariano test (Diebold & Mariano, 1995) with the Harvey–Leybourne–Newbold small-sample correction. Errors are averaged across provinces per origin week first, so the 34 correlated province series don't inflate the effective sample size.
- **Prediction intervals:** split-conformal style. Residuals are normalised by each province's recent 13-week volatility × √h, and their empirical 10th/90th percentiles come only from forecasts whose outcomes were already known at the time (sequential calibration).

## 6. Feature selection: literature + ablation

Each candidate feature group comes from the literature ([literature.md](literature.md)). Each group was then tested with two designs on the same backtest: **leave-one-group-out** (all groups vs all minus one) and **add-one-group-in** (core price features vs core plus one).

Rolling-origin backtest over 156 origins at h = 1 and 4 weeks. Change = MAE relative to the reference design, with the Diebold–Mariano p-value. Source: `outputs/ablation.csv`.

| Group | Add to core: Δ MAE 1w / 4w | DM p 1w / 4w | Leave out of all: Δ MAE 1w / 4w | DM p 1w / 4w | Decision |
|---|---|---|---|---|---|
| Calendar (Ramadan, Lebaran, holidays) | −1.2% / +0.2% | 0.24 / 0.94 | +1.3% / +2.6% | 0.13 / 0.40 | **Keep**: helps in 3 of 4 comparisons, strong literature |
| Supply chain (producer, wholesale) | −0.6% / −0.3% | **0.032** / 0.30 | +0.1% / +0.3% | 0.88 / 0.66 | **Keep**: significant at 1 week |
| Heat stress | −0.2% / −0.1% | 0.52 / 0.84 | +0.2% / +0.9% | 0.68 / 0.28 | **Keep**: small but same sign in all 4 comparisons |
| Global feed prices | **+2.6% / +9.3%** | **0.022 / 0.024** | −2.1% / −6.2% | **0.010** / 0.071 | **Drop**: significantly harmful |
| Rainfall | +0.2% / −0.2% | 0.43 / 0.75 | −0.2% / +0.1% | 0.58 / 0.81 | **Drop**: no effect |
| Egg price | −0.2% / −0.3% | 0.50 / 0.61 | −0.5% / +1.2% | 0.13 / 0.11 | **Drop**: mixed signs |

**Why does feed hurt, when the literature says feed drives broiler prices?** The two findings are compatible:

1. The literature establishes feed as the main driver of **cost and price levels** over years (feed ≈ 66% of farm cost, Sehabudin et al., 2022; feed price significant in an annual 34-province panel, Fitrimawati et al., 2024). Our target is the **1–4 week change** in retail price, which is dominated by demand timing and supply-chain adjustment.
2. Our feed series are **monthly global benchmarks** (IMF), published with a lag, not Indonesian feed-mill prices. Their slow trends let the trees identify time periods, so the model learns period-specific noise. The significant out-of-sample degradation shows exactly that.

A domestic, weekly feed-price series would be the right test of this hypothesis. None is publicly available.

**Rainfall** shows no effect, consistent with the absence of chicken-specific evidence in the literature (Section 7 of [literature.md](literature.md)).

**Final production feature set:** `core_price`, `calendar`, `supply_chain`, `weather_heat`. Hyper-parameters are the values declared before any testing. Four pre-declared alternatives (`scripts/tune.py`) differed by less than ±0.15 MAPE points, so we kept the originals rather than pick the best on the evaluation window.

## 7. Results

Backtest: 156 weekly origins (Oct 2023 – Nov 2026) × 34 provinces, after 13 warm-up origins. Source: `outputs/backtest_metrics.csv`.

| Horizon | Model | MAE (Rp/kg) | MAPE | Error vs naive | DM p | Direction | 80% PI coverage |
|---|---|---|---|---|---|---|---|
| 1 w | XGBoost | 861 | 2.29% | −4.9% | < 0.001 | 59.2% | 78.3% |
| 1 w | Naive | 906 | 2.42% | | | | |
| 1 w | Seasonal naive | 1,299 | 3.46% | +43.4% | < 0.001 | 50.2% | |
| 2 w | XGBoost | 1,377 | 3.68% | −3.1% | 0.143 | 58.8% | 77.0% |
| 2 w | Naive | 1,422 | 3.82% | | | | |
| 4 w | XGBoost | 1,903 | 5.07% | −4.4% | 0.305 | 61.8% | 75.9% |
| 4 w | Naive | 1,991 | 5.34% | | | | |
| 8 w | XGBoost | 2,494 | 6.65% | −4.4% | 0.537 | 63.1% | 72.7% |
| 8 w | Naive | 2,609 | 7.00% | | | | |

### Untouched hold-out

The feature groups were chosen on the most recent 52 origins (Section 6), so those weeks are not a clean test. For each horizon, the origins before its last 52 (roughly Oct 2023 – Nov 2025) were never used for any modelling decision. Source: `outputs/backtest_metrics_by_period.csv`.

| Horizon | Error vs naive (hold-out) | DM p (hold-out) | Error vs naive (selection window) | DM p (selection window) |
|---|---|---|---|---|
| 1 w | −4.7% | 0.003 | −5.2% | 0.059 |
| 2 w | −3.9% | 0.046 | −1.8% | 0.71 |
| 4 w | −3.8% | 0.25 | −5.3% | 0.60 |
| 8 w | +2.5% | 0.72 | −14.2% | 0.32 |

### Interpretation

- **1 week ahead:** a small (~5%) but **statistically significant** improvement over naive in both periods. This is the model's dependable use case.
- **2 weeks:** better than naive in both periods, significant on the hold-out (p = 0.046).
- **4 weeks:** better than naive in both periods (−3.8% / −5.3%) but not significant.
- **8 weeks:** the large gain in the recent window does **not** replicate on the hold-out (+2.5% worse than naive). At 4–8 weeks the point forecast should be treated as indicative and the interval used for planning. The dashboard shows a per-horizon reliability notice based on the hold-out.
- **Seasonal naive** is far worse than naive at every horizon. Chicken price seasonality is driven by the moving Hijri calendar, not the Gregorian year, which is exactly why the Hijri-based calendar features are needed.
- **Intervals** are well calibrated up to 4 weeks (76–78% for a nominal 80%) and somewhat too narrow at 8 weeks (73%).
- **Direction:** the model calls the direction of the price move correctly 59–63% of the time, versus 50% for a coin flip.

## 8. Limitations

- **Weekly chicken prices are close to a random walk.** The model's edge over "price stays the same" is real but modest. Treat forecasts as a slightly better-informed baseline with honest uncertainty bands, not as a crystal ball.
- **Missing drivers.** DOC prices, chick-in volumes and government supply-control interventions are significant in the literature but have no public weekly data.
- **Weather at the province capital** is a coarse proxy for conditions at farms, which are often in other districts and in closed houses.
- **Global feed benchmarks** don't capture Indonesian maize import policy, which can make domestic maize diverge sharply from world prices.
- **One evaluation window.** The backtest covers the most recent year. Different regimes (e.g., COVID-19 in 2020) may behave differently.
- **PIHPS revisions.** The cached snapshot is what PIHPS served on the download date. Past values may be revised later.

## References

See [literature.md](literature.md) for full citations. Additional methodology references:

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4
