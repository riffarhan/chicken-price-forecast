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

- **Rolling-origin backtest** (Tashman, 2000): the most recent **360 weekly origins** per horizon (Dec 2019 – Nov 2026, 12,240 forecasts per horizon), plus 13 earlier warm-up origins used only to calibrate intervals. The model is retrained every 13 weeks using only rows whose target week is on or before the retraining date.
- **Baselines** (Hyndman & Athanasopoulos, *FPP3*, Ch. 5):
  - *Naive:* next price = last observed price.
  - *Seasonal naive:* apply last year's change over the same calendar window.
- **Metrics:** MAE, RMSE, MAPE, MAE reduction vs naive, direction accuracy, 80% interval coverage, mean interval width, and the interval score (Gneiting & Raftery, 2007).
- **Significance:** Diebold–Mariano test (Diebold & Mariano, 1995) with the Harvey–Leybourne–Newbold small-sample correction. Errors are averaged across provinces per origin week first, so the 34 correlated province series don't inflate the effective sample size.
- **Stability:** metrics are also reported per calendar year of the forecast origin (`outputs/backtest_metrics_by_year.csv`).
- **Prediction intervals**, three methods compared on the same backtest, all calibrated only on information available at each origin:
  1. *Scaled conformal:* point-model residuals normalised by each province's 13-week volatility × √h; the 10th/90th percentiles of past residuals give the band.
  2. *Quantile regression:* an XGBoost model trained with the pinball loss for the 10th, 50th and 90th percentiles.
  3. *Conformalized quantile regression (CQR;* Romano et al., 2019*):* the quantile band widened (or narrowed) by the 80th percentile of past conformity scores max(q10 − y, y − q90).
- **National forecasts and reconciliation:** the PIHPS national average is modelled as a weighted sum of the provinces, with non-negative least-squares weights estimated on the 104 weeks before each refit. Three national forecasts are compared: *direct* (the province model applied to the national series, which it never trains on), *bottom-up* (weighted sum of province forecasts) and *reconciled* (weighted least-squares reconciliation of national and province forecasts with inverse-variance weights, in the spirit of Hyndman et al., 2011 and Wickramasuriya et al., 2019).

## 6. Feature selection: literature + ablation

Each candidate feature group comes from the literature ([literature.md](literature.md)). Each group was then tested with two designs on the same backtest: **leave-one-group-out** (all groups vs all minus one) and **add-one-group-in** (core price features vs core plus one).

**How the decision was made.** The keep/drop decisions were taken on the most recent 52 origins only, so that older origins stay an untouched hold-out. Hyper-parameters were declared before any testing.

**Confirmation on seven years.** The same ablation was then re-run on all 360 origins (Dec 2019 – Nov 2026) as a check; the model was **not** changed afterwards, to keep the hold-out untouched. Change = MAE relative to the reference design, with the Diebold–Mariano p-value. Source: `outputs/ablation.csv`.

| Group | Add to core: Δ MAE 1w / 4w | DM p 1w / 4w | Leave out of all: Δ MAE 1w / 4w | DM p 1w / 4w | Decision (52 weeks) | Seven-year check |
|---|---|---|---|---|---|---|
| Calendar (Ramadan, Lebaran, holidays) | −0.6% / 0.0% | 0.41 / 1.00 | +1.1% / **+4.3%** | 0.088 / **0.029** | Keep | Confirmed (significant at 4w) |
| Supply chain (producer, wholesale) | **−0.5%** / −0.7% | **0.035** / 0.23 | +0.5% / +0.2% | 0.073 / 0.78 | Keep | Confirmed (significant at 1w) |
| Heat stress | **+0.4%** / +0.2% | **0.043** / 0.43 | +0.4% / +0.4% | 0.10 / 0.38 | Keep | **Not confirmed**: mixed signs |
| Global feed prices | **+2.0% / +6.7%** | **0.013 / 0.024** | −1.6% / −3.1% | **0.018** / 0.16 | Drop | Confirmed harmful |
| Rainfall | +0.4% / +0.1% | 0.13 / 0.81 | −0.1% / +0.6% | 0.63 / 0.25 | Drop | Confirmed no effect |
| Egg price | +0.1% / +0.2% | 0.48 / 0.54 | +0.1% / +0.3% | 0.64 / 0.46 | Drop | Confirmed no effect |

**Why does feed hurt, when the literature says feed drives broiler prices?** The two findings are compatible:

1. The literature establishes feed as the main driver of **cost and price levels** over years (feed ≈ 66% of farm cost, Sehabudin et al., 2022; feed price significant in an annual 34-province panel, Fitrimawati et al., 2024). Our target is the **1–4 week change** in retail price, which is dominated by demand timing and supply-chain adjustment.
2. Our feed series are **monthly global benchmarks** (IMF), published with a lag, not Indonesian feed-mill prices. Their slow trends let the trees identify time periods, so the model learns period-specific noise. The significant out-of-sample degradation shows exactly that.

A domestic, weekly feed-price series would be the right test of this hypothesis. None is publicly available.

**Rainfall** shows no effect, consistent with the absence of chicken-specific evidence in the literature (Section 7 of [literature.md](literature.md)).

**Final production feature set:** `core_price`, `calendar`, `supply_chain`, `weather_heat` (heat stress is kept because it was selected on the 52-week window, but the seven-year check does not support it; dropping it is a candidate for the next model version, to be validated on future data). Hyper-parameters are the values declared before any testing. Four pre-declared alternatives (`scripts/tune.py`) differed by less than ±0.15 MAPE points, so we kept the originals rather than pick the best on the evaluation window.

## 7. Results

Backtest: 360 weekly origins (Dec 2019 – Nov 2026) × 34 provinces, after 13 warm-up origins. Source: `outputs/backtest_metrics.csv`.

| Horizon | Model | MAE (Rp/kg) | MAPE | Error vs naive | DM p | Direction | 80% PI coverage (CQR) |
|---|---|---|---|---|---|---|---|
| 1 w | XGBoost | 959 | 2.68% | −3.7% | 0.006 | 58.4% | 83.0% |
| 1 w | Naive | 996 | 2.79% | | | | |
| 1 w | Seasonal naive | 1,460 | 4.08% | +46.5% | < 0.001 | 50.3% | |
| 2 w | XGBoost | 1,525 | 4.26% | −2.4% | 0.307 | 58.9% | 83.1% |
| 2 w | Naive | 1,562 | 4.39% | | | | |
| 4 w | XGBoost | 2,061 | 5.72% | −6.6% | 0.048 | 62.4% | 83.8% |
| 4 w | Naive | 2,208 | 6.22% | | | | |
| 8 w | XGBoost | 2,586 | 7.15% | −12.4% | 0.030 | 66.2% | 82.5% |
| 8 w | Naive | 2,953 | 8.32% | | | | |

### Untouched hold-out

Feature groups were first chosen on the most recent 52 origins (Section 6), so those weeks are not a clean test. The older 308 origins per horizon (Dec 2019 – Nov 2025) were never used for any modelling decision. Source: `outputs/backtest_metrics_by_period.csv`.

| Horizon | Error vs naive (hold-out) | DM p (hold-out) | Error vs naive (selection window) | DM p (selection window) |
|---|---|---|---|---|
| 1 w | −3.6% | 0.019 | −4.5% | 0.082 |
| 2 w | −2.4% | 0.35 | −2.0% | 0.67 |
| 4 w | −6.8% | 0.060 | −5.8% | 0.56 |
| 8 w | −12.1% | 0.059 | −14.1% | 0.32 |

### By year

Error vs naive by year of the forecast origin (negative = better). Source: `outputs/backtest_metrics_by_year.csv`.

| Horizon | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| 1 w | −0.2% | −4.9% | −3.8% | −4.7% | −4.3%* | −3.1% | −7.1%* |
| 2 w | +3.9% | −3.6% | −9.2% | +0.5% | −5.2% | −1.2% | −4.9% |
| 4 w | −13.1% | −2.3% | −9.8% | −1.5% | −8.4%* | +0.7% | −8.7% |
| 8 w | −29.6% | −2.9% | −9.3% | −17.1%* | −7.7% | +15.0% | −20.6% |

\* DM p < 0.05 within that year. 2026 covers 38–45 origins.

### Prediction intervals

Source: `outputs/interval_metrics.csv`. Lower interval score is better.

| Horizon | Scaled conformal: coverage / width / score | Quantile regression (raw) | CQR |
|---|---|---|---|
| 1 w | 80.0% / 9.3% / 4,720 | 70.0% / 6.6% / 4,749 | **83.0% / 9.1% / 4,620** |
| 2 w | 79.9% / 15.4% / 7,541 | 65.5% / 9.5% / 7,398 | **83.1% / 14.1% / 7,026** |
| 4 w | 77.8% / 20.8% / 10,410 | 61.0% / 11.7% / 10,375 | **83.8% / 19.6% / 9,510** |
| 8 w | 74.8% / 25.0% / 13,276 | 53.3% / 12.7% / 13,570 | **82.5% / 23.4% / 11,703** |

Raw quantile regression is far too narrow. CQR is both narrower than scaled conformal and closer to (slightly above) nominal coverage, so its interval score is 2–12% lower; it is now the production interval. The quantile model's median is also a slightly better point forecast than the pseudo-Huber model (MAPE 2.64% vs 2.68% at 1 week, 7.01% vs 7.15% at 8 weeks); we report this but keep the pre-declared point model.

### National forecasts and reconciliation

Source: `outputs/hierarchy_metrics.csv`. The NNLS weights reproduce the published national average with 0.05% mean absolute error (weights sum to 0.99; the largest is West Java at 5.2%).

| Horizon | Naive MAPE | Direct | Bottom-up | Reconciled | Reconciled vs naive (DM p) |
|---|---|---|---|---|---|
| 1 w | 1.78% | 1.64% | 1.61% | **1.58%** | −11.8% (0.011) |
| 2 w | 2.82% | 2.74% | 2.70% | **2.69%** | −4.7% (0.42) |
| 4 w | 4.07% | 3.89% | 3.82% | **3.80%** | −5.8% (0.47) |
| 8 w | 5.87% | **5.25%** | 5.34% | 5.30% | −8.8% (0.39) |

Reconciliation leaves province accuracy essentially unchanged (1-week MAPE 2.681% before and after) while making province and national forecasts consistent with each other.

### Interpretation

- **Over seven years the model beats naive at every horizon**, significantly at 1, 4 and 8 weeks. The 2-week horizon is not significant.
- **The edge varies by year.** 2025 was poor at long horizons (8 weeks: 15% worse than naive), 2020 and 2023 were strong. The earlier one-year evaluation, which suggested no skill at 8 weeks, was an unlucky window.
- **Seasonal naive** is far worse than naive at every horizon: chicken price seasonality follows the moving Hijri calendar, not the Gregorian year.
- **Direction:** 58–66% correct, versus 50% for a coin flip.

## 8. Limitations

- **Weekly chicken prices are close to a random walk.** The model's edge over "price stays the same" is real but modest. Treat forecasts as a slightly better-informed baseline with honest uncertainty bands, not as a crystal ball.
- **Missing drivers.** DOC prices, chick-in volumes and government supply-control interventions are significant in the literature but have no public weekly data.
- **Weather at the province capital** is a coarse proxy for conditions at farms, which are often in other districts and in closed houses.
- **Global feed benchmarks** need not track domestic Indonesian maize prices.
- **Year-to-year variation.** The backtest covers seven years (2020–2026), and the model's edge varies a lot between years; any single year can mislead.
- **PIHPS revisions.** The cached snapshot is what PIHPS served on the download date. Past values may be revised later.

## References

See [literature.md](literature.md) for full citations. Additional methodology references:

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4
