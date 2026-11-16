# Literature review: what drives broiler chicken prices in Indonesia?

This document explains **why each feature group exists** in the model. The approach has two steps:

1. **Hypothesis from literature.** A feature group is only considered if published research gives a mechanism for it.
2. **Empirical test.** Every group is then tested with a leave-one-group-out ablation on a rolling-origin backtest (`chicken-forecast ablation`, results in [`outputs/ablation.csv`](../outputs/ablation.csv)). A group stays in the production model only if it improves out-of-sample accuracy. See [methodology.md](methodology.md) for the results.

Every citation below was checked against the paper's full text or its publisher/index page. Where we could only verify part of a paper, we say so.

---

## 1. Feed cost (maize, soybean meal, exchange rate)

**Mechanism.** Feed is the largest cost of producing a broiler, and Indonesian feed depends on maize and imported soybean meal. Rising feed costs squeeze farm margins and, over time, push the farm-gate and retail price up.

| Evidence | Source |
|---|---|
| Feed was the largest production cost in both partnership patterns (**66.64%** PIR, **66.08%** *makloon*), and DOC was second (~26%). 246 farms, Sukabumi, West Java. | Sehabudin, U., Daryanto, A., Sinaga, B. M., & Priyanti, A. (2022). The structure of costs and income of broiler chicken farming in different partnership patterns in Sukabumi Regency, West Java, Indonesia. *Jurnal Ilmu-Ilmu Peternakan*, 32(3), 380–387. https://doi.org/10.21776/ub.jiip.2022.032.03.09 |
| Panel regression across **34 provinces** (BPS data, 5 years): feed price (coef. 0.134, p < 0.001), DOC price (0.047, p = 0.040) and supply (0.120, p = 0.044) significantly affect broiler price. Demand was *not* significant (p = 0.23). Adj. R² = 0.686. | Fitrimawati, Nurhayati, Indrayani, I., & Qamara, C. (2024). The analysis of broiler price fluctuations in Indonesia. *Jurnal Ilmiah Peternakan Terpadu*, 12(1), 23–35. https://doi.org/10.23960/jipt.v12i1.p23-35 |
| Feed is ~70% of broiler production cost. Broiler feed is 40–50% maize, so "corn price will determine chicken feed price". | Firdaus, M., & Komalasari, L. (2010). Feasibility analyses of integrated broiler production. *Media Peternakan*, 33(3), 182–188. https://doi.org/10.5398/medpet.2010.33.3.182 |
| Feed "is the most crucial cost of production" in tropical broiler systems. Limited maize and oilseed-cake supply is a key constraint. | Kpomasse, C. C., Oke, O. E., Houndonougbo, F. M., & Tona, K. (2021). Broiler production challenges in the tropics: A review. *Veterinary Medicine and Science*, 7(3), 831–842. https://doi.org/10.1002/vms3.435 |

**Data used:** IMF global maize and soybean-meal prices (FRED `PMAIZMTUSDM`, `PSMEAUSDM`) converted to rupiah with the monthly IDR/USD rate (FRED `CCUSMA02IDM618N`). These are *global* benchmarks, not Indonesian feed-mill prices: a public weekly series of Indonesian feed prices does not exist. They therefore capture imported-input pressure only.

**Gap:** DOC (day-old chick) price is significant in Fitrimawati et al. (2024) and is the second-largest cost (Sehabudin et al., 2022), but there is **no free public time series** of DOC prices by province. It is a known missing variable.

## 2. Supply-chain price transmission (producer → wholesale → retail)

**Mechanism.** Retail prices follow farm-gate and wholesale prices with a lag. When live-bird prices fall, retailers may pass the change on slowly or only partly. That gap, the retail-to-producer margin, can predict where retail goes next.

| Evidence | Source |
|---|---|
| NARDL on monthly farmer, wholesale and retail prices (Jan 2014 – Dec 2019, West Java). Found **long-run asymmetric** transmission farmer→wholesale (Wald p = 0.0001) and wholesale→retail (p = 0.0006), plus short-run asymmetry farmer→wholesale (p = 0.0004). The direct farmer→retail link showed no long-run asymmetry (p = 0.26). | Jojo, J., Harianto, H., Nurmalina, R., & Hakim, D. B. (2021). Price transmission in the broiler chicken sector in Indonesia: An asymmetric NARDL approach. *Journal of the Austrian Society of Agricultural Economics*, 17(02), 293–302. |

**Data used:** PIHPS wholesale (*pedagang besar*) and producer (*produsen*) prices for the same commodity and province. Features: retail/producer and retail/wholesale log margins, plus recent producer and wholesale price changes.

## 3. Ramadan, Idul Fitri and other religious holidays

**Mechanism.** Household and food-business demand for chicken surges before and during Ramadan and peaks before Lebaran. The Hijri calendar moves ~11 days earlier each year, so fixed month or quarter dummies cannot capture it.

| Evidence | Source |
|---|---|
| Uses **weekly PIHPS data** (Mar 2017 – Apr 2023). Found significant ARCH effects and volatility clustering in broiler prices, with GARCH(1,1) persistence α + β = 0.743 (α = 0.258, β = 0.485). Ramadan is described as an annual period of escalating volatility. | Meilany, S. A., Sadi'yah, A. A., & Wulandari, A. (2025). Volatility analysis of broiler chicken prices in the Ramadhan period: A strategic food perspective. *Journal of Agricultural Socio-Economics*, 6(1), 20–29. https://doi.org/10.33474/jase.v6i1.23888 |
| Literature study: prices of livestock foods (beef, chicken, eggs) rise ahead of Ramadan and especially around Idul Fitri, driven by demand that supply cannot meet. | Permatasari, F. A., Utami, K. K., & Az-Zahra, Y. D. (2023). The effect of fluctuations in food commodity prices in the month of Ramadan on inflation in Indonesia. *ASIAN Economic and Business Development*, 6(1), 23–30. https://doi.org/10.54204/AEBD/Vol6No1August2023003 |
| Chicken meat prices "always fluctuate every year". Beef rises ahead of national religious holidays (HBKN), especially Idul Fitri. | Azalia, R., Sukiyono, K., & Windirah, N. (2023). Price behavior of beef and chicken meat in Indonesia. *Jurnal Ilmu-Ilmu Peternakan*, 33(2), 229–239. https://doi.org/10.21776/ub.jiip.2023.033.02.10 |

**Data used:** Hijri calendar from the Aladhan API, with official Indonesian dates from Google's public "Holidays in Indonesia" calendar substituted where available (2021+), and fixed-date public holidays from Nager.Date. Over the 12 holidays where both sources exist, the computed and official dates agree within ±1 day (`chicken_forecast.data.calendar.validate()`).

## 4. Volatility clustering and own-price dynamics

**Mechanism.** Large price moves tend to follow large moves (GARCH-type behaviour), and chicken prices are among the most unstable food prices in Indonesia. Recent price changes, deviations from moving averages, and recent volatility are therefore core predictors.

| Evidence | Source |
|---|---|
| Volatility clustering with persistence α + β = 0.743; "past volatility has a stronger effect on current price changes than new events". | Meilany et al. (2025), above |
| Monthly 2018–2022: chicken meat is **more unstable than beef** (CV 6.02% vs 5.14%, price-instability index 5.63 vs 1.41, divergence 25.13% vs 15.42%). | Azalia et al. (2023), above |

**Data used:** Lags, moving-average gaps and 13-week rolling volatility of the province's own PIHPS retail price.

## 5. Spatial link to the national market

**Mechanism.** Provincial markets are linked through inter-island trade, so a province priced far above or below the national average tends to converge.

**Evidence status: weak.** We did not find a verified paper that tests province-level cointegration for Indonesian *broiler* prices. The related work we found covers districts within Central Java or egg prices in Sumatra, and we could not read those in full. The feature (gap to the PIHPS national average) is kept as part of the core price block because it is cheap and leak-free, and its value is judged by the ablation, not by the literature.

## 6. Heat stress (temperature)

**Mechanism.** Above the thermoneutral zone, broilers eat less and grow more slowly. That reduces live weight supplied a few weeks later.

| Evidence | Source |
|---|---|
| Meta-analysis of 12 studies: heat stress significantly reduced feed intake (WMD −97.95), body-weight gain (WMD −151.40) and raised FCR (+0.17) and mortality (OR 3.74). | Liu, L., Ren, M., Ren, K., Jin, Y., & Yan, M. (2020). Heat stress impacts on broiler performance: A systematic review and meta-analysis. *Poultry Science*, 99(11), 6205–6211. https://doi.org/10.1016/j.psj.2020.08.019 |
| Reduced feed intake explains most of the growth loss under heat stress. Pair-feeding studies put the share at 60–99%. | Teyssier, J.-R., Brugaletta, G., Sirri, F., Dridi, S., & Rochell, S. J. (2022). A review of heat stress in chickens. Part II: Insights into protein and energy utilization and feeding. *Frontiers in Physiology*, 13, 943612. https://doi.org/10.3389/fphys.2022.943612 |
| Thermoneutral zone for broilers is about 18–24 °C. Heat stress sets in above ~25 °C in tropical production. | Kpomasse et al. (2021), above |

**Caveat:** this evidence is about **bird performance**, not prices. The link to prices (less supply → higher price) is our inference. Most Indonesian broilers are raised in closed or semi-closed houses, so outdoor temperature at the provincial capital is a rough proxy. The ablation decides whether it helps.

## 7. Rainfall

**Evidence status: weak for chicken.** The verified work links rainfall and climate to **rice** price volatility, not chicken:

| Evidence | Source |
|---|---|
| Province-level rice price volatility is driven by climate and macroeconomic variables, with effects that differ from region to region. Rainfall significantly affects rice prices in some provinces. | Putra, A. W., Supriatna, J., Koestoer, R. H., & Soesilo, T. E. B. (2021). Differences in local rice price volatility, climate, and macroeconomic determinants in the Indonesian market. *Sustainability*, 13(8), 4465. https://doi.org/10.3390/su13084465 |

A plausible channel for chicken is (a) flooding disrupting transport from farms to markets, and (b) climate effects on domestic maize, and therefore feed. **We did not find a paper that demonstrates either for broiler prices.** Rainfall is therefore treated as an exploratory hypothesis. It is dropped unless the ablation shows a clear gain.

## 8. Egg prices (related commodity)

**Mechanism.** Eggs and broiler meat share the same feed base (maize, soybean meal), and households substitute between them as protein sources.

**Evidence status: partial.** A regional study reporting short-run pass-through from egg prices to chicken prices in South Kalimantan appeared in search results, but we could not read its full text, so it is not cited here. Feed-cost commonality is supported by Sections 1 and 6. Kept only if the ablation supports it.

## 9. Forecasting methodology

| Choice | Source |
|---|---|
| Rolling-origin out-of-sample evaluation instead of a single train/test split | Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437–450. |
| Naive and seasonal-naive benchmarks. A model is only useful if it beats them. | Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.), Ch. 5. https://otexts.com/fpp3/accuracy.html |
| Gradient-boosted trees | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*. https://doi.org/10.1145/2939672.2939785 |
| Per-prediction explanations | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*. arXiv:1705.07874 |
| Prediction intervals calibrated on past out-of-sample residuals in time order | Stocker, M., Małgorzewicz, W., Fontana, M., & Ben Taieb, S. (2025). A gentle introduction to conformal time series forecasting. arXiv:2511.13608 |
| Conformalized quantile regression (CQR): quantile models + conformal calibration, adaptive interval widths | Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems 32*, 3543–3553. arXiv:1905.03222 |
| Interval score to compare interval methods (rewards narrow intervals, penalises misses) | Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437 |
| Reconciling province and national forecasts so they are coherent | Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., & Shang, H. L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics & Data Analysis*, 55(9), 2579–2589. https://doi.org/10.1016/j.csda.2011.03.006 |
| Weighted (MinT-family) reconciliation using forecast-error variances | Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*, 114(526), 804–819. https://doi.org/10.1080/01621459.2018.1448825 |

## Known missing variables

These are supported by the literature but have **no free, public, regularly updated source** we could use:

- **DOC (day-old chick) price**: significant in Fitrimawati et al. (2024).
- **Broiler supply / chick-in volumes**: significant in Fitrimawati et al. (2024). Published by the Ministry of Agriculture only as annual figures.
- **Government supply-control policy** (e.g., hatching-egg culling instructions): affects supply, but there is no machine-readable event list.

## Not verified / not cited

Search results also pointed to studies on corn and soybean prices vs broiler prices (ResearchGate), Ramadan and Eid egg-to-chicken pass-through in South Kalimantan, ENSO effects on food crop prices in Indonesia, and market integration in Central Java. We could not access the full text of these, so they are deliberately **not** used as justification above.
