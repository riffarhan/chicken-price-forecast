"""Model fitting, baselines, rolling-origin backtest and prediction intervals.

Strategy: one global XGBoost model per horizon (direct multi-step), trained
on all provinces together, predicting the log price change over h weeks.

Every result is compared with two baselines a real analyst would use:

* **naive**: next price = last observed price (random walk)
* **seasonal naive**: apply last year's change over the same calendar window

If the model can't beat the naive forecast, it isn't adding value. The
backtest reports skill relative to naive explicitly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from chicken_forecast.config import (
    BACKTEST_WEEKS,
    HORIZONS,
    INTERVAL_COVERAGE,
    QUANTILES,
    TARGET_SERIES,
    XGB_PARAMS,
)
from chicken_forecast.features import FEATURES, make_supervised

log = logging.getLogger(__name__)

REFIT_EVERY = 13  # weeks between refits during the backtest
CALIBRATION_WARMUP = 13  # backtest weeks used only to seed interval calibration
RECENCY_HALF_LIFE_WEEKS = 156  # older observations get gradually less weight
MIN_HISTORY_COL = "ret_4"  # rows must have at least 4 weeks of history


def _sample_weight(weeks: pd.Series, ref: pd.Timestamp) -> np.ndarray:
    age = (ref - weeks).dt.days.to_numpy() / 7.0
    return 0.5 ** (age / RECENCY_HALF_LIFE_WEEKS)


def fit_model(train: pd.DataFrame, features: list[str] = FEATURES) -> xgb.XGBRegressor:
    train = train.dropna(subset=["y", MIN_HISTORY_COL])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train[features],
        train["y"],
        sample_weight=_sample_weight(train["week"], train["week"].max()),
    )
    return model


def fit_quantile_model(train: pd.DataFrame, features: list[str] = FEATURES) -> xgb.XGBRegressor:
    """One multi-output XGBoost model for the 10th, 50th and 90th percentiles.

    Same trees and data as the point model, but trained with the pinball
    (quantile) loss, so it can learn uncertainty that differs by province,
    season and recent history instead of scaling one residual distribution.
    """
    train = train.dropna(subset=["y", MIN_HISTORY_COL])
    params = {k: v for k, v in XGB_PARAMS.items() if k not in ("objective", "huber_slope")}
    model = xgb.XGBRegressor(
        **params, objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES)
    )
    model.fit(
        train[features],
        train["y"],
        sample_weight=_sample_weight(train["week"], train["week"].max()),
    )
    return model


def predict_quantiles(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Quantile predictions, sorted per row so the quantiles never cross."""
    return np.sort(model.predict(X), axis=1)


def seasonal_naive_change(feats: pd.DataFrame, horizon: int) -> pd.Series:
    """log(p[t+h-52] / p[t-52]) for each row: last year's change over the same window."""
    lp = feats[["province", "week", TARGET_SERIES]].copy()
    lp["lp"] = np.log(lp[TARGET_SERIES])
    lookup = lp.set_index(["province", "week"])["lp"]
    base = pd.MultiIndex.from_arrays([feats["province"], feats["week"] - pd.Timedelta(weeks=52)])
    ahead = pd.MultiIndex.from_arrays(
        [feats["province"], feats["week"] + pd.Timedelta(weeks=horizon - 52)]
    )
    change = lookup.reindex(ahead).to_numpy() - lookup.reindex(base).to_numpy()
    return pd.Series(change, index=feats.index).fillna(0.0)


def _scale(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Per-row uncertainty scale: recent weekly volatility x sqrt(horizon)."""
    vol = df["vol_13"].fillna(df["vol_13"].median()).clip(lower=0.004)
    return vol.to_numpy() * np.sqrt(horizon)


def interval_quantiles(norm_resid: np.ndarray, coverage: float = INTERVAL_COVERAGE):
    a = (1 - coverage) / 2
    return np.nanquantile(norm_resid, a), np.nanquantile(norm_resid, 1 - a)


def backtest(
    feats: pd.DataFrame,
    features: list[str] = FEATURES,
    horizons: list[int] = HORIZONS,
    quantiles: bool = False,
    extra_feats: pd.DataFrame | None = None,
    n_weeks: int = BACKTEST_WEEKS,
) -> pd.DataFrame:
    """Rolling-origin evaluation over the last ``n_weeks`` (+ warm-up) weeks.

    For each block of REFIT_EVERY origins the model is retrained using only
    rows whose *target* week is on or before the block's first origin, i.e.
    exactly the information available at that time.

    quantiles:   also fit the quantile model in every block (q10/q50/q90).
    extra_feats: origin features of extra series (e.g. the national average)
                 that are not trained on but are predicted by the same block
                 models; returned in ``bt.attrs["extra"]``.
    """
    last_week = feats["week"].max()
    results, extra_results = [], []
    for h in horizons:
        sup = make_supervised(feats, h)
        ex_sup = make_supervised(extra_feats, h) if extra_feats is not None else None
        last_origin = last_week - pd.Timedelta(weeks=h)
        n_origins = n_weeks + CALIBRATION_WARMUP
        origins = pd.date_range(end=last_origin, periods=n_origins, freq="7D")
        sup["pred_snaive"] = seasonal_naive_change(sup, h)

        for b in range(0, n_origins, REFIT_EVERY):
            block = origins[b : b + REFIT_EVERY]
            cutoff = block[0]
            train = sup[sup["target_week"] <= cutoff]
            model = fit_model(train, features)
            test = sup[sup["week"].isin(block)].dropna(subset=["y", MIN_HISTORY_COL]).copy()
            test["pred_xgb"] = model.predict(test[features])
            test["scale"] = _scale(test, h)
            if quantiles:
                qm = fit_quantile_model(train, features)
                q = predict_quantiles(qm, test[features])
                test["q10"], test["q50"], test["q90"] = q[:, 0], q[:, 1], q[:, 2]
            results.append(test)
            if ex_sup is not None:
                ex = ex_sup[ex_sup["week"].isin(block)].dropna(subset=["y", MIN_HISTORY_COL]).copy()
                if len(ex):
                    ex["pred_xgb"] = model.predict(ex[features])
                    ex["scale"] = _scale(ex, h)
                    extra_results.append(ex)
        log.info("  backtest h=%d done", h)

    bt = pd.concat(results, ignore_index=True)
    bt["pred_naive"] = 0.0
    bt = _sequential_intervals(bt, horizons)

    keep = [
        "province", "week", "target_week", "horizon", TARGET_SERIES, "target_price",
        "y", "pred_xgb", "pred_naive", "pred_snaive", "scale", "lo", "hi", "is_warmup",
    ]
    if quantiles:
        keep += ["q10", "q50", "q90", "cqr_lo", "cqr_hi"]
    bt = bt[keep].rename(columns={"week": "origin_week", TARGET_SERIES: "last_price"})
    for m in ["xgb", "naive", "snaive"]:
        bt[f"price_{m}"] = bt["last_price"] * np.exp(bt[f"pred_{m}"])
    bt["price_lo"] = bt["last_price"] * np.exp(bt["lo"])
    bt["price_hi"] = bt["last_price"] * np.exp(bt["hi"])
    if quantiles:
        for c in ["q10", "q50", "q90", "cqr_lo", "cqr_hi"]:
            bt[f"price_{c}"] = bt["last_price"] * np.exp(bt[c])
    bt = bt.sort_values(["horizon", "province", "origin_week"]).reset_index(drop=True)
    if extra_results:
        ex = pd.concat(extra_results, ignore_index=True)
        ex = ex[["province", "week", "target_week", "horizon", TARGET_SERIES, "target_price",
                 "y", "pred_xgb", "scale"]].rename(columns={"week": "origin_week", TARGET_SERIES: "last_price"})
        ex["price_xgb"] = ex["last_price"] * np.exp(ex["pred_xgb"])
        bt.attrs["extra"] = ex
    return bt


def _sequential_intervals(bt: pd.DataFrame, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """Prediction intervals calibrated only on residuals already observable.

    For an origin week o, use residuals from earlier origins whose target week
    is <= o. This is honest split-conformal calibration in time order, so the
    reported coverage is what you'd have actually experienced.
    """
    bt = bt.copy()
    bt["norm_resid"] = (bt["y"] - bt["pred_xgb"]) / bt["scale"]
    bt["lo"] = np.nan
    bt["hi"] = np.nan
    has_q = "q10" in bt.columns
    if has_q:
        # Conformalized quantile regression (CQR) conformity score
        bt["cqr_score"] = np.maximum(bt["q10"] - bt["y"], bt["y"] - bt["q90"])
        bt["cqr_lo"] = np.nan
        bt["cqr_hi"] = np.nan
    for h in horizons:
        sub = bt[bt["horizon"] == h].sort_values("week")
        first_week = sub["week"].min()
        warmup_end = first_week + pd.Timedelta(weeks=CALIBRATION_WARMUP)
        for o in sub["week"].unique():
            if o < warmup_end:
                continue
            known_mask = sub["target_week"] <= o
            known = sub.loc[known_mask, "norm_resid"].to_numpy()
            if len(known) < 100:
                continue
            q_lo, q_hi = interval_quantiles(known)
            idx = sub.index[sub["week"] == o]
            bt.loc[idx, "lo"] = bt.loc[idx, "pred_xgb"] + q_lo * bt.loc[idx, "scale"]
            bt.loc[idx, "hi"] = bt.loc[idx, "pred_xgb"] + q_hi * bt.loc[idx, "scale"]
            if has_q:
                adj = np.nanquantile(sub.loc[known_mask, "cqr_score"].to_numpy(), INTERVAL_COVERAGE)
                bt.loc[idx, "cqr_lo"] = bt.loc[idx, "q10"] - adj
                bt.loc[idx, "cqr_hi"] = bt.loc[idx, "q90"] + adj
    bt["is_warmup"] = bt["lo"].isna()
    return bt


def diebold_mariano(err_a: pd.DataFrame, err_b: pd.DataFrame, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal accuracy (absolute-error loss).

    Diebold & Mariano (1995), with the Harvey, Leybourne & Newbold (1997)
    small-sample correction. Errors are first averaged across provinces for
    each origin week, giving one loss differential per week, so the
    cross-sectional correlation between provinces doesn't inflate the
    sample size. Long-run variance uses a rectangular window of h-1 lags,
    because h-step forecast errors overlap.

    Inputs: DataFrames with columns origin_week, abs_err (same rows in each).
    Returns (DM statistic, two-sided p-value). Negative DM = A more accurate.
    """
    from scipy import stats

    d = (err_a.groupby("origin_week")["abs_err"].mean() - err_b.groupby("origin_week")["abs_err"].mean()).dropna()
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(horizon)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm *= hln
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def backtest_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / MAPE per horizon and model, plus skill vs naive and coverage."""
    ev = bt[~bt["is_warmup"]]
    # Production intervals: CQR when the quantile models were run (it scored
    # best in interval_metrics), otherwise the scaled conformal interval.
    lo_c, hi_c = ("price_cqr_lo", "price_cqr_hi") if "price_cqr_lo" in ev.columns else ("price_lo", "price_hi")
    rows = []
    for h, sub in ev.groupby("horizon"):
        actual = sub["target_price"]
        naive_mae = (sub["price_naive"] - actual).abs().mean()
        for m, label in [("xgb", "XGBoost"), ("naive", "Naive (last value)"), ("snaive", "Seasonal naive")]:
            err = sub[f"price_{m}"] - actual
            mae = err.abs().mean()
            rows.append(
                {
                    "horizon": h,
                    "model": label,
                    "MAE": mae,
                    "RMSE": np.sqrt((err**2).mean()),
                    "MAPE": (err.abs() / actual).mean() * 100,
                    "skill_vs_naive": 1 - mae / naive_mae,
                    "direction_acc": (np.sign(sub[f"pred_{m}"]) == np.sign(sub["y"])).mean()
                    if m != "naive"
                    else np.nan,
                    "interval_coverage": (
                        ((actual >= sub[lo_c]) & (actual <= sub[hi_c])).mean()
                        if m == "xgb"
                        else np.nan
                    ),
                    "n": len(sub),
                }
            )
            if m != "naive":
                a = sub.assign(abs_err=(sub[f"price_{m}"] - actual).abs())
                b = sub.assign(abs_err=(sub["price_naive"] - actual).abs())
                rows[-1]["dm_vs_naive"], rows[-1]["dm_pvalue"] = diebold_mariano(a, b, int(h))
    return pd.DataFrame(rows)


def interval_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """Compare 80% interval methods: coverage, width and the interval score.

    Interval score (Gneiting & Raftery, 2007) for a central (1 - a) interval:
    width + (2/a)(lo - y) if y < lo + (2/a)(y - hi) if y > hi. Lower is
    better; it rewards narrow intervals but penalises misses.
    """
    a = 1 - INTERVAL_COVERAGE
    ev = bt[~bt["is_warmup"]]
    methods = [("Scaled conformal (residual quantiles)", "price_lo", "price_hi")]
    if "price_q10" in ev.columns:
        methods += [
            ("Quantile regression (raw)", "price_q10", "price_q90"),
            ("Conformalized quantile regression (CQR)", "price_cqr_lo", "price_cqr_hi"),
        ]
    rows = []
    for h, sub in ev.groupby("horizon"):
        y = sub["target_price"]
        for name, lo_c, hi_c in methods:
            lo, hi = sub[lo_c], sub[hi_c]
            ok = lo.notna() & hi.notna()
            lo, hi, yy = lo[ok], hi[ok], y[ok]
            score = (hi - lo) + (2 / a) * (lo - yy).clip(lower=0) + (2 / a) * (yy - hi).clip(lower=0)
            rows.append(
                {
                    "horizon": h,
                    "method": name,
                    "coverage": ((yy >= lo) & (yy <= hi)).mean(),
                    "mean_width_pct": ((hi - lo) / yy).mean() * 100,
                    "interval_score": score.mean(),
                    "n": int(ok.sum()),
                }
            )
        if "price_q50" in sub.columns:
            naive_mae = (sub["price_naive"] - y).abs().mean()
            for name, col in [("Point model (pseudo-Huber)", "price_xgb"), ("Quantile model median (q50)", "price_q50")]:
                rows.append(
                    {
                        "horizon": h,
                        "method": name,
                        "point_MAE": (sub[col] - y).abs().mean(),
                        "point_MAPE": ((sub[col] - y).abs() / y).mean() * 100,
                        "point_skill_vs_naive": 1 - (sub[col] - y).abs().mean() / naive_mae,
                        "n": len(sub),
                    }
                )
    return pd.DataFrame(rows)


def metrics_by_year(bt: pd.DataFrame) -> pd.DataFrame:
    """Model vs naive by calendar year of the forecast origin."""
    ev = bt[~bt["is_warmup"]].copy()
    ev["year"] = ev["origin_week"].dt.year
    rows = []
    for (h, yr), sub in ev.groupby(["horizon", "year"]):
        if sub["origin_week"].nunique() < 26:
            continue
        y = sub["target_price"]
        a = sub.assign(abs_err=(sub["price_xgb"] - y).abs())
        b = sub.assign(abs_err=(sub["price_naive"] - y).abs())
        dm, p = diebold_mariano(a, b, int(h))
        rows.append(
            {
                "horizon": h,
                "year": yr,
                "origins": sub["origin_week"].nunique(),
                "MAPE_model": (a["abs_err"] / y).mean() * 100,
                "MAPE_naive": (b["abs_err"] / y).mean() * 100,
                "skill_vs_naive": 1 - a["abs_err"].mean() / b["abs_err"].mean(),
                "dm_pvalue": p,
                "direction_acc": (np.sign(sub["pred_xgb"]) == np.sign(sub["y"])).mean(),
                "interval_coverage": ((y >= sub["price_lo"]) & (y <= sub["price_hi"])).mean(),
            }
        )
    return pd.DataFrame(rows)


def fit_final_and_forecast(
    feats: pd.DataFrame, bt: pd.DataFrame, features: list[str] = FEATURES
) -> tuple[dict[int, xgb.XGBRegressor], pd.DataFrame, dict[int, pd.DataFrame]]:
    """Train on all data and forecast every horizon from the latest week.

    Returns (models by horizon, forecast table, origin rows by horizon).
    """
    last_week = feats["week"].max()
    models, forecasts, origin_rows = {}, [], {}
    for h in HORIZONS:
        sup = make_supervised(feats, h)
        model = fit_model(sup, features)
        models[h] = model

        latest = sup[(sup["week"] == last_week)].dropna(subset=[TARGET_SERIES]).copy()
        latest["pred"] = model.predict(latest[features])

        resid = bt.loc[bt["horizon"] == h]
        norm = ((resid["y"] - resid["pred_xgb"]) / resid["scale"]).to_numpy()
        q_lo, q_hi = interval_quantiles(norm)
        scale = _scale(latest, h)
        cqr = None
        if "q10" in resid.columns:
            qm = fit_quantile_model(sup, features)
            q = predict_quantiles(qm, latest[features])
            score = np.maximum(resid["q10"] - resid["y"], resid["y"] - resid["q90"]).to_numpy()
            adj = np.nanquantile(score, INTERVAL_COVERAGE)
            cqr = (q[:, 0] - adj, q[:, 1], q[:, 2] + adj)

        out = pd.DataFrame(
            {
                "province": latest["province"].values,
                "origin_week": last_week,
                "horizon": h,
                "target_week": latest["target_week"].values,
                "last_price": latest[TARGET_SERIES].values,
                "forecast": latest[TARGET_SERIES].values * np.exp(latest["pred"].values),
                "lo": latest[TARGET_SERIES].values * np.exp(latest["pred"].values + q_lo * scale),
                "hi": latest[TARGET_SERIES].values * np.exp(latest["pred"].values + q_hi * scale),
            }
        )
        if cqr is not None:
            # Publish the CQR interval as the forecast interval; keep the
            # scaled-conformal one alongside for comparison.
            last = latest[TARGET_SERIES].values
            out["conformal_lo"], out["conformal_hi"] = out["lo"], out["hi"]
            out["lo"] = last * np.exp(cqr[0])
            out["q50"] = last * np.exp(cqr[1])
            out["hi"] = last * np.exp(cqr[2])
        out["pct_change"] = (out["forecast"] / out["last_price"] - 1) * 100
        forecasts.append(out)
        origin_rows[h] = latest
        log.info("  final model h=%d trained on %d rows", h, sup["y"].notna().sum())
    return models, pd.concat(forecasts, ignore_index=True), origin_rows
