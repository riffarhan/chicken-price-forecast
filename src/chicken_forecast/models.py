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
) -> pd.DataFrame:
    """Rolling-origin evaluation over the last BACKTEST_WEEKS (+ warm-up) weeks.

    For each block of REFIT_EVERY origins the model is retrained using only
    rows whose *target* week is on or before the block's first origin, i.e.
    exactly the information available at that time.
    """
    last_week = feats["week"].max()
    results = []
    for h in horizons:
        sup = make_supervised(feats, h)
        last_origin = last_week - pd.Timedelta(weeks=h)
        n_origins = BACKTEST_WEEKS + CALIBRATION_WARMUP
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
            results.append(test)
        log.info("  backtest h=%d done", h)

    bt = pd.concat(results, ignore_index=True)
    bt["pred_naive"] = 0.0
    bt = _sequential_intervals(bt, horizons)

    keep = [
        "province", "week", "target_week", "horizon", TARGET_SERIES, "target_price",
        "y", "pred_xgb", "pred_naive", "pred_snaive", "scale", "lo", "hi", "is_warmup",
    ]
    bt = bt[keep].rename(columns={"week": "origin_week", TARGET_SERIES: "last_price"})
    for m in ["xgb", "naive", "snaive"]:
        bt[f"price_{m}"] = bt["last_price"] * np.exp(bt[f"pred_{m}"])
    bt["price_lo"] = bt["last_price"] * np.exp(bt["lo"])
    bt["price_hi"] = bt["last_price"] * np.exp(bt["hi"])
    return bt.sort_values(["horizon", "province", "origin_week"]).reset_index(drop=True)


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
    for h in horizons:
        sub = bt[bt["horizon"] == h].sort_values("week")
        first_week = sub["week"].min()
        warmup_end = first_week + pd.Timedelta(weeks=CALIBRATION_WARMUP)
        for o in sub["week"].unique():
            if o < warmup_end:
                continue
            known = sub[sub["target_week"] <= o]["norm_resid"].to_numpy()
            if len(known) < 100:
                continue
            q_lo, q_hi = interval_quantiles(known)
            idx = sub.index[sub["week"] == o]
            bt.loc[idx, "lo"] = bt.loc[idx, "pred_xgb"] + q_lo * bt.loc[idx, "scale"]
            bt.loc[idx, "hi"] = bt.loc[idx, "pred_xgb"] + q_hi * bt.loc[idx, "scale"]
    bt["is_warmup"] = bt["lo"].isna()
    return bt


def backtest_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / MAPE per horizon and model, plus skill vs naive and coverage."""
    ev = bt[~bt["is_warmup"]]
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
                        ((actual >= sub["price_lo"]) & (actual <= sub["price_hi"])).mean()
                        if m == "xgb"
                        else np.nan
                    ),
                    "n": len(sub),
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
        out["pct_change"] = (out["forecast"] / out["last_price"] - 1) * 100
        forecasts.append(out)
        origin_rows[h] = latest
        log.info("  final model h=%d trained on %d rows", h, sup["y"].notna().sum())
    return models, pd.concat(forecasts, ignore_index=True), origin_rows
