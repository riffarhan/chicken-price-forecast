"""National forecasts and province-national reconciliation.

PIHPS publishes a national average ("Semua Provinsi") next to the 34
provinces, but its weights are not published: it is not a simple mean or sum
of the provinces. This module:

1. Builds a weekly panel for the national series (same four price levels;
   weather = mean over province capitals).
2. Estimates the implied province weights with non-negative least squares
   (NNLS) on the 104 weeks before each refit date, so no future data is used.
3. Produces three national forecasts per origin and horizon:
   * **direct**: the global province model applied to the national series
     (it never trains on it);
   * **bottom-up**: the weighted sum of the 34 province forecasts;
   * **reconciled**: a weighted-least-squares (WLS) reconciliation of the
     direct national and the 34 province forecasts, so that the national
     forecast equals the weighted sum of the province forecasts. Each
     forecast is weighted by the inverse of its expected error variance
     (recent volatility x sqrt(h) x price), in the spirit of the structural
     scaling of Hyndman et al. (2011) and Wickramasuriya et al. (2019).
4. Evaluates all of them against naive, for the national series and, after
   reconciliation, for the provinces.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from chicken_forecast.config import NATIONAL, PIHPS_SERIES, TARGET_SERIES
from chicken_forecast.data import pihps
from chicken_forecast.data.build import clean_daily, feed_drivers_weekly, to_weekly, weather_weekly

log = logging.getLogger(__name__)

WEIGHT_WINDOW_WEEKS = 104


def build_national_panel(province_panel: pd.DataFrame) -> pd.DataFrame:
    """Weekly panel for the PIHPS national average, same columns as the province panel."""
    nat = None
    for name in PIHPS_SERIES:
        daily = pihps.load_series(name)
        daily = clean_daily(daily[daily["province"] == NATIONAL])
        weekly = to_weekly(daily, name)
        nat = weekly if nat is None else nat.merge(weekly, on=["province", "week"], how="outer")
    nat = nat[nat["week"] <= province_panel["week"].max()]
    nat["national_retail"] = nat[TARGET_SERIES]
    weeks = pd.DatetimeIndex(sorted(nat["week"].unique()))
    nat = nat.merge(feed_drivers_weekly(weeks), on="week", how="left")
    w = weather_weekly().groupby("week")[["rain_mm", "tmax_c", "hot_days"]].mean().reset_index()
    nat = nat.merge(w, on="week", how="left")
    return nat.dropna(subset=[TARGET_SERIES]).sort_values("week").reset_index(drop=True)


def national_features(nat_feats: pd.DataFrame) -> pd.DataFrame:
    """The national series is unseen by the province model: no province code."""
    out = nat_feats.copy()
    out["province_code"] = np.nan
    return out


def estimate_weights(panel: pd.DataFrame, nat: pd.DataFrame, cutoff: pd.Timestamp,
                     provinces: list[str]) -> np.ndarray:
    """NNLS weights so that national ~= sum_i w_i * province_i, on data up to cutoff."""
    wide = panel.pivot(index="week", columns="province", values=TARGET_SERIES).reindex(columns=provinces)
    y = nat.set_index("week")[TARGET_SERIES]
    idx = wide.index[(wide.index <= cutoff)].intersection(y.index)[-WEIGHT_WINDOW_WEEKS:]
    X = wide.loc[idx].ffill().bfill()
    ok = X.notna().all(axis=1) & y.loc[idx].notna()
    w, _ = nnls(X[ok].to_numpy(), y.loc[idx][ok].to_numpy())
    return w


def reconcile(base_prov: np.ndarray, var_prov: np.ndarray, base_nat: float, var_nat: float,
              w: np.ndarray) -> np.ndarray:
    """WLS reconciliation of [national; provinces] onto national = w'provinces.

    S = [w'; I] maps provinces to all series. Reconciled provinces
    b~ = (S' L S)^-1 S' L yhat with L = diag(1/variances).
    """
    S = np.vstack([w[None, :], np.eye(len(w))])
    lam = np.concatenate([[1.0 / var_nat], 1.0 / var_prov])
    yhat = np.concatenate([[base_nat], base_prov])
    A = S.T @ (lam[:, None] * S)
    b = S.T @ (lam * yhat)
    return np.linalg.solve(A, b)


def evaluate(bt: pd.DataFrame, panel: pd.DataFrame, nat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """National and province accuracy of direct, bottom-up and reconciled forecasts."""
    from chicken_forecast.models import diebold_mariano

    ex = bt.attrs.get("extra")
    if ex is None or ex.empty:
        raise ValueError("backtest has no national predictions (pass extra_feats)")
    provinces = sorted(panel["province"].unique())
    ev = bt[~bt["is_warmup"]]
    weights_cache: dict[pd.Timestamp, np.ndarray] = {}

    nat_rows, prov_rows = [], []
    for (h, o), grp in ev.groupby(["horizon", "origin_week"]):
        nrow = ex[(ex["horizon"] == h) & (ex["origin_week"] == o)]
        if len(nrow) != 1:
            continue
        nrow = nrow.iloc[0]
        g = grp.set_index("province").reindex(provinces)
        if g["price_xgb"].isna().any() or g["target_price"].isna().any():
            continue
        # Weights from data known at the origin (refreshed every 13 weeks)
        key = o - pd.Timedelta(days=(o - pd.Timestamp("2000-01-02")).days % 91)
        if key not in weights_cache:
            weights_cache[key] = estimate_weights(panel, nat, key, provinces)
        w = weights_cache[key]

        base = g["price_xgb"].to_numpy()
        var_p = (g["scale"].to_numpy() * base) ** 2
        var_n = (nrow["scale"] * nrow["price_xgb"]) ** 2
        rec = reconcile(base, var_p, nrow["price_xgb"], var_n, w)

        nat_rows.append(
            {
                "horizon": h, "origin_week": o, "actual": nrow["target_price"],
                "naive": nrow["last_price"], "direct": nrow["price_xgb"],
                "bottom_up": float(w @ base), "reconciled": float(w @ rec),
                "weight_sum": float(w.sum()),
            }
        )
        prov_rows.append(
            pd.DataFrame(
                {
                    "horizon": h, "origin_week": o, "province": provinces,
                    "actual": g["target_price"].to_numpy(), "naive": g["last_price"].to_numpy(),
                    "base": base, "reconciled": rec,
                }
            )
        )

    nat_bt = pd.DataFrame(nat_rows)
    prov_bt = pd.concat(prov_rows, ignore_index=True)

    rows = []
    for level, df, methods in [
        ("national", nat_bt, ["naive", "direct", "bottom_up", "reconciled"]),
        ("provinces", prov_bt, ["naive", "base", "reconciled"]),
    ]:
        for h, sub in df.groupby("horizon"):
            naive_err = sub.assign(abs_err=(sub["naive"] - sub["actual"]).abs())
            for m in methods:
                err = sub.assign(abs_err=(sub[m] - sub["actual"]).abs())
                dm, p = diebold_mariano(err, naive_err, int(h)) if m != "naive" else (np.nan, np.nan)
                rows.append(
                    {
                        "level": level, "horizon": h, "method": m,
                        "MAE": err["abs_err"].mean(),
                        "MAPE": (err["abs_err"] / sub["actual"]).mean() * 100,
                        "skill_vs_naive": 1 - err["abs_err"].mean() / naive_err["abs_err"].mean(),
                        "dm_pvalue_vs_naive": p,
                        "n": len(sub),
                    }
                )
    metrics = pd.DataFrame(rows)
    log.info("\n%s", metrics.round(3).to_string(index=False))
    return metrics, nat_bt


def weight_fit(panel: pd.DataFrame, nat: pd.DataFrame) -> dict:
    """How well a weighted sum of provinces reproduces the published national average."""
    provinces = sorted(panel["province"].unique())
    cutoff = panel["week"].max()
    w = estimate_weights(panel, nat, cutoff, provinces)
    wide = panel.pivot(index="week", columns="province", values=TARGET_SERIES).reindex(columns=provinces)
    y = nat.set_index("week")[TARGET_SERIES]
    idx = wide.index.intersection(y.index)[-WEIGHT_WINDOW_WEEKS:]
    fit = wide.loc[idx].ffill().bfill().to_numpy() @ w
    ape = np.abs(fit - y.loc[idx].to_numpy()) / y.loc[idx].to_numpy() * 100
    top = sorted(zip(provinces, w, strict=True), key=lambda t: -t[1])[:5]
    return {"weight_sum": float(w.sum()), "fit_mape_pct": float(ape.mean()),
            "top_weights": {p: round(float(v), 3) for p, v in top}}
