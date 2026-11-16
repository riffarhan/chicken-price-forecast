"""End-to-end run: features -> backtest -> final models -> forecast -> SHAP -> outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from chicken_forecast import __version__, hierarchy
from chicken_forecast.config import (
    BACKTEST_WEEKS,
    HORIZONS,
    MODELS_DIR,
    NATIONAL,
    OUTPUTS_DIR,
    SELECTION_WEEKS,
    TARGET_SERIES,
)
from chicken_forecast.data.build import load_weekly_panel
from chicken_forecast.features import (
    FEATURE_LABELS,
    FEATURES,
    SELECTED_GROUPS,
    make_supervised,
    origin_features,
)
from chicken_forecast.models import (
    _scale,
    backtest,
    backtest_metrics,
    fit_final_and_forecast,
    interval_metrics,
    metrics_by_year,
)

log = logging.getLogger(__name__)

TOP_DRIVERS = 6


def explain(models: dict[int, xgb.XGBRegressor], origin_rows: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Top SHAP drivers per province and horizon, in % price-change terms."""
    rows = []
    for h, model in models.items():
        X = origin_rows[h][FEATURES]
        sv = shap.TreeExplainer(model).shap_values(X)
        provinces = origin_rows[h]["province"].to_numpy()
        for i, prov in enumerate(provinces):
            order = np.argsort(-np.abs(sv[i]))[:TOP_DRIVERS]
            for rank, j in enumerate(order, start=1):
                rows.append(
                    {
                        "province": prov,
                        "horizon": h,
                        "rank": rank,
                        "feature": FEATURES[j],
                        "label": FEATURE_LABELS.get(FEATURES[j], FEATURES[j]),
                        "feature_value": float(X.iloc[i, j]),
                        "impact_pct": float(sv[i][j]) * 100,  # log-change ~ % change
                    }
                )
    return pd.DataFrame(rows)


def global_importance(models: dict[int, xgb.XGBRegressor], origin_rows) -> pd.DataFrame:
    rows = []
    for h, model in models.items():
        sv = shap.TreeExplainer(model).shap_values(origin_rows[h][FEATURES])
        mean_abs = np.abs(sv).mean(axis=0) * 100
        for f, v in zip(FEATURES, mean_abs, strict=False):
            rows.append({"horizon": h, "feature": f, "label": FEATURE_LABELS.get(f, f), "mean_abs_impact_pct": v})
    return pd.DataFrame(rows).sort_values(["horizon", "mean_abs_impact_pct"], ascending=[True, False])


def national_forecast(models, nat_feats, forecast, origin_rows, panel, nat_panel, provinces) -> pd.DataFrame:
    """Direct, bottom-up and reconciled national forecasts from the latest week."""
    last_week = nat_feats["week"].max()
    w = hierarchy.estimate_weights(panel, nat_panel, last_week, provinces)
    rows = []
    for h, model in models.items():
        sup = make_supervised(nat_feats, h)
        row = sup[sup["week"] == last_week]
        if row.empty:
            continue
        last = float(row[TARGET_SERIES].iloc[0])
        direct = last * float(np.exp(model.predict(row[FEATURES])[0]))
        fh = forecast[forecast["horizon"] == h].set_index("province").reindex(provinces)
        base = fh["forecast"].to_numpy()
        orow = origin_rows[h].set_index("province").reindex(provinces)
        var_p = (_scale(orow, h) * base) ** 2  # same variance proxy as in the backtest
        var_n = (float(_scale(row, h)[0]) * direct) ** 2
        rec = hierarchy.reconcile(base, var_p, direct, var_n, w)
        rows.append(
            {
                "horizon": h, "origin_week": last_week, "target_week": row["target_week"].iloc[0],
                "last_price": last, "direct": direct, "bottom_up": float(w @ base),
                "reconciled": float(w @ rec),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_weekly_panel()
    provinces = sorted(panel["province"].unique())
    log.info("Engineering features for %d provinces", len(provinces))
    feats = origin_features(panel, provinces)

    log.info("Building the national series")
    nat_panel = hierarchy.build_national_panel(panel)
    nat_feats = hierarchy.national_features(origin_features(nat_panel, [NATIONAL]))

    log.info("Running rolling-origin backtest (%d weeks, with quantile models)", BACKTEST_WEEKS)
    bt = backtest(feats, quantiles=True, extra_feats=nat_feats)
    metrics = backtest_metrics(bt)
    log.info("\n%s", metrics.round(3).to_string(index=False))

    intervals = interval_metrics(bt)
    log.info("\n%s", intervals.round(3).to_string(index=False))
    by_year = metrics_by_year(bt)
    log.info("\n%s", by_year.round(3).to_string(index=False))

    log.info("Evaluating national and reconciled forecasts")
    hier_metrics, nat_bt = hierarchy.evaluate(bt, panel, nat_panel)
    wfit = hierarchy.weight_fit(panel, nat_panel)
    log.info("National weights: %s", wfit)

    log.info("Training final models and forecasting")
    models, forecast, origin_rows = fit_final_and_forecast(feats, bt)
    for h, m in models.items():
        m.save_model(MODELS_DIR / f"xgb_h{h}.json")

    log.info("Computing SHAP explanations")
    drivers = explain(models, origin_rows)
    importance = global_importance(models, origin_rows)

    # Feature groups were chosen on the most recent 52 origins, so report the
    # earlier origins separately: they were never used for any modelling decision.
    split = bt.groupby("horizon")["origin_week"].transform("max") - pd.Timedelta(weeks=SELECTION_WEEKS)
    recent = bt["origin_week"] > split
    by_period = pd.concat(
        [
            backtest_metrics(bt[~recent]).assign(period="untouched hold-out (older origins)"),
            backtest_metrics(bt[recent]).assign(period=f"selection window (last {SELECTION_WEEKS} origins)"),
        ]
    )
    by_period.to_csv(OUTPUTS_DIR / "backtest_metrics_by_period.csv", index=False, float_format="%.4f")
    by_year.to_csv(OUTPUTS_DIR / "backtest_metrics_by_year.csv", index=False, float_format="%.4f")
    intervals.to_csv(OUTPUTS_DIR / "interval_metrics.csv", index=False, float_format="%.4f")
    hier_metrics.to_csv(OUTPUTS_DIR / "hierarchy_metrics.csv", index=False, float_format="%.4f")
    nat_bt.to_csv(OUTPUTS_DIR / "national_backtest.csv", index=False, float_format="%.2f")
    national_forecast(models, nat_feats, forecast, origin_rows, panel, nat_panel, provinces).to_csv(
        OUTPUTS_DIR / "forecast_national.csv", index=False, float_format="%.2f"
    )

    bt.to_csv(OUTPUTS_DIR / "backtest_predictions.csv", index=False, float_format="%.4f")
    metrics.to_csv(OUTPUTS_DIR / "backtest_metrics.csv", index=False, float_format="%.4f")
    forecast.to_csv(OUTPUTS_DIR / "forecast_latest.csv", index=False, float_format="%.2f")
    drivers.to_csv(OUTPUTS_DIR / "forecast_drivers.csv", index=False, float_format="%.4f")
    importance.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False, float_format="%.4f")

    xgb_m = metrics[metrics["model"] == "XGBoost"].set_index("horizon")
    meta = {
        "version": __version__,
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_first_week": str(panel["week"].min().date()),
        "data_last_week": str(panel["week"].max().date()),
        "n_provinces": len(provinces),
        "n_weekly_rows": int(len(panel)),
        "target": TARGET_SERIES,
        "horizons": HORIZONS,
        "backtest_weeks": BACKTEST_WEEKS,
        "feature_groups": SELECTED_GROUPS,
        "backtest_mape": {int(h): round(float(v), 3) for h, v in xgb_m["MAPE"].items()},
        "backtest_skill_vs_naive": {int(h): round(float(v), 3) for h, v in xgb_m["skill_vs_naive"].items()},
        "features": FEATURES,
        "national_weights": wfit,
    }
    (OUTPUTS_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Outputs written to %s", OUTPUTS_DIR)
    return meta
