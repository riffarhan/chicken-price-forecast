"""End-to-end run: features -> backtest -> final models -> forecast -> SHAP -> outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from chicken_forecast import __version__
from chicken_forecast.config import BACKTEST_WEEKS, HORIZONS, MODELS_DIR, OUTPUTS_DIR, TARGET_SERIES
from chicken_forecast.data.build import load_weekly_panel
from chicken_forecast.features import FEATURE_LABELS, FEATURES, SELECTED_GROUPS, origin_features
from chicken_forecast.models import backtest, backtest_metrics, fit_final_and_forecast

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


def run() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_weekly_panel()
    provinces = sorted(panel["province"].unique())
    log.info("Engineering features for %d provinces", len(provinces))
    feats = origin_features(panel, provinces)

    log.info("Running rolling-origin backtest")
    bt = backtest(feats)
    metrics = backtest_metrics(bt)
    log.info("\n%s", metrics.round(3).to_string(index=False))

    log.info("Training final models and forecasting")
    models, forecast, origin_rows = fit_final_and_forecast(feats, bt)
    for h, m in models.items():
        m.save_model(MODELS_DIR / f"xgb_h{h}.json")

    log.info("Computing SHAP explanations")
    drivers = explain(models, origin_rows)
    importance = global_importance(models, origin_rows)

    # Feature groups were chosen on the most recent 52 origins, so report the
    # earlier origins separately: they were never used for any modelling decision.
    split = bt["origin_week"].max() - pd.Timedelta(weeks=52)
    by_period = pd.concat(
        [
            backtest_metrics(bt[bt["origin_week"] <= split]).assign(period="untouched hold-out (older origins)"),
            backtest_metrics(bt[bt["origin_week"] > split]).assign(period="selection window (last 52 origins)"),
        ]
    )
    by_period.to_csv(OUTPUTS_DIR / "backtest_metrics_by_period.csv", index=False, float_format="%.4f")

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
    }
    (OUTPUTS_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Outputs written to %s", OUTPUTS_DIR)
    return meta
