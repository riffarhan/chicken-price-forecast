"""Small, transparent hyper-parameter comparison on the rolling-origin backtest.

Not a big search (that would overfit the 52-week evaluation window): a
handful of regularisation settings, chosen up front, evaluated at h=1 and 4.

    python scripts/tune.py "core_price,calendar,supply_chain"
"""

import logging
import sys

import numpy as np
import pandas as pd

from chicken_forecast import config, models
from chicken_forecast.data.build import load_weekly_panel
from chicken_forecast.features import features_for, origin_features

logging.basicConfig(level=logging.WARNING)

groups = sys.argv[1].split(",") if len(sys.argv) > 1 else ["core_price", "calendar"]
panel = load_weekly_panel()
feats = origin_features(panel, sorted(panel["province"].unique()))
features = features_for(groups)
base = dict(config.XGB_PARAMS)

CANDIDATES = {
    "current": {},
    "shallow_strong_reg": {"max_depth": 3, "min_child_weight": 50, "n_estimators": 400,
                           "reg_lambda": 10.0, "colsample_bytree": 0.6},
    "very_shallow": {"max_depth": 2, "min_child_weight": 100, "n_estimators": 500,
                     "learning_rate": 0.02, "reg_lambda": 10.0, "colsample_bytree": 0.6},
    "squared_error_reg": {"objective": "reg:squarederror", "max_depth": 3, "min_child_weight": 50,
                          "n_estimators": 400, "reg_lambda": 10.0, "colsample_bytree": 0.6},
}

rows = []
for name, override in CANDIDATES.items():
    params = {**base, **override}
    if params["objective"] != "reg:pseudohubererror":
        params.pop("huber_slope", None)
    models.XGB_PARAMS.clear()
    models.XGB_PARAMS.update(params)
    bt = models.backtest(feats, features, horizons=[1, 4])
    m = models.backtest_metrics(bt)
    for _, r in m.iterrows():
        rows.append({"config": name, **r.to_dict()})
    # Blend with naive: shrink the predicted change toward zero
    for w in [0.5]:
        ev = bt[~bt["is_warmup"]]
        for h, sub in ev.groupby("horizon"):
            pred = sub["last_price"] * np.exp(sub["pred_xgb"] * w)
            err = (pred - sub["target_price"]).abs()
            rows.append({"config": f"{name}+blend{w}", "horizon": h, "model": "XGBoost",
                         "MAE": err.mean(), "MAPE": (err / sub["target_price"]).mean() * 100})
    print(name, "done", flush=True)

out = pd.DataFrame(rows)
out = out[out["model"].isin(["XGBoost", "Naive (last value)", "Seasonal naive"])]
print(out.pivot_table(index=["config", "model"], columns="horizon", values="MAPE").round(3).to_string())
