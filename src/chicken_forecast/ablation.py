"""Feature-group ablation: does each literature-backed driver actually help?

For every optional group in FEATURE_GROUPS we run the same rolling-origin
backtest in two designs:

* **leave one out**: all groups minus this one, compared with all groups
* **add one in**: core price features plus this group, compared with core only

Differences are tested with the Diebold-Mariano test, so a group's
contribution is reported with a p-value, not just a point estimate.

This complements the literature review: papers tell us *what could matter*,
the ablation tells us *whether it matters for forecasting this data*.
"""

from __future__ import annotations

import logging

import pandas as pd

from chicken_forecast.config import OUTPUTS_DIR
from chicken_forecast.data.build import load_weekly_panel
from chicken_forecast.features import FEATURE_GROUPS, features_for, origin_features
from chicken_forecast.models import backtest, diebold_mariano

log = logging.getLogger(__name__)

ABLATION_HORIZONS = [1, 4]


def _errors(feats: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    bt = backtest(feats, features_for(groups), horizons=ABLATION_HORIZONS)
    bt = bt[~bt["is_warmup"]].copy()
    bt["abs_err"] = (bt["price_xgb"] - bt["target_price"]).abs()
    bt["ape"] = bt["abs_err"] / bt["target_price"] * 100
    return bt[["province", "origin_week", "horizon", "abs_err", "ape"]]


def _compare(test: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Metrics of *test* vs *ref*; positive change = test has larger error."""
    out = {}
    for h in ABLATION_HORIZONS:
        a, b = test[test["horizon"] == h], ref[ref["horizon"] == h]
        out[f"mae_{h}w"] = a["abs_err"].mean()
        out[f"mape_{h}w"] = a["ape"].mean()
        out[f"mae_change_{h}w_pct"] = (a["abs_err"].mean() / b["abs_err"].mean() - 1) * 100
        out[f"dm_p_{h}w"] = diebold_mariano(a, b, h)[1]
    return out


def run() -> pd.DataFrame:
    panel = load_weekly_panel()
    feats = origin_features(panel, sorted(panel["province"].unique()))
    groups = list(FEATURE_GROUPS)
    optional = [g for g in groups if g != "core_price"]

    err_all = _errors(feats, groups)
    err_core = _errors(feats, ["core_price"])
    rows = [
        {"design": "reference", "group": "all groups", **_compare(err_all, err_all)},
        {"design": "reference", "group": "core price only", **_compare(err_core, err_all)},
    ]
    for g in optional:
        without = _errors(feats, [x for x in groups if x != g])
        added = _errors(feats, ["core_price", g])
        # leave-one-out: positive change = removing g hurt = g helps
        rows.append({"design": "leave one out (vs all)", "group": g, **_compare(without, err_all)})
        # add-one-in: negative change = adding g helped
        rows.append({"design": "add to core (vs core)", "group": g, **_compare(added, err_core)})
        log.info("ablation: %s done", g)

    df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS_DIR / "ablation.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200):
        log.info("\n%s", df.round(3).to_string(index=False))
    return df

