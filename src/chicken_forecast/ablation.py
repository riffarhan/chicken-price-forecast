"""Feature-group ablation: does each literature-backed driver actually help?

For every group in FEATURE_GROUPS we run the same rolling-origin backtest
twice: with all groups, and with that one group removed ("leave one group
out"). We also run each group added on its own to the core price features
("add one group in"). A group earns its place in the production model only
if removing it makes out-of-sample error worse.

This complements the literature review: papers tell us *what could matter*,
the ablation tells us *whether it matters for forecasting this data*.
"""

from __future__ import annotations

import logging

import pandas as pd

from chicken_forecast.config import OUTPUTS_DIR
from chicken_forecast.data.build import load_weekly_panel
from chicken_forecast.features import FEATURE_GROUPS, features_for, origin_features
from chicken_forecast.models import backtest, backtest_metrics

log = logging.getLogger(__name__)

ABLATION_HORIZONS = [1, 4]


def _score(feats: pd.DataFrame, groups: list[str]) -> dict[int, float]:
    bt = backtest(feats, features_for(groups), horizons=ABLATION_HORIZONS)
    m = backtest_metrics(bt)
    m = m[m["model"] == "XGBoost"].set_index("horizon")
    return {h: float(m.loc[h, "MAE"]) for h in ABLATION_HORIZONS} | {
        f"mape_{h}": float(m.loc[h, "MAPE"]) for h in ABLATION_HORIZONS
    }


def run() -> pd.DataFrame:
    panel = load_weekly_panel()
    feats = origin_features(panel, sorted(panel["province"].unique()))
    groups = list(FEATURE_GROUPS)
    optional = [g for g in groups if g != "core_price"]

    rows = []
    base_all = _score(feats, groups)
    base_core = _score(feats, ["core_price"])
    rows.append({"experiment": "all groups", "group": "-", **base_all})
    rows.append({"experiment": "core price only", "group": "-", **base_core})
    log.info("all: %s | core: %s", base_all, base_core)

    for g in optional:
        without = _score(feats, [x for x in groups if x != g])
        only = _score(feats, ["core_price", g])
        rows.append({"experiment": "leave out", "group": g, **without})
        rows.append({"experiment": "core + group", "group": g, **only})
        log.info("%-14s without: %s | core+: %s", g, without, only)

    df = pd.DataFrame(rows)
    # Positive = removing the group increases error = the group helps
    for h in ABLATION_HORIZONS:
        df[f"mae_change_when_removed_{h}w_pct"] = None
        df[f"mae_change_when_added_{h}w_pct"] = None
        lo = df["experiment"] == "leave out"
        add = df["experiment"] == "core + group"
        df.loc[lo, f"mae_change_when_removed_{h}w_pct"] = (df.loc[lo, h] / base_all[h] - 1) * 100
        df.loc[add, f"mae_change_when_added_{h}w_pct"] = (df.loc[add, h] / base_core[h] - 1) * 100
    df = df.rename(columns={h: f"mae_{h}w" for h in ABLATION_HORIZONS})
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS_DIR / "ablation.csv", index=False, float_format="%.4f")
    log.info("\n%s", df.round(3).to_string(index=False))
    return df
