"""Tests for reconciliation, the DM test and interval metrics."""

import numpy as np
import pandas as pd
import pytest

from chicken_forecast.hierarchy import reconcile
from chicken_forecast.models import diebold_mariano, interval_metrics


def test_reconciled_forecasts_are_coherent():
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(5))
    prov = rng.uniform(30000, 45000, 5)
    nat = float(w @ prov) * 1.03  # direct national disagrees with the provinces
    rec = reconcile(prov, np.full(5, 1e6), nat, 1e6, w)
    # the reconciled national is exactly the weighted sum of reconciled provinces,
    # and it lies between the bottom-up and the direct national forecast
    rec_nat = w @ rec
    assert min(w @ prov, nat) <= rec_nat <= max(w @ prov, nat)


def test_reconcile_trusts_the_more_certain_forecast():
    w = np.array([0.5, 0.5])
    prov = np.array([100.0, 100.0])
    rec_trust_nat = reconcile(prov, np.array([1e6, 1e6]), 110.0, 1e-6, w)
    rec_trust_prov = reconcile(prov, np.array([1e-6, 1e-6]), 110.0, 1e6, w)
    assert w @ rec_trust_nat == pytest.approx(110.0, rel=1e-4)
    assert w @ rec_trust_prov == pytest.approx(100.0, rel=1e-4)


def test_diebold_mariano_detects_a_clear_difference():
    rng = np.random.default_rng(1)
    weeks = pd.date_range("2025-01-05", periods=60, freq="7D")
    a = pd.DataFrame({"origin_week": np.repeat(weeks, 10), "abs_err": rng.gamma(2, 50, 600)})
    worse = a.assign(abs_err=a["abs_err"] * 1.3)
    same = a.assign(abs_err=a["abs_err"] + rng.normal(0, 1, 600))
    assert diebold_mariano(a, worse, 1)[1] < 0.01
    assert diebold_mariano(a, same, 1)[1] > 0.05


def test_interval_metrics_score_penalises_misses():
    bt = pd.DataFrame(
        {
            "horizon": [1, 1],
            "is_warmup": [False, False],
            "target_price": [100.0, 100.0],
            "price_lo": [90.0, 105.0],  # first covers, second misses by 5
            "price_hi": [110.0, 115.0],
        }
    )
    m = interval_metrics(bt).iloc[0]
    assert m["coverage"] == 0.5
    # widths 20 and 10, plus (2/0.2) * 5 = 50 penalty on the miss
    assert m["interval_score"] == pytest.approx((20 + 10 + 50) / 2)
