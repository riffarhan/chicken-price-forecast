"""Streamlit dashboard: weekly chicken price forecasts for 34 Indonesian provinces.

Run with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicken_forecast.config import OUTPUTS_DIR, WEEKLY_PANEL_PATH  # noqa: E402

LOCATIONS_PATH = ROOT / "data" / "raw" / "weather" / "locations.csv"

st.set_page_config(page_title="Chicken Price Forecast · Indonesia", page_icon="🐔", layout="wide")

# Colours: one accent for the model, neutral for actuals, red/green for direction.
# The "actual" line must stay visible in both light and dark themes.
_DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
C_ACTUAL = "#e5e7eb" if _DARK else "#1f2937"
C_MODEL = "#60a5fa" if _DARK else "#2563eb"
C_BAND = "rgba(96, 165, 250, 0.22)" if _DARK else "rgba(37, 99, 235, 0.15)"
C_NEUTRAL = "#6b7280" if _DARK else "#9ca3af"
C_PRODUCER = "#d97706"
C_WHOLESALE = "#7c3aed"
C_UP, C_DOWN = "#dc2626", "#16a34a"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load():
    need = ["forecast_latest.csv", "backtest_metrics.csv", "backtest_predictions.csv",
            "forecast_drivers.csv", "feature_importance.csv", "run_metadata.json"]
    if not all((OUTPUTS_DIR / f).exists() for f in need) or not WEEKLY_PANEL_PATH.exists():
        return None
    return {
        "panel": pd.read_csv(WEEKLY_PANEL_PATH, parse_dates=["week"]),
        "forecast": pd.read_csv(OUTPUTS_DIR / "forecast_latest.csv", parse_dates=["origin_week", "target_week"]),
        "metrics": pd.read_csv(OUTPUTS_DIR / "backtest_metrics.csv"),
        "backtest": pd.read_csv(OUTPUTS_DIR / "backtest_predictions.csv", parse_dates=["origin_week", "target_week"]),
        "drivers": pd.read_csv(OUTPUTS_DIR / "forecast_drivers.csv"),
        "importance": pd.read_csv(OUTPUTS_DIR / "feature_importance.csv"),
        "meta": json.loads((OUTPUTS_DIR / "run_metadata.json").read_text()),
        "locations": pd.read_csv(LOCATIONS_PATH) if LOCATIONS_PATH.exists() else None,
    }


D = load()
if D is None:
    st.error("No model outputs found. Run the pipeline first:")
    st.code("pip install -e .\nchicken-forecast all", language="bash")
    st.stop()

panel, fc, metrics, bt, drivers, importance, meta = (
    D["panel"], D["forecast"], D["metrics"], D["backtest"], D["drivers"], D["importance"], D["meta"]
)
provinces = sorted(fc["province"].unique())
horizons = sorted(fc["horizon"].unique())


def rp(x: float) -> str:
    return f"Rp{x:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    default_prov = provinces.index("DKI Jakarta") if "DKI Jakarta" in provinces else 0
    prov = st.selectbox("Province", provinces, index=default_prov)
    horizon = st.select_slider("Forecast horizon (weeks ahead)", options=horizons, value=4)
    history_weeks = st.slider("History shown (weeks)", 26, 400, 104, step=13)
    st.divider()
    st.caption(
        f"Data through week ending **{meta['data_last_week']}** · "
        f"{meta['n_provinces']} provinces · model run {meta['run_at_utc'][:10]}"
    )
    st.caption(
        "Prices: Bank Indonesia PIHPS, traditional markets, fresh broiler meat (Rp/kg)."
    )

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("🐔 Chicken Price Forecast — Indonesia")
st.markdown(
    "Weekly forecasts of retail broiler chicken prices (*daging ayam ras segar*) for every "
    "province, built on **real Bank Indonesia PIHPS data** (2018 → today), supply-chain prices, "
    "the Ramadan/Lebaran calendar and heat stress, with every feature backed by published research "
    "and an out-of-sample test."
)

m_h = metrics[(metrics["horizon"] == horizon)].set_index("model")
fc_h = fc[fc["horizon"] == horizon]
row = fc_h[fc_h["province"] == prov].iloc[0]
nat_last = panel.groupby("week")["national_retail"].first().dropna()

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{prov} · latest week", rp(row["last_price"]))
k2.metric(
    f"Forecast for w/e {row['target_week']:%d %b %Y}",
    rp(row["forecast"]),
    f"{row['pct_change']:+.1f}%",
    delta_color="inverse",
)
k3.metric(
    f"Backtest MAPE ({horizon}w ahead)",
    f"{m_h.loc['XGBoost', 'MAPE']:.2f}%",
    f"naive: {m_h.loc['Naive (last value)', 'MAPE']:.2f}%",
    delta_color="off",
)
_p = m_h.loc["XGBoost"].get("dm_pvalue", float("nan"))
k4.metric(
    "Error reduction vs naive",
    f"{m_h.loc['XGBoost', 'skill_vs_naive'] * 100:.1f}%",
    f"DM test p = {_p:.3f}" if _p == _p else None,
    delta_color="off",
    help=f"1 − MAE(model) / MAE(naive 'price stays the same' forecast), last {meta.get('backtest_weeks', 52)} weeks, all provinces.",
)

# Reliability notice, based on the untouched hold-out period of the backtest
_bp_path = OUTPUTS_DIR / "backtest_metrics_by_period.csv"
if _bp_path.exists():
    _bp = pd.read_csv(_bp_path)
    _ho = _bp[(_bp["period"].str.startswith("untouched")) & (_bp["model"] == "XGBoost")
              & (_bp["horizon"] == horizon)]
    if len(_ho):
        _skill, _pv = float(_ho["skill_vs_naive"].iloc[0]), float(_ho["dm_pvalue"].iloc[0])
        if _skill > 0 and _pv < 0.05:
            st.success(f"**Reliable horizon.** On older data never used to build the model, {horizon}-week "
                       f"forecasts had {_skill * 100:.1f}% lower error than 'price stays the same' "
                       f"(p = {_pv:.3f}).")
        else:
            st.warning(f"**Indicative only.** On older data never used to build the model, {horizon}-week "
                       f"forecasts were not reliably better than 'price stays the same' "
                       f"({-_skill * 100:+.1f}% error, p = {_pv:.2f}). Use the interval, not the point.")

tab_fc, tab_map, tab_drv, tab_acc, tab_data = st.tabs(
    ["📈 Forecast", "🗺️ All provinces", "🔍 Why?", "🎯 Accuracy", "📚 Data"]
)

# ---------------------------------------------------------------------------
# Forecast tab
# ---------------------------------------------------------------------------
with tab_fc:
    hist = panel[panel["province"] == prov].tail(history_weeks)
    fp = fc[fc["province"] == prov].sort_values("horizon")
    origin = fp["origin_week"].iloc[0]
    last_price = fp["last_price"].iloc[0]

    fig = go.Figure()
    # Interval band (anchored at the last observation)
    band_x = [origin] + fp["target_week"].tolist()
    fig.add_trace(go.Scatter(x=band_x + band_x[::-1],
                             y=[last_price] + fp["hi"].tolist() + fp["lo"].tolist()[::-1] + [last_price],
                             fill="toself", fillcolor=C_BAND, line=dict(width=0), mode="lines",
                             hoverinfo="skip", name="80% interval"))
    fig.add_trace(go.Scatter(x=hist["week"], y=hist["chicken_retail"], name="Actual (retail)",
                             line=dict(color=C_ACTUAL, width=2)))
    # Past one-week-ahead backtest predictions, to show how the model tracked reality
    bp = bt[(bt["province"] == prov) & (bt["horizon"] == 1)]
    bp = bp[bp["target_week"] >= hist["week"].min()]
    fig.add_trace(go.Scatter(x=bp["target_week"], y=bp["price_xgb"], name="Backtest (1w ahead)",
                             line=dict(color=C_MODEL, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=band_x, y=[last_price] + fp["forecast"].tolist(), name="Forecast",
                             mode="lines+markers", line=dict(color=C_MODEL, width=3),
                             marker=dict(size=8)))
    fig.update_layout(height=460, hovermode="x unified", yaxis_title="Rp / kg",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=30, l=10, r=10, b=10))
    fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    tbl = fp[["horizon", "target_week", "forecast", "lo", "hi", "pct_change"]].copy()
    tbl.columns = ["Weeks ahead", "Week ending", "Forecast (Rp/kg)", "Low (10%)", "High (90%)", "Change %"]
    st.dataframe(
        tbl.style.format({"Week ending": "{:%d %b %Y}", "Forecast (Rp/kg)": "{:,.0f}",
                          "Low (10%)": "{:,.0f}", "High (90%)": "{:,.0f}", "Change %": "{:+.1f}%"}),
        hide_index=True, use_container_width=True,
    )

    with st.expander("Retail vs wholesale vs farm-gate prices"):
        h2 = hist.melt(id_vars="week", value_vars=["chicken_retail", "chicken_wholesale", "chicken_producer"],
                       var_name="level", value_name="price").dropna()
        h2["level"] = h2["level"].map({"chicken_retail": "Retail (traditional market)",
                                       "chicken_wholesale": "Wholesale", "chicken_producer": "Producer"})
        f2 = px.line(h2, x="week", y="price", color="level",
                     color_discrete_sequence=[C_ACTUAL, C_WHOLESALE, C_PRODUCER])
        f2.update_layout(height=340, yaxis_title="Rp / kg", legend_title=None, margin=dict(t=10))
        st.plotly_chart(f2, use_container_width=True)

# ---------------------------------------------------------------------------
# Map / all provinces tab
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader(f"Expected change over the next {horizon} week(s)")
    allp = fc_h.copy()
    if D["locations"] is not None:
        geo = allp.merge(D["locations"][["province", "latitude", "longitude", "city"]], on="province")
        geo["size"] = geo["forecast"] / geo["forecast"].max() * 30
        figm = px.scatter_geo(geo, lat="latitude", lon="longitude", color="pct_change", size="size",
                              hover_name="province",
                              hover_data={"forecast": ":,.0f", "last_price": ":,.0f", "pct_change": ":+.1f",
                                          "size": False, "latitude": False, "longitude": False},
                              color_continuous_scale=[[0, C_DOWN], [0.5, "#9ca3af"], [1, C_UP]],
                              color_continuous_midpoint=0, labels={"pct_change": "Change %"})
        figm.update_geos(fitbounds="locations", showcountries=True, showland=True,
                         landcolor="#374151" if _DARK else "#f3f4f6",
                         countrycolor="#6b7280" if _DARK else "#d1d5db", showocean=True,
                         oceancolor="#111827" if _DARK else "#eff6ff", bgcolor="rgba(0,0,0,0)")
        figm.update_layout(height=430, margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(figm, use_container_width=True)

    allp = allp.sort_values("pct_change")
    figb = px.bar(allp, x="pct_change", y="province", orientation="h",
                  color=allp["pct_change"] > 0, color_discrete_map={True: C_UP, False: C_DOWN},
                  hover_data={"forecast": ":,.0f", "last_price": ":,.0f"})
    figb.update_layout(height=820, showlegend=False, xaxis_title="Forecast change %", yaxis_title=None,
                       margin=dict(t=10))
    st.plotly_chart(figb, use_container_width=True)

    out = allp[["province", "last_price", "forecast", "lo", "hi", "pct_change"]].sort_values("forecast", ascending=False)
    st.download_button("Download all forecasts (CSV)", fc.to_csv(index=False), "chicken_forecast_latest.csv",
                       "text/csv")

# ---------------------------------------------------------------------------
# Drivers tab
# ---------------------------------------------------------------------------
with tab_drv:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"What drives the {prov} forecast")
        d = drivers[(drivers["province"] == prov) & (drivers["horizon"] == horizon)].sort_values("impact_pct")
        figd = px.bar(d, x="impact_pct", y="label", orientation="h",
                      color=d["impact_pct"] > 0, color_discrete_map={True: C_UP, False: C_DOWN})
        figd.update_layout(height=380, showlegend=False, xaxis_title="Push on forecast (percentage points)",
                           yaxis_title=None, margin=dict(t=10))
        st.plotly_chart(figd, use_container_width=True)
        st.caption("SHAP values: how much each factor moves this forecast up (red) or down (green) "
                   "relative to the model's average prediction.")
    with c2:
        st.subheader("Most influential factors overall")
        imp = importance[importance["horizon"] == horizon].head(12).sort_values("mean_abs_impact_pct")
        figi = px.bar(imp, x="mean_abs_impact_pct", y="label", orientation="h",
                      color_discrete_sequence=[C_MODEL])
        figi.update_layout(height=380, xaxis_title="Mean |impact| (pp)", yaxis_title=None, margin=dict(t=10))
        st.plotly_chart(figi, use_container_width=True)

# ---------------------------------------------------------------------------
# Accuracy tab
# ---------------------------------------------------------------------------
with tab_acc:
    st.subheader(f"Rolling-origin backtest: last {meta.get('backtest_weeks', 52)} weeks, all provinces")
    st.markdown(
        "Every week the model is asked to forecast using **only data available at that time** "
        "(retrained every 13 weeks). It is compared against two honest baselines: "
        "*naive* (price stays the same) and *seasonal naive* (repeat last year's move)."
    )
    figa = px.line(metrics, x="horizon", y="MAPE", color="model", markers=True,
                   color_discrete_map={"XGBoost": C_MODEL, "Naive (last value)": C_NEUTRAL,
                                       "Seasonal naive": "#d97706"})
    figa.update_layout(height=340, xaxis_title="Weeks ahead", yaxis_title="MAPE (%)", legend_title=None,
                       margin=dict(t=10))
    st.plotly_chart(figa, use_container_width=True)

    def _fmt(v, f):
        return "—" if pd.isna(v) else f.format(v)

    show = pd.DataFrame({
        "Weeks": metrics["horizon"],
        "Model": metrics["model"],
        "MAE (Rp)": [_fmt(v, "{:,.0f}") for v in metrics["MAE"]],
        "MAPE %": [_fmt(v, "{:.2f}") for v in metrics["MAPE"]],
        "MAE vs naive %": [_fmt(-v * 100 + 0.0, "{:+.1f}") for v in metrics["skill_vs_naive"]],
        "DM p-value": [_fmt(v, "{:.3f}") for v in metrics.get("dm_pvalue", pd.Series([None] * len(metrics)))],
        "Direction %": [_fmt(v * 100, "{:.1f}") for v in metrics["direction_acc"]],
        "80% PI cover %": [_fmt(v * 100, "{:.1f}") for v in metrics["interval_coverage"]],
    })
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption("Diebold–Mariano test (Diebold & Mariano, 1995; Harvey et al., 1997) on weekly "
               "province-averaged absolute errors. p < 0.05 means the accuracy difference vs the naive "
               "forecast is unlikely to be luck.")

    st.subheader(f"{prov}: forecast made {horizon} week(s) earlier vs what happened")
    bpp = bt[(bt["province"] == prov) & (bt["horizon"] == horizon) & (~bt["is_warmup"])]
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=list(bpp["target_week"]) + list(bpp["target_week"])[::-1],
                              y=list(bpp["price_hi"]) + list(bpp["price_lo"])[::-1], fill="toself", mode="lines",
                              fillcolor=C_BAND, line=dict(width=0), name="80% interval", hoverinfo="skip"))
    figp.add_trace(go.Scatter(x=bpp["target_week"], y=bpp["target_price"], name="Actual",
                              line=dict(color=C_ACTUAL, width=2)))
    figp.add_trace(go.Scatter(x=bpp["target_week"], y=bpp["price_xgb"], name="XGBoost",
                              line=dict(color=C_MODEL, width=2)))
    figp.add_trace(go.Scatter(x=bpp["target_week"], y=bpp["price_naive"], name="Naive",
                              line=dict(color=C_NEUTRAL, width=1, dash="dot")))
    figp.update_layout(height=380, hovermode="x unified", yaxis_title="Rp / kg",
                       legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(figp, use_container_width=True)

    pm = bpp.assign(ape=(bpp["price_xgb"] - bpp["target_price"]).abs() / bpp["target_price"] * 100)
    prov_err = (bt[(bt["horizon"] == horizon) & (~bt["is_warmup"])]
                .assign(ape=lambda x: (x["price_xgb"] - x["target_price"]).abs() / x["target_price"] * 100)
                .groupby("province")["ape"].mean().sort_values())
    with st.expander("MAPE by province"):
        st.bar_chart(prov_err, horizontal=True, height=700)

# ---------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Data sources (all downloaded from public APIs, no synthetic data)")
    st.markdown(
        """
| Data | Source | Frequency | Used for |
|---|---|---|---|
| Chicken retail / wholesale / producer prices, egg prices | [Bank Indonesia PIHPS](https://www.bi.go.id/hargapangan) | Daily, 34 provinces, 2018 → | Target + supply-chain signals |
| Maize & soybean-meal world prices, IDR/USD | [FRED / IMF](https://fred.stlouisfed.org/series/PMAIZMTUSDM) | Monthly | Feed-cost hypothesis: tested, **not used** (hurt accuracy) |
| Ramadan, Idul Fitri, Idul Adha | [Aladhan Hijri calendar API](https://aladhan.com/islamic-calendar-api), checked against Google's official Indonesian holiday calendar | Daily | Demand seasonality |
| National public holidays | [Nager.Date](https://date.nager.at) | Daily | Market closures / demand |
| Rainfall & temperature | [Open-Meteo ERA5 archive](https://open-meteo.com/en/docs/historical-weather-api) at each province capital | Daily | Heat stress (used); rainfall tested, **not used** (no effect) |
"""
    )
    st.subheader("Which drivers made it into the model?")
    st.markdown(
        "Every candidate driver comes from published research ([literature review]"
        "(https://github.com/riffarhan/chicken-price-forecast/blob/main/docs/literature.md)). "
        "Each group was then removed from, or added to, the model and re-tested on the same backtest. "
        "A group is kept only if it lowers forecast error."
    )
    abl_path = OUTPUTS_DIR / "ablation.csv"
    if abl_path.exists() and "design" in pd.read_csv(abl_path, nrows=1).columns:
        abl = pd.read_csv(abl_path)
        abl = abl[abl["design"] != "reference"]
        cols = [c for c in abl.columns if c.startswith("mae_change_") or c.startswith("dm_p_")]
        tbl_a = abl[["design", "group", *cols]].copy()
        for c in cols:
            fmt = "{:+.1f}%" if c.startswith("mae_change_") else "{:.3f}"
            tbl_a[c] = [("—" if pd.isna(v) else fmt.format(v)) for v in tbl_a[c]]
        tbl_a.columns = [c.replace("mae_change_", "MAE change ").replace("dm_p_", "DM p ")
                         .replace("_pct", "").replace("w", " wk") if c not in ("design", "group") else c.title()
                         for c in tbl_a.columns]
        st.dataframe(tbl_a, hide_index=True, use_container_width=True)
        st.caption("Leave one out: positive = removing the group made forecasts worse (the group helps). "
                   "Add to core: negative = adding the group made forecasts better. "
                   f"Kept in the model: {', '.join(meta.get('feature_groups', []))}.")

    sub = panel[panel["province"] == prov].tail(history_weeks)
    c1, c2 = st.columns(2)
    with c1:
        feed = panel.drop_duplicates("week")[["week", "maize_idr_kg", "soymeal_idr_kg"]].tail(history_weeks)
        feed = feed.melt("week", var_name="series", value_name="Rp/kg").dropna()
        feed["series"] = feed["series"].map({"maize_idr_kg": "Maize", "soymeal_idr_kg": "Soybean meal"})
        figf = px.line(feed, x="week", y="Rp/kg", color="series", title="Feed ingredient prices (Rp/kg)",
                       color_discrete_sequence=[C_PRODUCER, C_WHOLESALE])
        figf.update_layout(height=320, legend_title=None)
        st.plotly_chart(figf, use_container_width=True)
    with c2:
        figr = px.bar(sub, x="week", y="rain_mm", title=f"Weekly rainfall, {prov} (mm)",
                      color_discrete_sequence=["#60a5fa"])
        figr.update_layout(height=320)
        st.plotly_chart(figr, use_container_width=True)
    st.download_button("Download weekly panel (CSV)", panel.to_csv(index=False), "weekly_panel.csv", "text/csv")
