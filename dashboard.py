"""
Streamlit dashboard for chicken price forecasting.

Displays predictions, performance metrics, SHAP explanations,
and province-level price comparisons.

Usage:
    streamlit run dashboard.py
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chicken Price Forecast", page_icon="🐔", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame | None:
    path = os.path.join(OUTPUTS_DIR, "predictions_latest.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(ttl=3600)
def load_metrics() -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, "model_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data(ttl=3600)
def load_shap() -> pd.DataFrame | None:
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    files = sorted(f for f in os.listdir(OUTPUTS_DIR) if f.startswith("shap_summary"))
    if not files:
        return None
    try:
        return pd.read_csv(os.path.join(OUTPUTS_DIR, files[-1]), parse_dates=["date"])
    except Exception:
        return None


df = load_predictions()
metrics = load_metrics()
shap_df = load_shap()

if df is None:
    st.error("No prediction data found. Run the pipeline first:")
    st.code("python generate_dataset.py\npython train_model.py\npython predict.py", language="bash")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Chicken Price Forecasting Dashboard")
st.markdown("Real-time predictions powered by XGBoost")

if len(metrics) > 0:
    last_update = metrics["timestamp"].max()
    st.caption(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
st.header("Model Performance")

if len(metrics) > 0:
    latest = metrics.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{latest['rmse']:,.0f} IDR/kg")
    c2.metric("MAPE", f"{latest['mape']:.2f}%")
    status_icon = "🟢" if latest["status"] == "HEALTHY" else "🔴"
    c3.metric("Status", f"{status_icon} {latest['status']}")
    c4.metric("Predictions", f"{int(latest['n_predictions']):,}")
else:
    st.info("No performance metrics available yet — run the prediction pipeline.")

# ---------------------------------------------------------------------------
# Province price trend
# ---------------------------------------------------------------------------
st.header("Price Trends by Province")

provinces = sorted(df["province"].unique())
selected = st.selectbox("Select Province", provinces, index=0)

prov = df[df["province"] == selected].sort_values("date").tail(30)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=prov["date"],
        y=prov["chicken_price"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6),
    )
)
fig.add_trace(
    go.Scatter(
        x=prov["date"],
        y=prov["predicted_price"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=6, symbol="square"),
    )
)
fig.update_layout(
    title=f"Chicken Price — {selected} (Last 30 Weeks)",
    xaxis_title="Date",
    yaxis_title="Price (IDR/kg)",
    hovermode="x unified",
    height=480,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Latest predictions table
# ---------------------------------------------------------------------------
st.header("Latest Predictions (All Provinces)")

latest_pred = (
    df.sort_values("date")
    .groupby("province")
    .tail(1)[["province", "date", "chicken_price", "predicted_price", "prediction_error_pct"]]
    .sort_values("predicted_price", ascending=False)
)

display = latest_pred.copy()
display.columns = ["Province", "Date", "Actual (IDR/kg)", "Predicted (IDR/kg)", "Error (%)"]
display["Actual (IDR/kg)"] = display["Actual (IDR/kg)"].map("{:,.0f}".format)
display["Predicted (IDR/kg)"] = display["Predicted (IDR/kg)"].map("{:,.0f}".format)
display["Error (%)"] = display["Error (%)"].map("{:+.2f}%".format)
st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------
if shap_df is not None:
    st.header("What Drives the Predictions? (SHAP)")
    st.markdown(
        "**Top factor per province** — positive (+) pushes the price up, "
        "negative (-) pushes it down."
    )
    shap_display = shap_df[
        ["province", "predicted_price", "top_driver", "top_driver_impact"]
    ].copy()
    shap_display["predicted_price"] = shap_display["predicted_price"].map(
        "{:,.0f} IDR/kg".format
    )
    shap_display["top_driver_impact"] = shap_display["top_driver_impact"].map(
        "{:+,.0f} IDR/kg".format
    )
    shap_display.columns = ["Province", "Predicted Price", "Top Factor", "Impact"]
    st.dataframe(shap_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Province comparison bar chart
# ---------------------------------------------------------------------------
st.header("Price Comparison Across Provinces")

by_prov = df.sort_values("date").groupby("province").tail(1).copy()
fig_bar = px.bar(
    by_prov.sort_values("predicted_price", ascending=False),
    x="predicted_price",
    y="province",
    orientation="h",
    title="Latest Predicted Prices by Province",
    labels={"predicted_price": "Price (IDR/kg)", "province": ""},
    color="predicted_price",
    color_continuous_scale="RdYlGn_r",
    text="predicted_price",
)
fig_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
fig_bar.update_layout(height=550, showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Performance over time
# ---------------------------------------------------------------------------
if len(metrics) > 1:
    st.header("Model Performance Over Time")
    fig_perf = go.Figure()
    fig_perf.add_trace(
        go.Scatter(
            x=metrics["timestamp"],
            y=metrics["rmse"],
            mode="lines+markers",
            name="RMSE",
            line=dict(color="purple", width=2),
            marker=dict(size=8),
        )
    )
    fig_perf.add_hline(
        y=1200,
        line_dash="dash",
        line_color="red",
        annotation_text="Alert Threshold (1,200)",
        annotation_position="right",
    )
    fig_perf.update_layout(
        title="RMSE Trend Over Time",
        xaxis_title="Date",
        yaxis_title="RMSE (IDR/kg)",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
st.header("Alerts & Status")

if len(metrics) > 0:
    lm = metrics.iloc[-1]
    if lm["status"] == "DEGRADED":
        st.error(
            f"Performance Alert: RMSE ({lm['rmse']:,.0f}) exceeds threshold. "
            "Consider retraining the model (`python train_model.py`)."
        )
    else:
        st.success("Model performance is healthy.")

    high_err = latest_pred[abs(latest_pred["prediction_error_pct"]) > 5]
    if len(high_err) > 0:
        st.warning(f"{len(high_err)} province(s) with prediction error > 5%")
        with st.expander("View provinces"):
            st.dataframe(high_err[["province", "prediction_error_pct"]])
    else:
        st.success("All provinces within acceptable accuracy (< 5% error).")
else:
    st.info("Run the prediction pipeline to see alerts.")

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.header("Download Data")

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download Predictions (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
with c2:
    if shap_df is not None:
        st.download_button(
            "Download SHAP Analysis (CSV)",
            shap_df.to_csv(index=False).encode("utf-8"),
            file_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "**Chicken Price Forecasting** | Powered by XGBoost | "
    "[GitHub](https://github.com/riffarhan/chicken-price-forecast)"
)
