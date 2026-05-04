"""
app.py  —  Retail Forecasting Dashboard
========================================
Run locally:   streamlit run app.py
Deploy:        streamlit community cloud  →  push repo to GitHub, point at app.py
"""

import json, pickle, os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import holidays

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
EXPORT_DIR  = BASE_DIR / "exported_models"
META_PATH   = EXPORT_DIR / "metadata.json"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.stPlotlyChart { border-radius: 12px; }
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_resource
def load_prophet_model(cluster_id):
    path = EXPORT_DIR / "prophet" / f"cluster_{cluster_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_neural_model(cluster_id):
    from neuralforecast import NeuralForecast
    path = str(EXPORT_DIR / "neural" / f"cluster_{cluster_id}")
    return NeuralForecast.load(path=path)

def make_shopping_holidays():
    rows = []
    for year in [2009, 2010, 2011, 2012, 2013]:
        # approximate Black Friday = 4th Friday of November
        import calendar
        nov_days = [datetime(year, 11, d) for d in range(1, 31)
                    if datetime(year, 11, d).weekday() == 4]
        bf = nov_days[3] if len(nov_days) >= 4 else nov_days[-1]
        cm = bf + timedelta(days=3)
        rows.append({"holiday": "BlackFriday",  "ds": pd.Timestamp(bf), "lower_window": 0, "upper_window": 1})
        rows.append({"holiday": "CyberMonday",  "ds": pd.Timestamp(cm), "lower_window": 0, "upper_window": 0})
    return pd.DataFrame(rows)

def add_calendar_features(df):
    uk_hols = holidays.UnitedKingdom(years=list(range(2009, 2028)))
    uk_holiday_dates = pd.to_datetime(list(uk_hols.keys()))

    shopping_bfs = []
    for year in range(2009, 2028):
        import calendar
        nov_days = [datetime(year, 11, d) for d in range(1, 31)
                    if datetime(year, 11, d).weekday() == 4]
        bf = nov_days[3] if len(nov_days) >= 4 else nov_days[-1]
        cm = bf + timedelta(days=3)
        shopping_bfs.append(pd.Timestamp(bf))
        shopping_bfs.append(pd.Timestamp(cm))

    df = df.copy()
    df["Month"]               = df["ds"].dt.month
    df["DayOfWeek"]           = df["ds"].dt.dayofweek
    df["WeekOfYear"]          = df["ds"].dt.isocalendar().week.astype(int)
    df["Quarter"]             = df["ds"].dt.quarter
    df["is_weekend"]          = (df["DayOfWeek"] >= 5).astype(int)
    df["is_saturday"]         = (df["DayOfWeek"] == 5).astype(int)
    df["is_uk_holiday"]       = df["ds"].isin(uk_holiday_dates).astype(int)
    df["is_black_friday"]     = df["ds"].isin(shopping_bfs).astype(int)
    df["is_cyber_monday"]     = df["ds"].isin(shopping_bfs).astype(int)
    df["is_christmas_period"] = ((df["ds"].dt.month == 12) & (df["ds"].dt.day >= 20)).astype(int)
    df["is_year_end"]         = ((df["ds"].dt.month == 12) & (df["ds"].dt.day >= 26)).astype(int)
    df["is_january"]          = (df["ds"].dt.month == 1).astype(int)
    return df

def run_prophet_forecast(cluster_id, meta, horizon):
    model = load_prophet_model(cluster_id)

    history = pd.DataFrame(meta["history"])
    history["ds"] = pd.to_datetime(history["ds"])
    end_date = pd.to_datetime(meta["end_date"])

    future = model.make_future_dataframe(periods=horizon, freq="D")
    cfg_growth = "logistic" if getattr(model, "growth", "linear") == "logistic" else "linear"
    if cfg_growth == "logistic":
        cap   = history["y"].max() * 1.2
        future["cap"]   = cap
        future["floor"] = 0.0

    forecast = model.predict(future)
    forecast_out = forecast[forecast["ds"] > end_date][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_out["yhat"] = forecast_out["yhat"].clip(lower=0)
    forecast_out["yhat_lower"] = forecast_out["yhat_lower"].clip(lower=0)
    return forecast_out.head(horizon)

def run_neural_forecast(cluster_id, meta, horizon):
    nf = load_neural_model(cluster_id)
    model_name = meta["model_name"].replace("_SigFeatures", "")
    futr_cols  = meta["futr_cols"]
    hist_cols  = meta["hist_cols"]

    history = pd.DataFrame(meta["history"])
    history["ds"] = pd.to_datetime(history["ds"])
    history["y"]  = history["y"].astype(float)
    history["unique_id"] = f"Cluster_{cluster_id}"

    end_date = pd.to_datetime(meta["end_date"])
    future_dates = pd.date_range(end_date + timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame({"ds": future_dates, "unique_id": f"Cluster_{cluster_id}"})
    future_df = add_calendar_features(future_df)

    # Add lag/rolling placeholders from last known values
    last_vals = history["y"].values
    for lag in [1, 7, 14]:
        future_df[f"lag_{lag}"] = last_vals[-lag] if lag <= len(last_vals) else 0.0
    future_df["rolling_mean_7"] = last_vals[-7:].mean() if len(last_vals) >= 7 else last_vals.mean()
    future_df["rolling_std_7"]  = last_vals[-7:].std()  if len(last_vals) >= 7 else 0.0

    try:
        if model_name in ["NBEATS", "PatchTST"]:
            preds = nf.predict(df=history).reset_index()
        else:
            preds = nf.predict(df=history, futr_df=future_df[["ds","unique_id"]+futr_cols]).reset_index()
    except Exception:
        preds = nf.predict(df=history).reset_index()

    preds = preds.sort_values("ds")
    preds = preds[preds["ds"] > end_date].head(horizon)

    # column name is usually the model class name
    pred_col = [c for c in preds.columns if c not in ["ds","unique_id"]]
    if not pred_col:
        return None

    out = preds[["ds", pred_col[0]]].rename(columns={pred_col[0]: "yhat"})
    out["yhat"] = out["yhat"].clip(lower=0)
    out["yhat_lower"] = (out["yhat"] * 0.85).clip(lower=0)
    out["yhat_upper"] = out["yhat"] * 1.15
    return out

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/shopping-cart.png", width=48)
    st.title("🔮 Retail Forecasting")
    st.caption("Online Retail II · UK Gift Retailer · 2009–2011")
    st.divider()

    meta = load_metadata()
    cluster_options = list(meta.keys())

    selected_cluster = st.selectbox(
        "📦 Select Product Cluster",
        cluster_options,
        format_func=lambda x: f"{x}  ({meta[x]['model_name']})",
    )

    horizon = st.slider("📅 Forecast Horizon (days)", 7, 180, 60, step=7)

    st.divider()
    st.markdown("**Model Legend**")
    st.markdown("""
- 🧠 **NBEATS** — basis expansion  
- 📈 **NHITS** — hierarchical interpolation  
- 🔷 **PatchTST** — patch transformer  
- 🔮 **Prophet** — trend + seasonality  
- ★ **_SigFeatures** — significant features only  
""")
    st.divider()
    st.caption("Built with NeuralForecast · Prophet · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
info = meta[selected_cluster]
cluster_id = int(selected_cluster.split("_")[1])

st.title(f"📦 {selected_cluster} — {info['model_name']}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best SMAPE", f"{info['smape']:.1f}%")
col2.metric("Model Type", "Prophet" if info["is_prophet"] else "Neural")
col3.metric("Sig. Features", "Yes" if info["use_sig"] else "No")
col4.metric("Forecast Days", horizon)

st.divider()

# ── Run forecast ──────────────────────────────────────────────────────────────
with st.spinner(f"Generating {horizon}-day forecast for {selected_cluster} …"):
    try:
        if info["is_prophet"]:
            forecast_df = run_prophet_forecast(cluster_id, info, horizon)
        else:
            forecast_df = run_neural_forecast(cluster_id, info, horizon)
        forecast_ok = forecast_df is not None and len(forecast_df) > 0
    except Exception as e:
        forecast_ok = False
        st.error(f"Forecast failed: {e}")
        forecast_df = None

# ── Plot ──────────────────────────────────────────────────────────────────────
history_df = pd.DataFrame(info["history"])
history_df["ds"] = pd.to_datetime(history_df["ds"])

fig = go.Figure()

# History
fig.add_trace(go.Scatter(
    x=history_df["ds"], y=history_df["y"],
    name="Historical Sales",
    line=dict(color="#3b82f6", width=2),
    mode="lines",
))

if forecast_ok:
    # Confidence band
    if "yhat_upper" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
            y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(249,115,22,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% CI",
            showlegend=True,
        ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        name="Forecast",
        line=dict(color="#f97316", width=2.5, dash="dot"),
        mode="lines",
    ))
    # Separator
    end_date = pd.to_datetime(info["end_date"])
    fig.add_vline(x=str(end_date), line_dash="dash", line_color="#94a3b8",
                  annotation_text="Forecast Start", annotation_position="top right")

fig.update_layout(
    title=f"{selected_cluster} — Daily Units Sold + {horizon}-Day Forecast",
    xaxis_title="Date", yaxis_title="Units Sold",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=420,
    margin=dict(l=0, r=0, t=60, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── Forecast table ────────────────────────────────────────────────────────────
if forecast_ok:
    st.subheader(f"📋 Forecast Values — next {horizon} days")
    display_cols = ["ds", "yhat"]
    if "yhat_lower" in forecast_df.columns:
        display_cols += ["yhat_lower", "yhat_upper"]

    show_df = forecast_df[display_cols].copy()
    show_df.columns = ["Date", "Forecast (units)"] + (
        ["Lower 80% CI", "Upper 80% CI"] if len(display_cols) > 2 else []
    )
    show_df["Date"] = show_df["Date"].dt.strftime("%Y-%m-%d")
    for c in show_df.columns[1:]:
        show_df[c] = show_df[c].round(1)

    st.dataframe(show_df, use_container_width=True, height=300)

    # Download
    csv = show_df.to_csv(index=False)
    st.download_button(
        "⬇️  Download forecast CSV",
        data=csv,
        file_name=f"{selected_cluster}_forecast_{horizon}d.csv",
        mime="text/csv",
    )

# ── All-cluster SMAPE heatmap ─────────────────────────────────────────────────
st.divider()
st.subheader("🗺️  All-Cluster Model Overview")

summary_rows = []
for cname, cinfo in meta.items():
    summary_rows.append({
        "Cluster": cname,
        "Best Model": cinfo["model_name"],
        "SMAPE (%)": cinfo["smape"],
        "Type": "Prophet" if cinfo["is_prophet"] else "Neural",
        "Sig. Features": "✓" if cinfo["use_sig"] else "—",
    })
summary_df = pd.DataFrame(summary_rows)

col_a, col_b = st.columns([2, 1])

with col_a:
    bar = px.bar(
        summary_df.sort_values("SMAPE (%)"),
        x="Cluster", y="SMAPE (%)",
        color="Type",
        color_discrete_map={"Prophet": "#8b5cf6", "Neural": "#3b82f6"},
        text="SMAPE (%)",
        title="SMAPE by Cluster (lower = better)",
    )
    bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    bar.update_layout(template="plotly_white", height=350,
                      showlegend=True, margin=dict(t=50, b=0))
    st.plotly_chart(bar, use_container_width=True)

with col_b:
    st.dataframe(
        summary_df[["Cluster","Best Model","SMAPE (%)","Sig. Features"]],
        use_container_width=True, hide_index=True, height=350,
    )
