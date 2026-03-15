"""
streamlit_app.py — Ontario Electricity Demand Forecast Dashboard
Run: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from app.model import load_models, predict

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ontario Grid Intelligence | Demand Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #050C1A; }

[data-testid="stAppViewContainer"] { overflow-y: auto !important; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #07111F !important;
    border-right: 1px solid #0F2035 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #7A9BB8 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, #08172A 0%, #060D1A 100%) !important;
    border: 1px solid #0F2540 !important;
    border-top: 2px solid #0D4F8A !important;
    border-radius: 3px !important;
    padding: 1rem 1.1rem 0.9rem !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #2E6EA6 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 400 !important;
    color: #D6EEFF !important;
    letter-spacing: -0.02em !important;
}

/* Tabs */
[data-testid="stTabs"] { border-bottom: 1px solid #0F2035 !important; }
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #2E6EA6 !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] button:hover { color: #7AB8E8 !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #5BC0F8 !important;
    border-bottom: 2px solid #5BC0F8 !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #0D4F8A !important;
    color: #5BC0F8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.45rem 1rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: rgba(13,79,138,0.2) !important;
    border-color: #5BC0F8 !important;
}

/* Sliders */
[data-testid="stSlider"] label p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #2E6EA6 !important;
}

/* Toggle */
[data-testid="stToggle"] label p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #2E6EA6 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #07111F !important;
    border: 1px dashed #0F2540 !important;
    border-radius: 3px !important;
}
[data-testid="stFileUploader"] * {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #07111F !important;
    border: 1px solid #0F2035 !important;
    border-radius: 3px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #2E6EA6 !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid #0F2035 !important;
    margin: 1rem 0 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    background: #07111F !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #0F2035 !important; border-radius: 3px !important; }
[data-testid="stDataFrame"] * {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #7AB8E8 !important;
}

/* Headings */
h1, h2, h3 { color: #D6EEFF !important; font-weight: 300 !important; }

/* ── Custom components ── */
.page-header { border-bottom: 1px solid #0F2035; padding-bottom: 1rem; margin-bottom: 1.2rem; }
.page-title  { font-family: 'IBM Plex Sans', sans-serif; font-size: 1.5rem; font-weight: 300; color: #D6EEFF; letter-spacing: -0.02em; margin: 0 0 4px 0; }
.page-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.16em; text-transform: uppercase; color: #1A4A6E; }
.timestamp-badge { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #1A4A6E; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.16em; text-transform: uppercase; color: #1E5880; padding-bottom: 0.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid #0A1E30; }
.badge-live   { display:inline-block; background:rgba(0,180,80,0.08); border:1px solid rgba(0,180,80,0.25); color:#00C853; font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.14em; text-transform:uppercase; padding:1px 7px; border-radius:2px; }
.badge-upload { display:inline-block; background:rgba(13,79,138,0.12); border:1px solid rgba(13,79,138,0.35); color:#5BC0F8; font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.14em; text-transform:uppercase; padding:1px 7px; border-radius:2px; }
.waiting-card { background:#07111F; border:1px solid #0F2035; border-radius:3px; padding:3rem 2rem; text-align:center; margin:2rem 0; }
.waiting-title { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; letter-spacing:0.18em; text-transform:uppercase; color:#1A4A6E; margin-bottom:0.8rem; }
.waiting-body { font-family:'IBM Plex Sans',sans-serif; font-size:0.88rem; font-weight:300; color:#2E6EA6; line-height:2; }
.waiting-source { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#0F2540; margin-top:1rem; }
.sidebar-logo-eyebrow { font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#1A4060 !important; margin-bottom:3px; }
.sidebar-logo-title { font-family:'IBM Plex Sans',sans-serif; font-size:1.05rem; font-weight:300; color:#D6EEFF !important; letter-spacing:-0.01em; }
.sidebar-stats { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#2E6EA6; line-height:2.1; }
.sidebar-stats span { color:#5BC0F8 !important; }
.risk-note { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#1A4060; line-height:1.9; margin-top:0.6rem; }
.model-tag { display:inline-block; background:rgba(13,79,138,0.1); border:1px solid #0D3A60; color:#3A88C0; font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.1em; padding:2px 6px; border-radius:2px; margin:2px; }
.arch-body { font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem; font-weight:300; color:#4A8AB8; line-height:2; }
.limits-body { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#1E5070; line-height:2.2; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly base theme
# ─────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#060E1C",
    font=dict(family="IBM Plex Mono, monospace", color="#2E6EA6", size=10),
    xaxis=dict(
        gridcolor="#080F1A", gridwidth=1, linecolor="#0F2035",
        tickcolor="#0F2035", tickfont=dict(size=10, color="#1E5070"),
        title_font=dict(size=10, color="#1E5070"), zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#080F1A", gridwidth=1, linecolor="#0F2035",
        tickcolor="#0F2035", tickfont=dict(size=10, color="#1E5070"),
        title_font=dict(size=10, color="#1E5070"),
        zeroline=False, tickformat=",.0f",
    ),
    legend=dict(
        bgcolor="rgba(5,12,26,0.9)", bordercolor="#0F2035", borderwidth=1,
        font=dict(size=10, color="#2E6EA6"),
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
    ),
    margin=dict(l=10, r=10, t=36, b=10),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#07111F", bordercolor="#0F2035",
        font=dict(family="IBM Plex Mono, monospace", size=10, color="#7AB8E8"),
    ),
)

# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

try:
    bundle = get_models()
    models_ok = True
except FileNotFoundError:
    models_ok = False


@st.cache_data
def load_uploaded(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"] - 1, unit="h")
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[["hourly_demand", "hourly_average_price"]].rename(
        columns={"hourly_demand": "demand_kwh", "hourly_average_price": "price_cents"}
    )
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(expected).interpolate(method="time")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding: 0.6rem 0 1.2rem 0;">
            <div class="sidebar-logo-eyebrow">IESO · Ontario Grid Intelligence</div>
            <div class="sidebar-logo-title">Demand Forecast</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)

    st.markdown("""
        <div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;
                    color:#2E6EA6;line-height:2.2;margin-bottom:0.8rem;">
            Download the latest hourly demand<br>data directly from IESO:
        </div>
        <a href="https://reports-public.ieso.ca/public/Demand/PUB_Demand_2024.csv"
           target="_blank"
           style="display:block;background:transparent;border:1px solid #0D4F8A;
                  color:#5BC0F8;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                  font-weight:500;letter-spacing:0.1em;text-transform:uppercase;
                  border-radius:2px;padding:0.45rem 1rem;text-align:center;
                  text-decoration:none;margin-bottom:6px;">
            ↓ Download 2024 Demand CSV
        </a>
        <a href="https://reports-public.ieso.ca/public/Demand/PUB_Demand_2025.csv"
           target="_blank"
           style="display:block;background:transparent;border:1px solid #0D4F8A;
                  color:#5BC0F8;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                  font-weight:500;letter-spacing:0.1em;text-transform:uppercase;
                  border-radius:2px;padding:0.45rem 1rem;text-align:center;
                  text-decoration:none;margin-bottom:14px;">
            ↓ Download 2025 Demand CSV
        </a>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                    color:#1A4060;line-height:1.8;margin-bottom:0.8rem;">
            Then upload the file below.
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Columns required: date, hour, hourly_demand, hourly_average_price",
    )

    st.divider()
    st.markdown('<div class="section-label">Forecast Parameters</div>', unsafe_allow_html=True)
    forecast_horizon = st.slider("Horizon (hours)", 6,    168,  24,  6)
    n_simulations    = st.slider("MC Simulations",  100, 2000, 500, 100)
    run_mc           = st.toggle("Risk Simulation", value=True)

    st.divider()
    m = bundle.metrics if models_ok else {}
    st.markdown('<div class="section-label">Model Statistics</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="sidebar-stats">
            QR MAE &nbsp;&nbsp;&nbsp;&nbsp; <span>{m.get('quantile_median_mae',0):,.0f} kWh</span><br>
            QR RMSE &nbsp;&nbsp;&nbsp; <span>{m.get('quantile_median_rmse',0):,.0f} kWh</span><br>
            LR MAE &nbsp;&nbsp;&nbsp;&nbsp; <span>{m.get('linear_regression_mae',0):,.0f} kWh</span><br>
            Coverage &nbsp;&nbsp; <span>{m.get('empirical_coverage_80pct',0):.1%}</span><br>
            Train Cut &nbsp; <span>2019-01-01</span>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────
st.markdown(f"""
    <div class="page-header">
        <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:4px;">
            <div class="page-title">Ontario Electricity Demand Forecast</div>
            <div class="timestamp-badge">{datetime.utcnow().strftime('%Y-%m-%d  %H:%M  UTC')}</div>
        </div>
        <div class="page-subtitle">
            Probabilistic · 24h Ahead · Quantile Regression + Monte Carlo Simulation
        </div>
    </div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error("Model artifacts not found. Run: python -m app.model --data data/ontario_electricity_demand.csv")
    st.stop()

# ─────────────────────────────────────────────
# Data loading
# Priority: 1) user upload  2) auto-updated CSV from GitHub Action  3) waiting state
# ─────────────────────────────────────────────
AUTO_DATA_PATH = Path(__file__).parent / "data" / "ontario_electricity_demand.csv"

@st.cache_data
def load_auto() -> pd.DataFrame:
    return load_uploaded(AUTO_DATA_PATH)

df, data_source = None, None

if uploaded_file is not None:
    df = load_uploaded(uploaded_file)
    data_source = "uploaded"

if df is None and AUTO_DATA_PATH.exists():
    try:
        df = load_auto()
        data_source = "live"
    except Exception:
        df = None

if df is None:
    st.markdown("""
        <div class="waiting-card">
            <div class="waiting-title">Awaiting Data Input</div>
            <div class="waiting-body">
                Download a CSV using the sidebar links,<br>
                then upload <code style="color:#5BC0F8;font-size:0.82rem;">ontario_electricity_demand.csv</code>
            </div>
            <div class="waiting-source">Source: reports-public.ieso.ca/public/Demand/</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Performance — Test Set (Post 2019)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LR MAE",       f"{m.get('linear_regression_mae',0):,.0f} kWh")
    c2.metric("LR RMSE",      f"{m.get('linear_regression_rmse',0):,.0f} kWh")
    c3.metric("QR MAE",       f"{m.get('quantile_median_mae',0):,.0f} kWh")
    c4.metric("QR RMSE",      f"{m.get('quantile_median_rmse',0):,.0f} kWh")
    c5.metric("80% Coverage", f"{m.get('empirical_coverage_80pct',0):.1%}")
    st.stop()

if len(df) < 168:
    st.error("Insufficient history — minimum 168 hours required.")
    st.stop()

# ─────────────────────────────────────────────
# Run forecast
# ─────────────────────────────────────────────
with st.spinner("Computing probabilistic forecast..."):
    forecast_list, mc_summary = predict(
        bundle=bundle,
        recent_demand=df["demand_kwh"].tolist(),
        recent_price=df["price_cents"].tolist(),
        forecast_horizon=forecast_horizon,
        n_simulations=n_simulations,
        run_monte_carlo=run_mc,
    )

forecast_df = pd.DataFrame(forecast_list)
forecast_df["timestamp"] = pd.date_range(
    start=df.index[-1] + pd.Timedelta(hours=1),
    periods=len(forecast_df), freq="h",
)

badge = (
    '<span class="badge-live">● LIVE IESO</span>'   if data_source == "live"
    else '<span class="badge-upload">● CSV UPLOAD</span>'
)

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
st.markdown(f'<div class="section-label">Forecast Summary &nbsp;{badge}</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Median Peak Demand",  f"{forecast_df['median'].max():,.0f} kWh")
c2.metric("Forecast Horizon",    f"{forecast_horizon}h")
c3.metric("Avg Uncertainty",     f"±{(forecast_df['upper_80']-forecast_df['lower_80']).mean()/2:,.0f} kWh")
c4.metric("Interval Coverage",   "80.9%")
c5.metric("Data Through",        df.index[-1].strftime("%Y-%m-%d %H:00"))

st.divider()

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Demand Forecast", "Risk & Simulation", "Model Diagnostics"])

# ══════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-label">Probabilistic Forecast — Next {forecast_horizon} Hours</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
        y=pd.concat([forecast_df["upper_80"], forecast_df["lower_80"][::-1]]),
        fill="toself", fillcolor="rgba(13,79,138,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Prediction Interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["upper_80"],
        mode="lines", name="Q90 Upper",
        line=dict(color="#0D4F8A", width=1, dash="dot"),
        hovertemplate="%{y:,.0f} kWh<extra>Q90</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["lower_80"],
        mode="lines", name="Q10 Lower",
        line=dict(color="#0D4F8A", width=1, dash="dot"),
        hovertemplate="%{y:,.0f} kWh<extra>Q10</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["median"],
        mode="lines", name="Median Forecast (Q50)",
        line=dict(color="#5BC0F8", width=2.5),
        hovertemplate="%{y:,.0f} kWh<extra>Q50 Median</extra>",
    ))
    fig.update_layout(**PLOT_BASE, height=420, xaxis_title="", yaxis_title="Demand (kWh)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Historical Demand — Prior 7 Days</div>', unsafe_allow_html=True)
    history = df["demand_kwh"].tail(168)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=history.index, y=history.values,
        mode="lines", name="Actual Demand",
        line=dict(color="#1A7A4A", width=1.5),
        fill="tozeroy", fillcolor="rgba(26,122,74,0.05)",
        hovertemplate="%{y:,.0f} kWh<extra>Actual</extra>",
    ))
    fig2.update_layout(**PLOT_BASE, height=260, xaxis_title="", yaxis_title="Demand (kWh)", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw Forecast Data"):
        disp = forecast_df[["timestamp","lower_80","median","upper_80"]].copy()
        disp.columns = ["Timestamp","Q10 Lower (kWh)","Q50 Median (kWh)","Q90 Upper (kWh)"]
        st.dataframe(disp.set_index("Timestamp").style.format("{:,.0f}"), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — RISK
# ══════════════════════════════════════════════
with tab2:
    if not run_mc or mc_summary is None:
        st.info("Enable Risk Simulation in the sidebar to view this analysis.")
        st.stop()

    st.markdown('<div class="section-label">Peak Demand Risk Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Peak",   f"{mc_summary['mean_peak']:,.0f} kWh")
    c2.metric("Std Deviation",   f"{mc_summary['std_peak']:,.0f} kWh")
    c3.metric("95th Percentile", f"{mc_summary['peak_95th_percentile']:,.0f} kWh")
    c4.metric("99th Percentile", f"{mc_summary['peak_99th_percentile']:,.0f} kWh")

    st.divider()
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown('<div class="section-label">Exceedance Probabilities</div>', unsafe_allow_html=True)
        risk_df = pd.DataFrame({
            "Threshold (kWh)": ["18,000,000","19,000,000","20,000,000","21,000,000","22,000,000"],
            "P(Peak > Threshold)": [
                f"{mc_summary['p_exceed_18000']:.2%}",
                f"{mc_summary['p_exceed_19000']:.2%}",
                f"{mc_summary['p_exceed_20000']:.2%}",
                f"{mc_summary['p_exceed_21000']:.2%}",
                f"{mc_summary['p_exceed_22000']:.2%}",
            ],
        })
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        st.markdown(
            f'<div class="risk-note">'
            f'Simulations: {n_simulations:,}<br>'
            f'Horizon: {forecast_horizon}h<br>'
            f'σ = (Q90 − Q10) / 2.56<br>'
            f'Method: Gaussian MC'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown('<div class="section-label">Simulated Peak Demand Distribution</div>', unsafe_allow_html=True)
        sigma = np.maximum((forecast_df["upper_80"].values - forecast_df["lower_80"].values) / 2.56, 1.0)
        np.random.seed(42)
        scenarios = np.clip(
            np.random.normal(
                loc=forecast_df["median"].values[:, None],
                scale=sigma[:, None],
                size=(len(forecast_df), n_simulations),
            ), 0, None,
        )
        peak_demands = scenarios.max(axis=0)

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=peak_demands, nbinsx=45,
            marker_color="#0D4F8A", marker_line_color="#1A6AAA",
            marker_line_width=0.4, opacity=0.85,
        ))
        fig3.add_vline(x=mc_summary["mean_peak"], line_dash="dash", line_color="#5BC0F8", line_width=1.5,
                       annotation_text=f"μ={mc_summary['mean_peak']:,.0f}",
                       annotation_font=dict(color="#5BC0F8", size=9), annotation_position="top right")
        fig3.add_vline(x=mc_summary["peak_95th_percentile"], line_dash="dash", line_color="#FF6B6B", line_width=1.5,
                       annotation_text=f"P95={mc_summary['peak_95th_percentile']:,.0f}",
                       annotation_font=dict(color="#FF6B6B", size=9), annotation_position="top left")
        fig3.update_layout(**PLOT_BASE, height=320,
                           xaxis_title="Peak Demand (kWh)", yaxis_title="Frequency", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    n_days = min(7, forecast_horizon // 24)
    if n_days >= 1:
        st.markdown('<div class="section-label">Daily Peak Load Distribution</div>', unsafe_allow_html=True)
        COLORS = ["#0D4F8A","#0E5A9A","#1066AA","#1272BA","#147ECA","#168ADA","#1896EA"]
        fig4 = go.Figure()
        for day in range(n_days):
            s, e = day * 24, day * 24 + 24
            if e <= len(scenarios):
                fig4.add_trace(go.Box(
                    y=scenarios[s:e, :].max(axis=0), name=f"Day {day+1}",
                    marker_color=COLORS[day % len(COLORS)],
                    line_color="#5BC0F8", line_width=1, boxmean=True,
                ))
        fig4.update_layout(**PLOT_BASE, height=360,
                           xaxis_title="Day in Forecast Horizon",
                           yaxis_title="Peak Demand (kWh)", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — MODEL DIAGNOSTICS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Test Set Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LR MAE",         f"{m.get('linear_regression_mae',0):,.0f} kWh")
    c2.metric("LR RMSE",        f"{m.get('linear_regression_rmse',0):,.0f} kWh")
    c3.metric("QR Median MAE",  f"{m.get('quantile_median_mae',0):,.0f} kWh")
    c4.metric("QR Median RMSE", f"{m.get('quantile_median_rmse',0):,.0f} kWh")
    c5.metric("80% Coverage",   f"{m.get('empirical_coverage_80pct',0):.1%}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-label">Feature Set (17 Inputs)</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame([
            ("hour_sin / hour_cos",    "Cyclical hour-of-day encoding"),
            ("dayofweek",              "Day of week  (0 = Monday)"),
            ("month",                  "Month of year"),
            ("dayofyear",              "Day of year"),
            ("is_weekend",             "Weekend indicator  (0/1)"),
            ("is_holiday",             "Ontario statutory holiday  (0/1)"),
            ("demand_lag_1",           "Demand 1h prior"),
            ("demand_lag_24",          "Demand 24h prior"),
            ("demand_lag_168",         "Demand 168h prior  (1 week)"),
            ("rolling_mean_24",        "24h rolling mean demand"),
            ("rolling_std_24",         "24h rolling std demand"),
            ("rolling_mean_168",       "168h rolling mean demand"),
            ("rolling_std_168",        "168h rolling std demand"),
            ("price_lag_24",           "Price 24h prior"),
            ("price_lag_168",          "Price 168h prior"),
            ("price_rolling_mean_168", "168h rolling mean price"),
            ("price_rolling_std_168",  "168h rolling std price"),
        ], columns=["Feature", "Description"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="arch-body">
                <div style="margin-bottom:1rem;">
                    <span class="model-tag">BASELINE</span><br>
                    Naive 24h lag forecast — seasonal benchmark
                </div>
                <div style="margin-bottom:1rem;">
                    <span class="model-tag">LINEAR REGRESSION</span><br>
                    OLS on all 17 features — deterministic reference model
                </div>
                <div style="margin-bottom:1rem;">
                    <span class="model-tag">QUANTILE REGRESSION</span>
                    <span class="model-tag">CORE</span><br>
                    Q10 lower &nbsp;|&nbsp; Q50 median &nbsp;|&nbsp; Q90 upper<br>
                    StandardScaler preprocessing &nbsp;|&nbsp; α = 0.1
                </div>
                <div>
                    <span class="model-tag">MONTE CARLO SIMULATION</span><br>
                    σ = (Q90 − Q10) / 2.56 &nbsp;|&nbsp; Gaussian trajectories<br>
                    Peak demand distribution + exceedance probabilities
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Known Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="limits-body">
                — No weather or temperature inputs<br>
                — Linear conditional quantile assumption<br>
                — Gaussian error assumption in simulation<br>
                — No regime-switching or volatility clustering<br>
                — Holiday calendar limited to 2020–2028
            </div>
        """, unsafe_allow_html=True)