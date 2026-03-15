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
import plotly.express as px
import streamlit as st
from app.model import load_models, predict

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ontario Electricity Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Load models (cached so they only load once)
# ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

try:
    bundle = get_models()
    models_ok = True
except FileNotFoundError:
    models_ok = False

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Ontario Energy")
    st.markdown("**Probabilistic Demand Forecasting**")
    st.divider()

    st.subheader("📁 Data Input")
    uploaded_file = st.file_uploader(
        "Upload CSV (ontario_electricity_demand.csv)",
        type=["csv"],
        help="Must contain columns: date, hour, hourly_demand, hourly_average_price"
    )

    st.divider()
    st.subheader("⚙️ Forecast Settings")
    forecast_horizon = st.slider(
        "Forecast Horizon (hours)", min_value=6, max_value=168, value=24, step=6
    )
    n_simulations = st.slider(
        "Monte Carlo Simulations", min_value=100, max_value=2000, value=500, step=100
    )
    run_mc = st.toggle("Run Monte Carlo Risk Analysis", value=True)

    st.divider()
    st.caption("Coverage (80% interval): **80.9%**")
    st.caption("Model: Quantile Regression")
    st.caption("Train cutoff: 2019-01-01")

# ─────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"] - 1, unit="h")
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df[["hourly_demand", "hourly_average_price"]].rename(
        columns={"hourly_demand": "demand_kwh", "hourly_average_price": "price_cents"}
    )
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(expected).interpolate(method="time")
    return df

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.title("⚡ Ontario Electricity Demand Forecast")
st.markdown("Probabilistic forecasting with uncertainty quantification and peak-load risk analysis.")
st.divider()

if not models_ok:
    st.error(
        "⚠️ Trained models not found. "
        "Run `python -m app.model --data data/ontario_electricity_demand.csv` first."
    )
    st.stop()

if uploaded_file is None:
    st.info("👈 Upload your CSV file in the sidebar to get started.")

    # Show sample metrics while waiting
    st.subheader("Model Performance (Test Set)")
    m = bundle.metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("LR MAE", f"{m['linear_regression_mae']:,.0f}")
    col2.metric("LR RMSE", f"{m['linear_regression_rmse']:,.0f}")
    col3.metric("QR MAE", f"{m['quantile_median_mae']:,.0f}")
    col4.metric("QR RMSE", f"{m['quantile_median_rmse']:,.0f}")
    col5.metric("80% Coverage", f"{m['empirical_coverage_80pct']:.1%}")
    st.stop()

# ── Data loaded ───────────────────────────────────────────────────────────────
df = load_and_prepare(uploaded_file)

if len(df) < 168:
    st.error("Not enough data — need at least 168 hours of history.")
    st.stop()

recent_demand = df["demand_kwh"].tolist()
recent_price  = df["price_cents"].tolist()

# ── Run forecast ──────────────────────────────────────────────────────────────
with st.spinner("Running forecast..."):
    forecast_list, mc_summary = predict(
        bundle=bundle,
        recent_demand=recent_demand,
        recent_price=recent_price,
        forecast_horizon=forecast_horizon,
        n_simulations=n_simulations,
        run_monte_carlo=run_mc,
    )

forecast_df = pd.DataFrame(forecast_list)

# Build a time index starting from the last timestamp in the data
last_ts = df.index[-1]
forecast_df["timestamp"] = pd.date_range(
    start=last_ts + pd.Timedelta(hours=1),
    periods=len(forecast_df),
    freq="h"
)

# ─────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🎲 Risk Analysis", "📊 Model Info"])

# ══════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════
with tab1:
    # Key metrics row
    st.subheader("Forecast Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Median Peak Demand",
        f"{forecast_df['median'].max():,.0f} kWh"
    )
    col2.metric(
        "Forecast Horizon",
        f"{forecast_horizon}h"
    )
    col3.metric(
        "Uncertainty Range",
        f"±{(forecast_df['upper_80'] - forecast_df['lower_80']).mean() / 2:,.0f} kWh"
    )
    col4.metric(
        "Interval Coverage",
        "80.9%"
    )

    st.divider()

    # Fan chart
    st.subheader(f"Probabilistic Demand Forecast — Next {forecast_horizon} Hours")

    fig = go.Figure()

    # 80% prediction band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
        y=pd.concat([forecast_df["upper_80"], forecast_df["lower_80"][::-1]]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Prediction Interval",
        hoverinfo="skip",
    ))

    # Median forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["median"],
        mode="lines",
        name="Median Forecast",
        line=dict(color="#636EFA", width=2.5),
    ))

    # Upper/lower bounds
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["upper_80"],
        mode="lines", name="Upper 80%",
        line=dict(color="#636EFA", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["lower_80"],
        mode="lines", name="Lower 80%",
        line=dict(color="#636EFA", width=1, dash="dot"),
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Demand (kWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Historical context
    st.subheader("Historical Demand (Last 7 Days)")
    history_7d = df["demand_kwh"].tail(168)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=history_7d.index, y=history_7d.values,
        mode="lines", name="Actual Demand",
        line=dict(color="#EF553B", width=1.5),
    ))
    fig2.update_layout(
        xaxis_title="Time", yaxis_title="Demand (kWh)",
        height=300, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Raw forecast table
    with st.expander("View Raw Forecast Data"):
        display_df = forecast_df[["timestamp", "lower_80", "median", "upper_80"]].copy()
        display_df.columns = ["Timestamp", "Lower 80%", "Median", "Upper 80%"]
        display_df = display_df.set_index("Timestamp")
        st.dataframe(display_df.style.format("{:,.0f}"), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    if not run_mc or mc_summary is None:
        st.info("Enable Monte Carlo Risk Analysis in the sidebar to see this tab.")
        st.stop()

    st.subheader("Monte Carlo Peak Demand Risk")

    # Key risk metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Peak", f"{mc_summary['mean_peak']:,.0f} kWh")
    col2.metric("Std Deviation", f"{mc_summary['std_peak']:,.0f} kWh")
    col3.metric("95th Percentile", f"{mc_summary['peak_95th_percentile']:,.0f} kWh")
    col4.metric("99th Percentile", f"{mc_summary['peak_99th_percentile']:,.0f} kWh")

    st.divider()

    col_left, col_right = st.columns(2)

    # Exceedance probability table
    with col_left:
        st.subheader("Exceedance Probabilities")
        st.markdown("*Probability that peak demand exceeds threshold over the forecast horizon*")

        risk_data = {
            "Threshold (kWh)": ["18,000,000", "19,000,000", "20,000,000", "21,000,000", "22,000,000"],
            "P(Peak > Threshold)": [
                f"{mc_summary['p_exceed_18000']:.1%}",
                f"{mc_summary['p_exceed_19000']:.1%}",
                f"{mc_summary['p_exceed_20000']:.1%}",
                f"{mc_summary['p_exceed_21000']:.1%}",
                f"{mc_summary['p_exceed_22000']:.1%}",
            ]
        }
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    # Simulated peak distribution chart
    with col_right:
        st.subheader("Simulated Peak Distribution")

        # Reconstruct MC scenarios for histogram
        y_med = forecast_df["median"].values
        y_low = forecast_df["lower_80"].values
        y_high = forecast_df["upper_80"].values
        sigma = np.maximum((y_high - y_low) / 2.56, 1.0)

        np.random.seed(42)
        scenarios = np.random.normal(
            loc=y_med[:, None], scale=sigma[:, None],
            size=(len(y_med), n_simulations)
        )
        scenarios = np.clip(scenarios, 0, None)
        peak_demands = scenarios.max(axis=0)

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=peak_demands, nbinsx=40,
            marker_color="#636EFA", opacity=0.75,
            name="Simulated Peaks"
        ))
        fig3.add_vline(
            x=mc_summary["mean_peak"], line_dash="dash", line_color="blue",
            annotation_text=f"Mean: {mc_summary['mean_peak']:,.0f}",
            annotation_position="top right"
        )
        fig3.add_vline(
            x=mc_summary["peak_95th_percentile"], line_dash="dash", line_color="red",
            annotation_text=f"95th: {mc_summary['peak_95th_percentile']:,.0f}",
            annotation_position="top left"
        )
        fig3.update_layout(
            xaxis_title="Peak Demand (kWh)",
            yaxis_title="Frequency",
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Daily peak boxplots
    st.subheader("Daily Peak Load Distribution")
    n_days = min(7, forecast_horizon // 24)

    if n_days >= 1:
        daily_peaks = []
        for day in range(n_days):
            start = day * 24
            end   = start + 24
            if end <= len(scenarios):
                daily_peaks.append(scenarios[start:end, :].max(axis=0))

        if daily_peaks:
            fig4 = go.Figure()
            for i, dp in enumerate(daily_peaks):
                fig4.add_trace(go.Box(
                    y=dp, name=f"Day {i+1}",
                    marker_color="#00CC96", boxmean=True,
                ))
            fig4.update_layout(
                yaxis_title="Peak Demand (kWh)",
                xaxis_title="Day in Forecast Horizon",
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Set forecast horizon to at least 24 hours to see daily breakdown.")

# ══════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Metrics")

    m = bundle.metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("LR MAE",       f"{m['linear_regression_mae']:,.0f} kWh")
    col2.metric("LR RMSE",      f"{m['linear_regression_rmse']:,.0f} kWh")
    col3.metric("QR MAE",       f"{m['quantile_median_mae']:,.0f} kWh")
    col4.metric("QR RMSE",      f"{m['quantile_median_rmse']:,.0f} kWh")
    col5.metric("80% Coverage", f"{m['empirical_coverage_80pct']:.1%}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Features Used")
        features = [
            ("hour_sin, hour_cos", "Cyclical hour encoding"),
            ("dayofweek", "Day of week (0=Mon, 6=Sun)"),
            ("month", "Month of year"),
            ("dayofyear", "Day of year"),
            ("is_weekend", "Weekend indicator"),
            ("is_holiday", "Ontario statutory holiday"),
            ("demand_lag_1", "Demand 1 hour ago"),
            ("demand_lag_24", "Demand 24 hours ago"),
            ("demand_lag_168", "Demand 168 hours ago (1 week)"),
            ("rolling_mean_24", "24h rolling mean demand"),
            ("rolling_std_24", "24h rolling std demand"),
            ("rolling_mean_168", "168h rolling mean demand"),
            ("rolling_std_168", "168h rolling std demand"),
            ("price_lag_24", "Price 24 hours ago"),
            ("price_lag_168", "Price 168 hours ago"),
            ("price_rolling_mean_168", "168h rolling mean price"),
            ("price_rolling_std_168", "168h rolling std price"),
        ]
        feat_df = pd.DataFrame(features, columns=["Feature", "Description"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Model Architecture")
        st.markdown("""
        **Baseline**
        - Naive 24h lag forecast

        **Linear Regression**
        - Deterministic point forecast
        - OLS on all 17 features

        **Quantile Regression (Core)**
        - Q10 → lower bound
        - Q50 → median forecast
        - Q90 → upper bound
        - StandardScaler preprocessing
        - Alpha = 0.1 (L1 regularization)

        **Monte Carlo Simulation**
        - Derives σ from quantile spread: `(Q90 - Q10) / 2.56`
        - Samples N Gaussian trajectories
        - Reports peak demand distribution and exceedance probabilities
        """)

        st.subheader("Known Limitations")
        st.markdown("""
        - No weather / temperature inputs
        - Linear conditional quantile assumption
        - Gaussian error assumption in simulation
        - No regime-switching or volatility clustering
        """)