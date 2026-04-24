import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from urllib.parse import quote_plus


st.set_page_config(page_title="Smartgrid Forecasting Dashboard", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg-primary: #081322;
        --bg-accent: #102942;
        --card-bg: #112d4a;
        --text-main: #f3f6fb;
        --text-muted: #9fb3c8;
        --ok: #16a34a;
        --warn: #f59e0b;
    }
    .stApp {
        background: radial-gradient(circle at 20% 20%, #12375d 0%, var(--bg-primary) 55%, #050b15 100%);
    }
    .hero {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(143, 194, 255, 0.2);
        border-radius: 14px;
        background: linear-gradient(115deg, rgba(14, 52, 84, 0.9), rgba(9, 25, 41, 0.92));
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.1rem;
        letter-spacing: 0.03rem;
        color: var(--text-muted);
        margin: 0;
    }
    .hero-subtitle {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0 0.2rem 0;
        color: var(--text-main);
    }
    .hero-caption {
        color: var(--text-muted);
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _secret_or_env(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def _get_engine(host: str, port: str, db: str, user: str, password: str):
    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    return create_engine(f"postgresql+psycopg2://{encoded_user}:{encoded_password}@{host}:{port}/{db}")


@st.cache_data(ttl=20)
def _read_table(query: str, host: str, port: str, db: str, user: str, password: str) -> pd.DataFrame:
    engine = _get_engine(host, port, db, user, password)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def load_data(host: str, port: str, db: str, user: str, password: str):
    feature_df = _read_table("SELECT * FROM feature_store ORDER BY datetime", host, port, db, user, password)
    drift_df = _read_table("SELECT * FROM drift_monitoring ORDER BY run_ts DESC", host, port, db, user, password)
    metrics_df = _read_table("SELECT * FROM model_metrics ORDER BY run_ts DESC", host, port, db, user, password)
    preds_df = _read_table("SELECT * FROM forecast_predictions ORDER BY datetime", host, port, db, user, password)
    return feature_df, drift_df, metrics_df, preds_df


def _run_pipeline(host: str, port: str, db: str, user: str, password: str) -> tuple[bool, str]:
    run_env = os.environ.copy()
    run_env.update(
        {
            "PGHOST": host,
            "PGPORT": port,
            "PGDATABASE": db,
            "PGUSER": user,
            "PGPASSWORD": password,
        }
    )
    proc = subprocess.run(
        [sys.executable, "-m", "src.pipeline.run_pipeline"],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=os.getcwd(),
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


st.sidebar.header("Connection")
default_host = _secret_or_env("PGHOST", "localhost")
default_port = _secret_or_env("PGPORT", "5432")
default_db = _secret_or_env("PGDATABASE", "postgres")
default_user = _secret_or_env("PGUSER", "postgres")
default_password = _secret_or_env("PGPASSWORD", "")

host = st.sidebar.text_input("PGHOST", value=default_host)
port = st.sidebar.text_input("PGPORT", value=default_port)
db = st.sidebar.text_input("PGDATABASE", value=default_db)
user = st.sidebar.text_input("PGUSER", value=default_user)
password = st.sidebar.text_input("PGPASSWORD", value=default_password, type="password")

run_pipeline_clicked = st.sidebar.button("Run Pipeline Now", type="primary", use_container_width=True)
refresh_clicked = st.sidebar.button("Refresh Dashboard", use_container_width=True)

if run_pipeline_clicked:
    with st.sidebar:
        with st.spinner("Running pipeline..."):
            ok, run_output = _run_pipeline(host, port, db, user, password)
    if ok:
        st.sidebar.success("Pipeline finished successfully.")
        _read_table.clear()
    else:
        st.sidebar.error("Pipeline failed. Check output below.")
    with st.expander("Pipeline run output", expanded=not ok):
        st.code(run_output or "No output captured")

if refresh_clicked:
    _read_table.clear()

st.markdown(
    """
    <div class="hero">
      <p class="hero-title">Production Monitoring Dashboard</p>
      <p class="hero-subtitle">Smartgrid Forecasting Monitoring</p>
      <p class="hero-caption">PostgreSQL feature store, drift checks, model benchmark, and forecast tracking</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    feature_store, drift_monitoring, model_metrics, forecasts = load_data(host, port, db, user, password)
except Exception as exc:
    st.error("Could not load dashboard data from PostgreSQL.")
    st.code(str(exc))
    st.info("Set valid DB credentials in the sidebar, then click Run Pipeline Now.")
    st.stop()

if feature_store.empty:
    st.warning("No data in feature_store yet. Run the pipeline and refresh.")
    st.stop()

feature_store["datetime"] = pd.to_datetime(feature_store["datetime"])
if not forecasts.empty:
    forecasts["datetime"] = pd.to_datetime(forecasts["datetime"])

latest_gbr = model_metrics[model_metrics["model_name"] == "HistGradientBoostingRegressor"].head(1)
latest_improvement = float(latest_gbr["improvement_pct"].iloc[0]) if not latest_gbr.empty else 0.0
latest_mae = float(latest_gbr["mae"].iloc[0]) if not latest_gbr.empty else 0.0
drift_count = int(drift_monitoring["is_drifted"].sum()) if not drift_monitoring.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Feature Store Rows", f"{len(feature_store):,}")
col2.metric("Latest GBR MAE", f"{latest_mae:.4f}")
col3.metric("Improvement vs ARIMA", f"{latest_improvement:.2f}%")
col4.metric("Drifted Features", f"{drift_count}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Demand and Weather Timeline")
    fig_demand = px.line(
        feature_store,
        x="datetime",
        y=["demand_kw", "temperature_c"],
        labels={"value": "Value", "variable": "Signal"},
    )
    st.plotly_chart(fig_demand, use_container_width=True)

with right:
    st.subheader("Model MAE Comparison")
    if model_metrics.empty:
        st.info("No model metrics available yet.")
    else:
        latest_run = model_metrics["run_ts"].max()
        current = model_metrics[model_metrics["run_ts"] == latest_run].copy()
        fig_mae = px.bar(current, x="model_name", y="mae", color="model_name")
        st.plotly_chart(fig_mae, use_container_width=True)

st.subheader("Forecast vs Actual")
if forecasts.empty:
    st.info("No forecasts available yet.")
else:
    fig_fc = px.line(
        forecasts,
        x="datetime",
        y=["actual_demand_kw", "predicted_demand_kw"],
        labels={"value": "kW", "variable": "Series"},
    )
    st.plotly_chart(fig_fc, use_container_width=True)

st.subheader("Data Drift Monitoring")
if drift_monitoring.empty:
    st.info("No drift records available yet.")
else:
    drift_view = drift_monitoring[["run_ts", "feature_name", "mean_shift_pct", "is_drifted"]].copy()
    drift_view["mean_shift_pct"] = (drift_view["mean_shift_pct"] * 100).round(2)
    st.dataframe(drift_view, use_container_width=True)

if latest_improvement >= 15:
    st.success("Gradient boosted model is outperforming ARIMA by at least 15% MAE.")
else:
    st.warning("Improvement is currently below 15%. Tune model hyperparameters or features before final resume claim.")
