
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# -----------------------
# Connect to SQL
# -----------------------
password = "lld%401501S"
engine = create_engine(f"postgresql+psycopg2://postgres:{password}@localhost:5432/postgres")

# -----------------------
# Streamlit Sidebar
# -----------------------
st.sidebar.title("Smart Grid Dashboard")
appliance_options = pd.read_sql("SELECT appliance_id, appliance_name FROM appliances", engine)
selected_appliance_id = st.sidebar.selectbox("Select Appliance", appliance_options['appliance_id'])
date_range = st.sidebar.date_input("Select Date Range", [])
cluster_option = st.sidebar.selectbox("Complaint Cluster (Optional)", [-1, 0, 1, 2])

# -----------------------
# Load Data
# -----------------------
usage_df = pd.read_sql(f"""
    SELECT * FROM usage_logs
    WHERE appliance_id = {selected_appliance_id}
""", engine)

forecast_df = pd.read_sql(f"""
    SELECT * FROM forecast_log
    WHERE appliance_id = {selected_appliance_id}
""", engine)

complaints_df = pd.read_sql(f"""
    SELECT * FROM feedback
    WHERE appliance_id = {selected_appliance_id}
""", engine)

# Optional: filter complaints by cluster
if cluster_option != -1:
    complaints_df = complaints_df[complaints_df['cluster'] == cluster_option]

# -----------------------
# Visualize Energy Usage
# -----------------------
st.title("Smart Grid Energy Dashboard")
st.subheader("Historical Usage")
fig_usage = px.line(usage_df, x='datetime', y='global_active_power', title='Global Active Power')
st.plotly_chart(fig_usage, use_container_width=True)

# -----------------------
# Visualize Forecast
# -----------------------
st.subheader("Predicted Usage (Next 48h)")
fig_forecast = px.line(forecast_df, x='forecast_date', y='predicted_usage', title='Forecasted Power Usage')
st.plotly_chart(fig_forecast, use_container_width=True)

# -----------------------
# Complaints & Clusters
# -----------------------
st.subheader("User Complaints")
st.dataframe(complaints_df[['complaint_date', 'complaint_text', 'cluster']])

st.write("✅ Dashboard loaded successfully!")
