import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px


# Simulate Base Usage Data (90 days)

np.random.seed(42)
dates = pd.date_range(start=datetime.now() - timedelta(days=90), periods=2160, freq='H')
usage_power = np.random.normal(loc=2.0, scale=0.5, size=len(dates))  # kW

usage_df = pd.DataFrame({
    'datetime': dates,
    'global_active_power': usage_power
})

# Simulate appliance complaints
appliances = ["Air Conditioner", "Refrigerator", "Heater", "Washing Machine"]
issues = [
    "consuming too much power",
    "turned off unexpectedly",
    "causing voltage fluctuations",
    "making unusual noise"
]
complaints = []
for _ in range(100):
    appliance = np.random.choice(appliances)
    issue = np.random.choice(issues)
    date = datetime.now() - timedelta(days=np.random.randint(1, 90))
    complaints.append({"complaint_date": date, "appliance": appliance, "complaint_text": f"My {appliance} is {issue}."})
complaints_df = pd.DataFrame(complaints)


# Streamlit Sidebar Controls

st.sidebar.header("📅 Filters")
date_range = st.sidebar.selectbox("Select Timeframe", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"])

if date_range == "Custom Range":
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())
else:
    days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
    start_date = datetime.now() - timedelta(days=days_map[date_range])
    end_date = datetime.now()

# Filter datasets
usage_filtered = usage_df[(usage_df['datetime'] >= pd.to_datetime(start_date)) &
                          (usage_df['datetime'] <= pd.to_datetime(end_date))]
complaints_filtered = complaints_df[(complaints_df['complaint_date'] >= pd.to_datetime(start_date)) &
                                    (complaints_df['complaint_date'] <= pd.to_datetime(end_date))]



# Metrics

avg_hourly = usage_filtered['global_active_power'].mean()
avg_daily = usage_filtered.resample('D', on='datetime')['global_active_power'].mean().mean()

st.title("⚡ Smart Grid Energy Dashboard")
st.markdown(f"**Showing data from {start_date.date()} to {end_date.date()}**")

col1, col2 = st.columns(2)
col1.metric("Avg Hourly kW", f"{avg_hourly:.2f}")
col2.metric("Avg Daily kW", f"{avg_daily:.2f}")

# -----------------------
# Anomaly Detection (Dynamic Z-score)
# -----------------------
usage_filtered['z_score'] = (usage_filtered['global_active_power'] - usage_filtered['global_active_power'].mean()) / usage_filtered['global_active_power'].std()
anomalies = usage_filtered[np.abs(usage_filtered['z_score']) > 2.5]


# Historical usage chart

st.subheader("1️⃣. Historical Usage")
fig_usage = px.line(usage_filtered, x='datetime', y='global_active_power', title='Global Active Power (kW)')
fig_usage.add_scatter(x=anomalies['datetime'], y=anomalies['global_active_power'],
                      mode='markers', name='Anomalies', marker=dict(color='red', size=8))
st.plotly_chart(fig_usage, use_container_width=True)


# Forecast using Prophet

prophet_df = usage_filtered.rename(columns={'datetime':'ds', 'global_active_power':'y'})
model = Prophet(daily_seasonality=True)
model.fit(prophet_df)
future = model.make_future_dataframe(periods=48, freq='H')
forecast = model.predict(future)

st.subheader("2️⃣. Forecasted Usage (Next 48 Hours)")
fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Global Active Power')
st.plotly_chart(fig_forecast, use_container_width=True)


# Complaints Clustering

if not complaints_filtered.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(complaints_filtered['complaint_text'])
    kmeans = KMeans(n_clusters=3, random_state=42)
    complaints_filtered['cluster'] = kmeans.fit_predict(X)

    st.subheader("3️⃣. Complaints Clustering")
    st.dataframe(complaints_filtered[['complaint_date', 'appliance', 'complaint_text', 'cluster']])

    # Distribution
    st.subheader("4️⃣. Complaint Distribution")
    fig_complaints = px.histogram(complaints_filtered, x='appliance', color='cluster',
                                  title="Complaints by Appliance and Cluster")
    st.plotly_chart(fig_complaints, use_container_width=True)
else:
    st.warning("No complaints in this date range.")


# Probability of Complaint During High Usage

if not complaints_filtered.empty:
    high_usage = usage_filtered['global_active_power'] > usage_filtered['global_active_power'].quantile(0.9)
    prob_complaint = (len(complaints_filtered[complaints_filtered['complaint_date'].dt.hour.isin(
                        usage_filtered[high_usage]['datetime'].dt.hour)]) / len(complaints_filtered)) * 100
    st.subheader("5️⃣. Probability of Complaint During High Usage")
    st.write(f"**{prob_complaint:.2f}%** of complaints occurred during top 10% usage hours.")

st.success("✅ Dashboard updated with dynamic date filtering!")
