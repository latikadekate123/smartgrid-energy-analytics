import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px

# Simulate Usage Data

np.random.seed(42)
dates = pd.date_range(start=datetime.now()-timedelta(days=30), periods=720, freq='H')
usage_power = np.random.normal(loc=2.0, scale=0.5, size=len(dates))  # kW

usage_df = pd.DataFrame({
    'datetime': dates,
    'global_active_power': usage_power
})

# Simulate Forecast using Prophet

prophet_df = usage_df.rename(columns={'datetime':'ds', 'global_active_power':'y'})
model = Prophet(daily_seasonality=True)
model.fit(prophet_df)
future = model.make_future_dataframe(periods=48, freq='H')
forecast = model.predict(future)

# Simulate Complaints

appliances = ["Air Conditioner", "Refrigerator", "Heater"]
issues = [
    "consuming too much power",
    "turned off unexpectedly",
    "causing voltage fluctuations"
]
complaints = []
for _ in range(30):
    appliance = np.random.choice(appliances)
    issue = np.random.choice(issues)
    date = datetime.now() - timedelta(days=np.random.randint(1,30))
    complaints.append({"complaint_date": date, "appliance": appliance, "complaint_text": f"My {appliance} is {issue}."})

complaints_df = pd.DataFrame(complaints)

# NLP Clustering

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(complaints_df['complaint_text'])
kmeans = KMeans(n_clusters=3, random_state=42)
complaints_df['cluster'] = kmeans.fit_predict(X)

# Anomaly Detection (Z-score)

usage_df['z_score'] = (usage_df['global_active_power'] - usage_df['global_active_power'].mean()) / usage_df['global_active_power'].std()
anomalies = usage_df[np.abs(usage_df['z_score']) > 2.5]

# Streamlit Dashboard

st.title("✨Smart Grid Energy Dashboard Demo✨")

# Historical usage
st.subheader("1️⃣ Historical Usage")
fig_usage = px.line(usage_df, x='datetime', y='global_active_power', title='Global Active Power (kW)')
fig_usage.add_scatter(x=anomalies['datetime'], y=anomalies['global_active_power'],
                      mode='markers', name='Anomalies', marker=dict(color='red', size=8))
st.plotly_chart(fig_usage, use_container_width=True)

# Forecast
st.subheader("2️⃣ Forecasted Usage (Next 48 Hours)")
fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Global Active Power')
st.plotly_chart(fig_forecast, use_container_width=True)

# Complaints
st.subheader("3️⃣ Complaints Clustering")
st.dataframe(complaints_df[['complaint_date', 'complaint_text', 'cluster']])

# Probability Insights
st.subheader("4️⃣ Probability of Complaint During High Usage")
high_usage = usage_df['global_active_power'] > usage_df['global_active_power'].quantile(0.9)
prob_complaint = (len(complaints_df[complaints_df['complaint_date'].dt.hour.isin(usage_df[high_usage]['datetime'].dt.hour)])
                  / len(complaints_df)) * 100
st.write(f"Probability of complaint during top 10% high usage hours: {prob_complaint:.2f}%")

st.success("✅ Demo loaded successfully!")
