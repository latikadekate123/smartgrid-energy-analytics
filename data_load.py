import pandas as pd
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# 1. Load Energy Dataset

file_path = "C:/Users/latik/Downloads/individual+household+electric+power+consumption/household_power_consumption.txt"

df = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, dayfirst=True, na_values=['?'], low_memory=False)

print(df.head())
print(df.info())

# 2. Simulate Appliance Feedback

appliances = ["Air Conditioner", "Refrigerator", "Heater", "Washing Machine", "Oven"]
issues = [
    "consuming too much power",
    "showing abnormal usage at night",
    "turned off unexpectedly",
    "causing voltage fluctuations",
    "not responding to control signals"
]

complaints = []
for _ in range(20):
    appliance = random.choice(appliances)
    issue = random.choice(issues)
    date = datetime.now() - timedelta(days=random.randint(1, 90))
    text_feedback = f"My {appliance} is {issue}."
    complaints.append({"complaint_date": date, "appliance": appliance, "complaint_text": text_feedback})

df_feedback = pd.DataFrame(complaints)
print(df_feedback.head())

# 3. Connect to PostgreSQL

password = "lld%401501S"  # URL-encoded password
engine = create_engine(f"postgresql+psycopg2://postgres:{password}@localhost:5432/postgres")

# 4. Create Schema

schema_sql = """
CREATE TABLE appliances (
    appliance_id SERIAL PRIMARY KEY,
    appliance_name TEXT UNIQUE NOT NULL
);

CREATE TABLE usage_logs (
    log_id SERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    appliance_id INT REFERENCES appliances(appliance_id),
    global_active_power FLOAT,
    global_reactive_power FLOAT,
    voltage FLOAT,
    global_intensity FLOAT,
    sub_metering_1 FLOAT,
    sub_metering_2 FLOAT,
    sub_metering_3 FLOAT
);

CREATE TABLE feedback (
    feedback_id SERIAL PRIMARY KEY,
    complaint_date TIMESTAMP,
    appliance_id INT REFERENCES appliances(appliance_id),
    complaint_text TEXT
);

CREATE TABLE forecast_log (
    forecast_id SERIAL PRIMARY KEY,
    appliance_id INT REFERENCES appliances(appliance_id),
    forecast_date TIMESTAMP,
    predicted_usage FLOAT,
    model_version TEXT
);
"""

with engine.connect() as conn:
    conn.execute(text(schema_sql))
    conn.commit()
print(" Tables created successfully!")

# 5. Insert Appliances

with engine.connect() as conn:
    for app in appliances:
        stmt = text("""
            INSERT INTO appliances (appliance_name)
            VALUES (:name)
            ON CONFLICT (appliance_name) DO NOTHING
        """)
        conn.execute(stmt, {"name": app})
    conn.commit()
print(" Appliances inserted!")

# 6. Insert Usage Logs

# Map sample usage to one appliance (Air Conditioner)
usage_sample = df.head(500).copy()
usage_sample["appliance_id"] = 1  # Example mapping

# Ensure columns match table
usage_sample = usage_sample.rename(columns={
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3"
})

usage_sample.to_sql("usage_logs", engine, if_exists="append", index=False)
print(" Sample usage logs inserted!")

# 7. Insert Feedback

# Map appliances to IDs
appliance_map = pd.read_sql("SELECT appliance_id, appliance_name FROM appliances", engine).set_index("appliance_name")["appliance_id"].to_dict()
df_feedback["appliance_id"] = df_feedback["appliance"].map(appliance_map)
df_feedback.drop(columns=["appliance"], inplace=True)
df_feedback.to_sql("feedback", engine, if_exists="append", index=False)
print(" Feedback inserted!")

# 8. Forecast Appliance Usage with Prophet

from prophet import Prophet

# Take one appliance's data (Air Conditioner)
df_ac = usage_sample[usage_sample["appliance_id"] == 1][["datetime", "global_active_power"]].copy()
df_ac.rename(columns={"datetime": "ds", "global_active_power": "y"}, inplace=True)

# Initialize Prophet model
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(df_ac)

# Make future dataframe
future = model.make_future_dataframe(periods=48, freq='H')  # next 48 hours
forecast = model.predict(future)

# Insert forecast into SQL
forecast_to_insert = forecast[['ds', 'yhat']].copy()
forecast_to_insert['appliance_id'] = 1
forecast_to_insert['model_version'] = 'Prophet_v1'
forecast_to_insert.rename(columns={'ds': 'forecast_date', 'yhat': 'predicted_usage'}, inplace=True)

forecast_to_insert.to_sql("forecast_log", engine, if_exists="append", index=False)
print(" Forecast inserted!")

# 9. NLP Clustering on Complaints

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

complaints_text = df_feedback['complaint_text'].tolist()

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(complaints_text)

# Cluster complaints into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42)
df_feedback['cluster'] = kmeans.fit_predict(X)

# print cluster groups
for i in range(3):
    print(f"\nCluster {i}:")
    print(df_feedback[df_feedback['cluster'] == i]['complaint_text'].values)

print(" Complaints clustered with NLP!")
