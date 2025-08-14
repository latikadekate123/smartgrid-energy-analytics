# Smart Grid Energy Analytics

This is a project I built to monitor and detect anomalies in household appliance energy consumption.
It’s a complete ETL + analytics + visualization pipeline — from raw data → clean dataset → usage metrics → anomaly detection → interactive dashboard.

---
## Dataset

The project uses smart grid household appliance consumption data : **UCI Smart Grid Household Energy Dataset**.  

| Column Name                        | Description                                 |
| ---------------------------------- | ------------------------------------------- |
| `datetime`                         | Timestamp of the reading                    |
| `appliance_id` or `appliance_name` | Which appliance the reading is for          |
| `global_active_power`              | Active power consumption in kW              |
| `global_reactive_power`            | Reactive power in kW                        |
| `voltage`                          | Voltage at that timestamp                   |
| `global_intensity`                 | Current intensity in A                      |
| `sub_metering_1/2/3`               | Power usage of specific zones or appliances |

---

## Features
- **ETL Pipeline**:  
  - Read raw smart grid appliance consumption data.  
  - Clean and standardize timestamps, missing values, and column names.  
  - Saves the cleaned dataset locally or to a database for further analysis.
  
- **Usage Metrics**:  
  - Calculates average hourly and daily kWh for each appliance.
  - Summarizes total household consumption for quick insights.

- **Anomaly Detection**:  
  - Z-score method to flag unusual consumption patterns.

- **Interactive Dashboard (Streamlit)**:  
  - Shows appliance-wise consumption trends over time.
  - Displays the distribution of anomalies.
  - Lets you filter by date range, appliance type, and choose time aggregation

 ## Example Insights You Can Get
  
  - Which appliance consumes the most energy over a given period.
  - Hours/days when consumption spikes unexpectedly.
  - Trends in household usage over time.
