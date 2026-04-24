CREATE TABLE IF NOT EXISTS demand_signals (
    datetime TIMESTAMP PRIMARY KEY,
    demand_kw DOUBLE PRECISION NOT NULL,
    voltage DOUBLE PRECISION,
    global_intensity DOUBLE PRECISION,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_signals (
    datetime TIMESTAMP PRIMARY KEY,
    temperature_c DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    wind_speed_mps DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS feature_store (
    datetime TIMESTAMP PRIMARY KEY,
    demand_kw DOUBLE PRECISION NOT NULL,
    temperature_c DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    wind_speed_mps DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS drift_monitoring (
    run_ts TIMESTAMP NOT NULL DEFAULT NOW(),
    feature_name TEXT NOT NULL,
    reference_mean DOUBLE PRECISION,
    recent_mean DOUBLE PRECISION,
    mean_shift_pct DOUBLE PRECISION,
    is_drifted BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS model_metrics (
    run_ts TIMESTAMP NOT NULL DEFAULT NOW(),
    model_name TEXT NOT NULL,
    mae DOUBLE PRECISION NOT NULL,
    baseline_model TEXT,
    improvement_pct DOUBLE PRECISION,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS forecast_predictions (
    datetime TIMESTAMP NOT NULL,
    model_name TEXT NOT NULL,
    predicted_demand_kw DOUBLE PRECISION NOT NULL,
    actual_demand_kw DOUBLE PRECISION,
    run_ts TIMESTAMP NOT NULL DEFAULT NOW()
);
