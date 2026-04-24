# Smartgrid Power System Load Forecasting

This is my end-to-end load forecasting project.
I built it to be resume-ready and easy to demo with a live Streamlit dashboard.

## What I Built

- A production-style forecasting pipeline that joins demand and weather data in PostgreSQL.
- Data quality handling for missing values and data drift checks.
- Model benchmarking between ARIMA baseline and a gradient-boosted model.
- A Streamlit app that shows metrics, drift status, and forecast vs actual values.

## Resume Highlights

- Built a production-ready forecasting pipeline joining weather and demand signals in PostgreSQL, using Python scripts to handle data drift and missing values.
- Deployed a gradient-boosted model in a Dockerized setup and benchmarked it against ARIMA using MAE.

## How The Pipeline Works

1. Load demand data from the UCI dataset (or synthetic fallback if raw file path is not provided).
2. Generate hourly weather signals.
3. Inject and impute missing values.
4. Join weather and demand into a PostgreSQL feature store.
5. Run drift checks between reference and recent windows.
6. Train and compare:
   - ARIMA(1,0,0)
   - HistGradientBoostingRegressor
7. Save metrics and predictions in PostgreSQL.

## Main Files

- `src/pipeline/run_pipeline.py`: runs the full pipeline.
- `src/pipeline/modeling.py`: ARIMA vs gradient-boosted benchmark.
- `src/pipeline/data_processing.py`: cleaning, missing data, drift prep.
- `src/pipeline/db.py`: schema setup and SQL operations.
- `sql/schema.sql`: PostgreSQL tables.
- `streamlit_modified.py`: dashboard app.

## Run Locally

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
PGHOST=localhost
PGPORT=5432
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=your_password
```

3. Run pipeline:

```bash
python -m src.pipeline.run_pipeline
```

4. Run dashboard:

```bash
streamlit run streamlit_modified.py
```

## Run With Docker

```bash
docker compose up --build pipeline
docker compose up --build streamlit
```

## Deploy And Get Shareable Link (For Resume)

GitHub link only shows code. To get a clickable live demo, deploy the app.

### Streamlit Community Cloud (recommended)

1. Push this repo to GitHub.
2. Create a managed PostgreSQL database (Neon / Supabase / Render).
3. Go to Streamlit Cloud and create a new app.
4. Select this repo and set main file path to `streamlit_modified.py`.
5. Add these secrets in Streamlit Cloud:

```toml
PGHOST="<db-host>"
PGPORT="5432"
PGDATABASE="<db-name>"
PGUSER="<db-user>"
PGPASSWORD="<db-password>"
```

6. Deploy and use the generated URL on your resume:

```text
https://<your-app-name>.streamlit.app
```

## Suggested Resume Links

- GitHub: `https://github.com/latikadekate123/Smartgrid-Power-System-Load-Forecasting`
- Live App: `https://<your-app-name>.streamlit.app`
