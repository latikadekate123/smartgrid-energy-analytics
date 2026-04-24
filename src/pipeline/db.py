from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import Settings


def get_engine(settings: Settings) -> Engine:
    return create_engine(settings.sqlalchemy_url)


def init_schema(engine: Engine, schema_path: str = "sql/schema.sql") -> None:
    sql_text = Path(schema_path).read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql_text))


def upsert_dataframe(df: pd.DataFrame, table_name: str, engine: Engine) -> None:
    if df.empty:
        return
    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)


def reset_runtime_tables(engine: Engine) -> None:
    tables = [
        "forecast_predictions",
        "model_metrics",
        "drift_monitoring",
        "feature_store",
        "weather_signals",
        "demand_signals",
    ]
    with engine.begin() as conn:
        for table in tables:
            conn.execute(text(f"TRUNCATE TABLE {table}"))


def build_feature_store(engine: Engine) -> None:
    query = """
    INSERT INTO feature_store (datetime, demand_kw, temperature_c, humidity_pct, wind_speed_mps)
    SELECT d.datetime,
           d.demand_kw,
           w.temperature_c,
           w.humidity_pct,
           w.wind_speed_mps
    FROM demand_signals d
    JOIN weather_signals w USING (datetime)
    ON CONFLICT (datetime) DO UPDATE
    SET demand_kw = EXCLUDED.demand_kw,
        temperature_c = EXCLUDED.temperature_c,
        humidity_pct = EXCLUDED.humidity_pct,
        wind_speed_mps = EXCLUDED.wind_speed_mps
    """
    with engine.begin() as conn:
        conn.execute(text(query))


def read_feature_store(engine: Engine) -> pd.DataFrame:
    query = """
    SELECT datetime, demand_kw, temperature_c, humidity_pct, wind_speed_mps
    FROM feature_store
    ORDER BY datetime
    """
    return pd.read_sql(query, engine)
