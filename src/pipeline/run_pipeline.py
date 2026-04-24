from __future__ import annotations

from datetime import datetime

import pandas as pd

from .config import Settings
from .data_processing import (
    build_weather_signals,
    detect_drift,
    impute_missing_values,
    inject_missing_values,
    load_demand_data,
    split_reference_recent,
)
from .db import build_feature_store, get_engine, init_schema, read_feature_store, reset_runtime_tables, upsert_dataframe
from .modeling import benchmark_models


def _persist_monitoring_and_metrics(engine, drift_df: pd.DataFrame, metrics: dict) -> None:
    drift_df.to_sql("drift_monitoring", engine, if_exists="append", index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "ARIMA(1,0,0)",
                "mae": metrics["arima_mae"],
                "baseline_model": None,
                "improvement_pct": None,
                "notes": "Univariate baseline",
            },
            {
                "model_name": "HistGradientBoostingRegressor",
                "mae": metrics["gbr_mae"],
                "baseline_model": "ARIMA(1,0,0)",
                "improvement_pct": metrics["improvement_pct"],
                "notes": "Weather + lag features",
            },
        ]
    )
    metrics_df.to_sql("model_metrics", engine, if_exists="append", index=False)

    preds_df = metrics["test_df"][["datetime", "demand_kw"]].copy()
    preds_df["model_name"] = "HistGradientBoostingRegressor"
    preds_df["predicted_demand_kw"] = metrics["gbr_preds"]
    preds_df.rename(columns={"demand_kw": "actual_demand_kw"}, inplace=True)
    preds_df = preds_df[["datetime", "model_name", "predicted_demand_kw", "actual_demand_kw"]]
    preds_df.to_sql("forecast_predictions", engine, if_exists="append", index=False)


def run() -> None:
    settings = Settings()
    engine = get_engine(settings)
    init_schema(engine)
    reset_runtime_tables(engine)

    demand_df = load_demand_data(settings.raw_data_path)
    weather_df = build_weather_signals(demand_df)

    demand_df = inject_missing_values(demand_df, ratio=0.02)
    weather_df = inject_missing_values(weather_df, ratio=0.02)

    demand_clean = impute_missing_values(demand_df)
    weather_clean = impute_missing_values(weather_df)

    upsert_dataframe(demand_clean, "demand_signals", engine)
    upsert_dataframe(weather_clean, "weather_signals", engine)

    build_feature_store(engine)
    feature_store_df = read_feature_store(engine)

    split = split_reference_recent(feature_store_df)
    drift_df = detect_drift(split["reference"], split["recent"], threshold_pct=0.15)

    metrics = benchmark_models(feature_store_df, settings.model_output_path)
    _persist_monitoring_and_metrics(engine, drift_df, metrics)

    improvement = metrics["improvement_pct"]
    print(f"Pipeline completed at {datetime.utcnow().isoformat()}Z")
    print(f"ARIMA MAE: {metrics['arima_mae']:.4f}")
    print(f"Gradient Boosted MAE: {metrics['gbr_mae']:.4f}")
    print(f"Improvement vs ARIMA: {improvement:.2f}%")
    if improvement < 15.0:
        print("Warning: improvement is below 15%; tune hyperparameters or feature engineering for your target dataset.")


if __name__ == "__main__":
    run()
