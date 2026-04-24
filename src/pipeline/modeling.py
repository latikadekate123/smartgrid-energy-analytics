from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


def _build_supervised_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = pd.to_datetime(out["datetime"]).dt.hour
    out["day_of_week"] = pd.to_datetime(out["datetime"]).dt.dayofweek
    out["lag_1"] = out["demand_kw"].shift(1)
    out["lag_24"] = out["demand_kw"].shift(24)
    return out.dropna().reset_index(drop=True)


def train_test_split_time(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def evaluate_arima_baseline(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[float, np.ndarray]:
    model = ARIMA(train["demand_kw"], order=(1, 0, 0))
    fit = model.fit()
    preds = fit.forecast(steps=len(test)).to_numpy()
    mae = float(mean_absolute_error(test["demand_kw"], preds))
    return mae, preds


def train_gradient_boosted(train: pd.DataFrame, test: pd.DataFrame, model_output_path: str) -> Tuple[float, np.ndarray]:
    feature_cols = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_mps",
        "hour",
        "day_of_week",
        "lag_1",
        "lag_24",
    ]
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=350,
        random_state=42,
    )
    model.fit(train[feature_cols], train["demand_kw"])
    preds = model.predict(test[feature_cols])
    mae = float(mean_absolute_error(test["demand_kw"], preds))

    out_path = Path(model_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return mae, preds


def benchmark_models(feature_store_df: pd.DataFrame, model_output_path: str) -> dict:
    supervised = _build_supervised_features(feature_store_df)
    train_df, test_df = train_test_split_time(supervised, test_ratio=0.2)

    arima_mae, arima_preds = evaluate_arima_baseline(train_df, test_df)
    gbr_mae, gbr_preds = train_gradient_boosted(train_df, test_df, model_output_path)

    improvement_pct = ((arima_mae - gbr_mae) / arima_mae) * 100 if arima_mae else 0.0

    metrics = {
        "train_df": train_df,
        "test_df": test_df,
        "arima_mae": arima_mae,
        "gbr_mae": gbr_mae,
        "improvement_pct": float(improvement_pct),
        "arima_preds": arima_preds,
        "gbr_preds": gbr_preds,
    }
    return metrics
