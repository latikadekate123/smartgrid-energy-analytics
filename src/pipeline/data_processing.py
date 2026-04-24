from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "Global_active_power": "demand_kw",
            "Global_reactive_power": "global_reactive_power",
            "Voltage": "voltage",
            "Global_intensity": "global_intensity",
            "Sub_metering_1": "sub_metering_1",
            "Sub_metering_2": "sub_metering_2",
            "Sub_metering_3": "sub_metering_3",
        }
    )


def load_demand_data(raw_data_path: str) -> pd.DataFrame:
    if raw_data_path:
        df = pd.read_csv(
            raw_data_path,
            sep=";",
            parse_dates={"datetime": ["Date", "Time"]},
            dayfirst=True,
            na_values=["?"],
            low_memory=False,
        )
        df = _normalize_columns(df)
        for col in ["demand_kw", "voltage", "global_intensity"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["datetime", "demand_kw", "voltage", "global_intensity"]]
        df = (
            df.set_index("datetime")
            .sort_index()
            .resample("H")
            .mean(numeric_only=True)
            .reset_index()
        )
        df["source"] = "uci_household"
        return df

    periods = 24 * 180
    idx = pd.date_range(start=datetime(2024, 1, 1), periods=periods, freq="H")
    hour = idx.hour.to_numpy()
    day_of_year = idx.dayofyear.to_numpy()
    weekly = np.sin(2 * np.pi * idx.dayofweek.to_numpy() / 7)
    seasonal = np.sin(2 * np.pi * day_of_year / 365)
    demand = 2.3 + 0.9 * np.sin(2 * np.pi * hour / 24) + 0.6 * weekly + 0.35 * seasonal
    demand += np.random.default_rng(42).normal(0, 0.12, size=periods)

    df = pd.DataFrame(
        {
            "datetime": idx,
            "demand_kw": demand,
            "voltage": 236 + np.random.default_rng(7).normal(0, 2.5, size=periods),
            "global_intensity": np.maximum(4.2, demand * 3.2 + np.random.default_rng(99).normal(0, 0.8, size=periods)),
            "source": "synthetic",
        }
    )
    return df


def build_weather_signals(demand_df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(demand_df["datetime"])
    hours = ts.dt.hour.to_numpy()
    day_of_year = ts.dt.dayofyear.to_numpy()
    rng = np.random.default_rng(123)

    temp = 18 + 9 * np.sin(2 * np.pi * (hours - 14) / 24) + 5 * np.sin(2 * np.pi * day_of_year / 365)
    temp += rng.normal(0, 1.0, size=len(ts))
    humidity = 60 - 0.8 * temp + rng.normal(0, 3.0, size=len(ts))
    wind = 2.2 + 1.1 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.35, size=len(ts))

    weather = pd.DataFrame(
        {
            "datetime": ts,
            "temperature_c": temp,
            "humidity_pct": np.clip(humidity, 20, 95),
            "wind_speed_mps": np.clip(wind, 0.2, None),
        }
    )
    return weather


def inject_missing_values(df: pd.DataFrame, ratio: float = 0.02) -> pd.DataFrame:
    out = df.copy()
    n_rows = len(out)
    n_missing = max(1, int(n_rows * ratio))
    rng = np.random.default_rng(2026)

    for col in [c for c in out.columns if c not in {"datetime", "source"}]:
        idx = rng.choice(n_rows, size=n_missing, replace=False)
        out.loc[idx, col] = np.nan
    return out


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("datetime").copy()
    cols = [c for c in out.columns if c not in {"datetime", "source"}]
    out[cols] = out[cols].ffill().bfill()
    for col in cols:
        out[col] = out[col].fillna(out[col].median())
    return out


def detect_drift(reference_df: pd.DataFrame, recent_df: pd.DataFrame, threshold_pct: float = 0.15) -> pd.DataFrame:
    rows = []
    for feature in ["demand_kw", "temperature_c", "humidity_pct", "wind_speed_mps"]:
        ref_mean = float(reference_df[feature].mean())
        recent_mean = float(recent_df[feature].mean())
        denom = abs(ref_mean) if abs(ref_mean) > 1e-6 else 1.0
        mean_shift_pct = abs(recent_mean - ref_mean) / denom
        rows.append(
            {
                "feature_name": feature,
                "reference_mean": ref_mean,
                "recent_mean": recent_mean,
                "mean_shift_pct": mean_shift_pct,
                "is_drifted": bool(mean_shift_pct > threshold_pct),
            }
        )
    return pd.DataFrame(rows)


def split_reference_recent(df: pd.DataFrame, reference_ratio: float = 0.7) -> Dict[str, pd.DataFrame]:
    cut = int(len(df) * reference_ratio)
    return {"reference": df.iloc[:cut].copy(), "recent": df.iloc[cut:].copy()}
