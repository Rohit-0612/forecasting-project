"""Feature engineering for the tabular models (XGBoost) and helpers.

Generates lag/rolling/calendar features from a [ds, y] weekly DataFrame
and provides a strictly time-based train/validation split.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

LAGS: List[int] = [1, 4, 8]
ROLL_MEAN_WINDOWS: List[int] = [4, 8]
ROLL_STD_WINDOWS: List[int] = [4]
VAL_WEEKS: int = 8

FEATURE_COLS: List[str] = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_STD_WINDOWS]
    + ["week_of_year", "month", "year"]
)


def add_calendar_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["year"] = dt.dt.year.astype(int)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    target_col: str = "y",
    lags: List[int] = LAGS,
    roll_means: List[int] = ROLL_MEAN_WINDOWS,
    roll_stds: List[int] = ROLL_STD_WINDOWS,
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for w in roll_means:
        out[f"roll_mean_{w}"] = out[target_col].shift(1).rolling(window=w, min_periods=1).mean()
    for w in roll_stds:
        out[f"roll_std_{w}"] = out[target_col].shift(1).rolling(window=w, min_periods=2).std()
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Take a [ds, y] DataFrame and return a feature-augmented DataFrame.

    Rows where lag features are undefined are dropped. The caller can rely
    on FEATURE_COLS to know which columns to feed the model.
    """
    feat = add_lag_rolling_features(df)
    feat = add_calendar_features(feat)
    feat = feat.dropna(subset=[f"lag_{max(LAGS)}"]).reset_index(drop=True)
    feat["roll_std_4"] = feat["roll_std_4"].fillna(0.0)
    return feat


def time_split(
    df: pd.DataFrame, val_weeks: int = VAL_WEEKS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly time-based train/validation split (no shuffling)."""
    if len(df) <= val_weeks:
        # Degenerate: keep everything as train, return empty val
        return df.copy(), df.iloc[0:0].copy()
    return df.iloc[:-val_weeks].copy(), df.iloc[-val_weeks:].copy()


def make_future_frame(history: pd.DataFrame, horizon: int = 8, freq: str = "W-SAT") -> pd.DataFrame:
    """Build a placeholder DataFrame with future ds values (y=NaN)."""
    last_ds = pd.to_datetime(history["ds"].iloc[-1])
    future_ds = pd.date_range(start=last_ds + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
    return pd.DataFrame({"ds": future_ds, "y": np.nan})


if __name__ == "__main__":
    from src.preprocess import preprocess

    series = preprocess()
    s_name = next(iter(series))
    feat = build_feature_frame(series[s_name])
    train, val = time_split(feat)
    print(f"State: {s_name}")
    print(f"Total rows after feature build: {len(feat)}")
    print(f"Train: {len(train)} | Val: {len(val)}")
    print(feat.head())
    print(f"Feature cols: {FEATURE_COLS}")
