"""XGBoost regressor with iterative (recursive) multi-step forecasting.

Trains on lag/rolling/calendar features. At forecast time, predictions are
fed back into the lag/rolling buffer to compute the next step.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import (
    FEATURE_COLS,
    LAGS,
    ROLL_MEAN_WINDOWS,
    ROLL_STD_WINDOWS,
    build_feature_frame,
)


class XGBoostModel:
    name = "XGBoost"

    def __init__(self, freq: str = "W-SAT", **xgb_kwargs):
        self.freq = freq
        defaults = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
            objective="reg:squarederror",
        )
        defaults.update(xgb_kwargs)
        self.params = defaults
        self.model: XGBRegressor | None = None
        # Keep the last full history series (ds + y) so we can iteratively
        # forecast from it without recomputing features for every state.
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "XGBoostModel":
        self._history = history[["ds", "y"]].copy().reset_index(drop=True)
        feat = build_feature_frame(self._history)
        if len(feat) < 10:
            raise ValueError("Not enough rows to train XGBoost")
        X = feat[FEATURE_COLS].values
        y = feat["y"].values
        m = XGBRegressor(**self.params)
        m.fit(X, y)
        self.model = m
        return self

    def _build_features_for_step(self, working: pd.DataFrame, target_ds: pd.Timestamp) -> np.ndarray:
        """Build a single feature row given the (extended) history series."""
        y_series = working["y"].astype(float)
        feat = {}
        for lag in LAGS:
            feat[f"lag_{lag}"] = y_series.iloc[-lag]
        for w in ROLL_MEAN_WINDOWS:
            feat[f"roll_mean_{w}"] = y_series.iloc[-w:].mean()
        for w in ROLL_STD_WINDOWS:
            feat[f"roll_std_{w}"] = y_series.iloc[-w:].std() if w >= 2 else 0.0
        feat["roll_std_4"] = 0.0 if pd.isna(feat["roll_std_4"]) else feat["roll_std_4"]
        feat["week_of_year"] = int(target_ds.isocalendar().week)
        feat["month"] = int(target_ds.month)
        feat["year"] = int(target_ds.year)
        return np.array([[feat[c] for c in FEATURE_COLS]], dtype=float)

    def forecast(self, horizon: int) -> np.ndarray:
        if self.model is None or self._history is None:
            raise RuntimeError("Model not fit")
        working = self._history.copy()
        last_ds = pd.to_datetime(working["ds"].iloc[-1])
        future_ds = pd.date_range(
            start=last_ds + pd.tseries.frequencies.to_offset(self.freq),
            periods=horizon,
            freq=self.freq,
        )
        preds: list[float] = []
        for ds in future_ds:
            X = self._build_features_for_step(working, ds)
            yhat = float(self.model.predict(X)[0])
            preds.append(yhat)
            working = pd.concat(
                [working, pd.DataFrame({"ds": [ds], "y": [yhat]})],
                ignore_index=True,
            )
        return np.array(preds, dtype=float)

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "history": self._history,
                     "freq": self.freq, "params": self.params}, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        data = joblib.load(path)
        obj = cls(freq=data["freq"], **data["params"])
        obj.model = data["model"]
        obj._history = data["history"]
        return obj
