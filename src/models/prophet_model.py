"""Facebook Prophet wrapper with US holidays and weekly+yearly seasonality."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetModel:
    name = "Prophet"

    def __init__(self, freq: str = "W-SAT"):
        self.freq = freq
        self.model: Prophet | None = None

    def fit(self, history: pd.DataFrame) -> "ProphetModel":
        df = history[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.8,
        )
        try:
            m.add_country_holidays(country_name="US")
        except Exception:
            pass
        m.fit(df)
        self.model = m
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit")
        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq)
        fcst = self.model.predict(future)
        return fcst["yhat"].iloc[-horizon:].to_numpy(dtype=float)

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "freq": self.freq}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ProphetModel":
        data = joblib.load(path)
        obj = cls(freq=data["freq"])
        obj.model = data["model"]
        return obj
