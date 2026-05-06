"""SARIMA model wrapper using pmdarima.auto_arima.

We expose a small ARIMAModel class with fit/forecast/save/load so the
selector can treat all four model families uniformly.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    name = "ARIMA"

    def __init__(self, seasonal_period: int = 52):
        self.seasonal_period = seasonal_period
        self.model = None
        self._uses_pmdarima = False

    def fit(self, history: pd.DataFrame) -> "ARIMAModel":
        y = history["y"].astype(float).values
        # auto_arima with seasonality of 52 on a ~250-row series can be slow
        # and unstable; cap m at min(seasonal_period, len/2) and let it pick.
        m = self.seasonal_period if len(y) >= 2 * self.seasonal_period else 1
        if HAS_PMDARIMA:
            try:
                self.model = auto_arima(
                    y,
                    seasonal=(m > 1),
                    m=m,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=3, max_q=3, max_P=1, max_Q=1, max_d=2, max_D=1,
                    information_criterion="aic",
                )
                self._uses_pmdarima = True
                return self
            except Exception:
                self.model = None
        # Fallback: a small SARIMAX(1,1,1)(1,0,0,m) — good enough as a baseline.
        try:
            order = (1, 1, 1)
            seasonal_order = (1, 0, 0, m) if m > 1 else (0, 0, 0, 0)
            self.model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        except Exception:
            self.model = SARIMAX(y, order=(1, 1, 1), enforce_stationarity=False).fit(disp=False)
        self._uses_pmdarima = False
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit")
        if self._uses_pmdarima:
            preds = self.model.predict(n_periods=horizon)
            return np.asarray(preds, dtype=float)
        preds = self.model.forecast(steps=horizon)
        return np.asarray(preds, dtype=float)

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "uses_pmdarima": self._uses_pmdarima,
                     "seasonal_period": self.seasonal_period}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ARIMAModel":
        data = joblib.load(path)
        obj = cls(seasonal_period=data["seasonal_period"])
        obj.model = data["model"]
        obj._uses_pmdarima = data["uses_pmdarima"]
        return obj
