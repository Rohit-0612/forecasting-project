"""2-layer LSTM with dropout for univariate weekly forecasting.

Implementation uses PyTorch (a TensorFlow install was not workable in
this environment — the model API stays the same as the rest of the
project so the selector treats it like any other model).

- Sequence length 12 weeks as input window
- Two stacked LSTM layers with dropout in between
- Data scaled with MinMaxScaler
- 50 epochs by default
- Recursive multi-step forecasting
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

torch.manual_seed(42)
np.random.seed(42)


class _LSTMNet(nn.Module):
    def __init__(self, units: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=units, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=units, hidden_size=units, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


class LSTMModel:
    name = "LSTM"

    def __init__(self, seq_len: int = 12, epochs: int = 50, batch_size: int = 8,
                 units: int = 32, dropout: float = 0.2, lr: float = 1e-3):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.dropout = dropout
        self.lr = lr
        self.scaler: MinMaxScaler | None = None
        self.model: _LSTMNet | None = None
        self._last_window: np.ndarray | None = None
        self._min_rows = seq_len + 8

    def _make_sequences(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i - self.seq_len:i, 0])
            y.append(scaled[i, 0])
        return np.array(X, dtype=np.float32)[..., None], np.array(y, dtype=np.float32)

    def fit(self, history: pd.DataFrame) -> "LSTMModel":
        y_raw = history["y"].astype(float).values.reshape(-1, 1)
        if len(y_raw) < self._min_rows:
            raise ValueError(f"Need at least {self._min_rows} rows; got {len(y_raw)}")
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(y_raw)
        X_np, y_np = self._make_sequences(scaled)
        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)

        self.model = _LSTMNet(units=self.units, dropout=self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n = X.shape[0]
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self.model(X[idx])
                loss = loss_fn(pred, y[idx])
                loss.backward()
                opt.step()
        self.model.eval()
        self._last_window = scaled[-self.seq_len:].astype(np.float32).copy()
        return self

    @torch.no_grad()
    def forecast(self, horizon: int) -> np.ndarray:
        if self.model is None or self.scaler is None or self._last_window is None:
            raise RuntimeError("Model not fit")
        window = self._last_window.copy()
        preds_scaled: list[float] = []
        for _ in range(horizon):
            x = torch.from_numpy(window.reshape(1, self.seq_len, 1))
            yhat = float(self.model(x).item())
            preds_scaled.append(yhat)
            window = np.vstack([window[1:], [[yhat]]]).astype(np.float32)
        preds = self.scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
        return preds.flatten()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch_path = path.with_suffix(".pt")
        torch.save(self.model.state_dict(), torch_path)
        joblib.dump({
            "scaler": self.scaler,
            "last_window": self._last_window,
            "seq_len": self.seq_len,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "units": self.units,
            "dropout": self.dropout,
            "lr": self.lr,
            "torch_path": str(torch_path),
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        meta = joblib.load(path)
        obj = cls(
            seq_len=meta["seq_len"], epochs=meta["epochs"],
            batch_size=meta["batch_size"], units=meta["units"],
            dropout=meta["dropout"], lr=meta["lr"],
        )
        obj.scaler = meta["scaler"]
        obj._last_window = meta["last_window"]
        obj.model = _LSTMNet(units=obj.units, dropout=obj.dropout)
        obj.model.load_state_dict(torch.load(meta["torch_path"], weights_only=True))
        obj.model.eval()
        return obj
