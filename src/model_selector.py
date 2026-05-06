"""Per-state model selection and forecast generation pipeline.

For each state:
  1. Split history into train/validation (last 8 weeks = validation).
  2. Fit each candidate model on train and forecast 8 weeks.
  3. Score forecasts on validation with RMSE (primary) and MAE (secondary).
  4. Pick the best model, retrain it on the FULL history, then forecast 8
     weeks into the future.
  5. Persist the selection report and the final forecasts.

This module is also the training entry point used by the pipeline.
"""
from __future__ import annotations

import os

# Cap thread pools BEFORE importing numpy/xgboost/torch to avoid pthread
# mutex init errors observed on some macOS sandbox environments.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import mae, rmse  # noqa: E402
from src.models.arima_model import ARIMAModel  # noqa: E402
from src.models.lstm_model import LSTMModel  # noqa: E402
from src.models.prophet_model import ProphetModel  # noqa: E402
from src.models.xgboost_model import XGBoostModel  # noqa: E402
from src.preprocess import WEEKLY_FREQ, preprocess  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

VAL_WEEKS = 8
HORIZON = 8
LSTM_MIN_ROWS = 24  # need >= seq_len(12) + some training samples
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models_saved"


def _candidates_for(history: pd.DataFrame) -> List[str]:
    """Pick which models are eligible to compete on this series."""
    n = len(history)
    cands = ["ARIMA", "Prophet", "XGBoost"]
    if n >= LSTM_MIN_ROWS:
        cands.append("LSTM")
    return cands


def _build(name: str):
    if name == "ARIMA":
        return ARIMAModel()
    if name == "Prophet":
        return ProphetModel(freq=WEEKLY_FREQ)
    if name == "XGBoost":
        return XGBoostModel(freq=WEEKLY_FREQ)
    if name == "LSTM":
        return LSTMModel()
    raise ValueError(name)


def _evaluate_state(
    state: str, history: pd.DataFrame, val_weeks: int = VAL_WEEKS
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame, pd.DataFrame]:
    """Evaluate every candidate model on this state's validation window."""
    train, val = history.iloc[:-val_weeks].copy(), history.iloc[-val_weeks:].copy()
    y_val = val["y"].to_numpy(dtype=float)
    metrics: Dict[str, Dict[str, float]] = {}
    for name in _candidates_for(history):
        try:
            t0 = time.time()
            mdl = _build(name).fit(train)
            yhat = mdl.forecast(val_weeks)
            yhat = np.asarray(yhat, dtype=float)
            if len(yhat) != len(y_val):
                raise RuntimeError(f"Forecast length mismatch ({len(yhat)} vs {len(y_val)})")
            metrics[name] = {
                "rmse": rmse(y_val, yhat),
                "mae": mae(y_val, yhat),
                "fit_seconds": round(time.time() - t0, 2),
            }
        except Exception as exc:
            metrics[name] = {"rmse": float("inf"), "mae": float("inf"),
                             "error": f"{type(exc).__name__}: {exc}"}
    return metrics, train, val


def _pick_best(metrics: Dict[str, Dict[str, float]]) -> str:
    ranking = sorted(metrics.items(), key=lambda kv: (kv[1].get("rmse", float("inf")),
                                                       kv[1].get("mae", float("inf"))))
    for name, m in ranking:
        if np.isfinite(m.get("rmse", float("inf"))):
            return name
    return "Prophet"  # fallback if every candidate failed


def _final_forecast(
    state: str, best_name: str, history: pd.DataFrame, horizon: int = HORIZON,
    save_models: bool = True,
) -> pd.DataFrame:
    """Retrain `best_name` on full history and forecast `horizon` weeks ahead."""
    try:
        mdl = _build(best_name).fit(history)
    except Exception:
        # Unexpected failure on full-history retrain → fall back to Prophet.
        best_name = "Prophet"
        mdl = ProphetModel(freq=WEEKLY_FREQ).fit(history)
    yhat = np.asarray(mdl.forecast(horizon), dtype=float)
    last_ds = pd.to_datetime(history["ds"].iloc[-1])
    future_ds = pd.date_range(
        start=last_ds + pd.tseries.frequencies.to_offset(WEEKLY_FREQ),
        periods=horizon, freq=WEEKLY_FREQ,
    )
    if save_models:
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            slug = state.replace(" ", "_")
            mdl.save(MODELS_DIR / f"{slug}__{best_name}.joblib")
        except Exception:
            pass
    return pd.DataFrame({
        "state": state,
        "week_number": range(1, horizon + 1),
        "forecast_date": future_ds.strftime("%Y-%m-%d"),
        "predicted_sales": yhat,
        "model_used": best_name,
    })


def run_pipeline(states: list[str] | None = None, val_weeks: int = VAL_WEEKS,
                 horizon: int = HORIZON, save_models: bool = True,
                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run end-to-end pipeline. Returns (selection_df, forecasts_df)."""
    series = preprocess()
    if states:
        series = {s: series[s] for s in states if s in series}

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    selection_rows: List[Dict] = []
    forecast_frames: List[pd.DataFrame] = []
    for i, (state, hist) in enumerate(sorted(series.items()), start=1):
        if verbose:
            print(f"[{i:>2}/{len(series)}] {state} (rows={len(hist)})", flush=True)
        metrics, _, _ = _evaluate_state(state, hist, val_weeks=val_weeks)
        best = _pick_best(metrics)
        row = {
            "state": state,
            "best_model": best,
            "rmse": metrics[best].get("rmse"),
            "mae": metrics[best].get("mae"),
        }
        for name, m in metrics.items():
            row[f"{name}_rmse"] = m.get("rmse")
            row[f"{name}_mae"] = m.get("mae")
        selection_rows.append(row)
        try:
            fcst = _final_forecast(state, best, hist, horizon=horizon, save_models=save_models)
        except Exception as exc:
            if verbose:
                print(f"   final-forecast failed for {state}: {exc}", flush=True)
            continue
        forecast_frames.append(fcst)

    selection_df = pd.DataFrame(selection_rows)
    forecasts_df = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()

    selection_df.to_csv(OUTPUTS_DIR / "model_selection.csv", index=False)
    forecasts_df.to_csv(OUTPUTS_DIR / "forecasts.csv", index=False)
    if verbose:
        print(f"\nSaved {len(selection_df)} selection rows and "
              f"{len(forecasts_df)} forecast rows to {OUTPUTS_DIR}/")
    return selection_df, forecasts_df


def retrain_state(state: str, horizon: int = HORIZON,
                  save_models: bool = True) -> pd.DataFrame:
    """Retrain the pipeline for a single state and update outputs/."""
    sel_df, fcst_df = run_pipeline(states=[state], horizon=horizon,
                                   save_models=save_models, verbose=False)

    # Merge into existing outputs files (replace this state's rows).
    sel_path = OUTPUTS_DIR / "model_selection.csv"
    fcst_path = OUTPUTS_DIR / "forecasts.csv"
    if sel_path.exists():
        existing = pd.read_csv(sel_path)
        existing = existing[existing["state"] != state]
        sel_df = pd.concat([existing, sel_df], ignore_index=True).sort_values("state")
    sel_df.to_csv(sel_path, index=False)
    if fcst_path.exists():
        existing = pd.read_csv(fcst_path)
        existing = existing[existing["state"] != state]
        fcst_df = pd.concat([existing, fcst_df], ignore_index=True).sort_values(["state", "week_number"])
    fcst_df.to_csv(fcst_path, index=False)
    return fcst_df


if __name__ == "__main__":
    run_pipeline()
