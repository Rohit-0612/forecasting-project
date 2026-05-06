"""FastAPI service for serving pre-computed weekly forecasts.

Reads `outputs/forecasts.csv` and `outputs/model_selection.csv` produced by
`src/model_selector.py`. The /retrain endpoint runs the per-state pipeline
in a background thread and updates the CSVs in place.
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_selector import retrain_state  # noqa: E402

OUTPUTS_DIR = ROOT / "outputs"
FORECASTS_PATH = OUTPUTS_DIR / "forecasts.csv"
SELECTION_PATH = OUTPUTS_DIR / "model_selection.csv"

app = FastAPI(title="State Sales Forecasting API", version="1.0.0")
_retrain_lock = threading.Lock()
_retrain_in_progress: set[str] = set()


def _load_forecasts() -> pd.DataFrame:
    if not FORECASTS_PATH.exists():
        raise HTTPException(503, "forecasts.csv not found — run the training pipeline first.")
    return pd.read_csv(FORECASTS_PATH)


def _load_selection() -> pd.DataFrame:
    if not SELECTION_PATH.exists():
        raise HTTPException(503, "model_selection.csv not found — run the training pipeline first.")
    return pd.read_csv(SELECTION_PATH)


def _safe_float(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def forecast(
    state: str = Query(..., description="State name, e.g. Alabama"),
    weeks: int = Query(8, ge=1, le=8, description="Number of weeks to return (1–8)"),
):
    df = _load_forecasts()
    sel = _load_selection()
    sub = df[df["state"].str.lower() == state.lower()].copy()
    if sub.empty:
        raise HTTPException(404, f"No forecast for state={state!r}")
    sub = sub.sort_values("week_number").head(weeks)
    sel_row = sel[sel["state"].str.lower() == state.lower()]
    model_used = sub["model_used"].iloc[0]
    return {
        "state": sub["state"].iloc[0],
        "model_used": model_used,
        "validation_rmse": _safe_float(sel_row["rmse"].iloc[0]) if not sel_row.empty else None,
        "validation_mae": _safe_float(sel_row["mae"].iloc[0]) if not sel_row.empty else None,
        "forecast": [
            {
                "week": int(r["week_number"]),
                "date": str(r["forecast_date"]),
                "predicted_sales": _safe_float(r["predicted_sales"]),
            }
            for _, r in sub.iterrows()
        ],
    }


@app.get("/models")
def models():
    sel = _load_selection()
    out = []
    for _, r in sel.iterrows():
        out.append({
            "state": r["state"],
            "best_model": r["best_model"],
            "rmse": _safe_float(r.get("rmse")),
            "mae": _safe_float(r.get("mae")),
        })
    return {"states": out}


def _retrain_worker(state: str) -> None:
    try:
        retrain_state(state)
    finally:
        with _retrain_lock:
            _retrain_in_progress.discard(state)


@app.post("/retrain")
def retrain(
    background_tasks: BackgroundTasks,
    state: str = Query(..., description="State to retrain"),
):
    sel = _load_selection()
    if state not in set(sel["state"].astype(str)):
        raise HTTPException(404, f"Unknown state {state!r}")
    with _retrain_lock:
        if state in _retrain_in_progress:
            raise HTTPException(409, f"Retrain already running for {state}")
        _retrain_in_progress.add(state)
    background_tasks.add_task(_retrain_worker, state)
    return {"status": "started", "state": state}


@app.get("/retrain/status")
def retrain_status():
    with _retrain_lock:
        return {"in_progress": sorted(_retrain_in_progress)}
