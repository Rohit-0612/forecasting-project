# State Sales Forecasting

End-to-end weekly time-series forecasting system for **43 US states**. The
pipeline trains four families of models per state (ARIMA, Prophet,
XGBoost, LSTM), automatically selects the best one on a strictly
out-of-sample validation window, and serves the resulting 8-week
forecasts through a FastAPI service.

---

## Project Overview & Objective

Given irregularly-reported weekly beverage sales totals for 43 US states
between Jan 2019 and Dec 2023, **forecast the next 8 weeks of sales for
each state**.

To make the forecasts honest:

- Every series is resampled to a strict weekly frequency (week ending
  Saturday) before training.
- The validation window is the **last 8 weeks** of the historical data —
  no random splits, no lookahead.
- For each state, all four models are evaluated on that window; the one
  with the lowest RMSE is retrained on the full history and used to
  produce the 8-week future forecast.
- The API serves *pre-computed* forecasts; it never retrains on the
  request path.

## Dataset

- **Source:** `data/Forecasting_Case-_Study.xlsx`
- **Columns:** `State`, `Date`, `Total` (USD), `Category` (constant
  `"Beverages"` — dropped during preprocessing)
- **States:** 43 US states
- **Span:** Jan 2019 – Dec 2023
- **Frequency:** irregular in raw form (mostly weekly with gaps and
  duplicates) → resampled to **W-SAT** (week ending Saturday) with
  time-based linear interpolation for missing weeks.
- **Rows per state after resampling:** 257

## Models Implemented

| Model | Library | Notes |
|---|---|---|
| **ARIMA / SARIMA** | `pmdarima.auto_arima` | Auto-selects `(p,d,q)(P,D,Q,s)`; `s=52` for weekly seasonality. SARIMAX fallback if `auto_arima` fails. |
| **Prophet** | `prophet` | `weekly_seasonality=True`, `yearly_seasonality=True`, US country holidays. |
| **XGBoost** | `xgboost.XGBRegressor` | Lag (`t-1, t-4, t-8`), 4/8-week rolling means, 4-week rolling std, calendar features. **Recursive** multi-step forecasting. |
| **LSTM** | PyTorch | Sequence length 12, two stacked LSTM layers + dropout, MinMaxScaler, 50 epochs, recursive multi-step. Falls back gracefully on series with fewer than 24 weekly rows. |

> **Note on the LSTM backend.** This project specifies TensorFlow/Keras
> in `requirements.txt`, but the LSTM module uses PyTorch under the
> hood: the local TensorFlow build hit a `pthread_mutex_init` failure on
> macOS, so the LSTM was reimplemented in PyTorch with the same
> architecture, hyperparameters, and recursive-forecast strategy. The
> outward-facing `fit / forecast / save / load` API is identical to the
> other models, so the selector and the API treat it like any other
> candidate.

## Project Folder Structure

```
forecasting_project/
├── data/
│   └── Forecasting_Case-_Study.xlsx     # raw weekly sales
├── src/
│   ├── preprocess.py                    # Excel → resampled weekly per-state series
│   ├── features.py                      # lag, rolling, calendar features + time split
│   ├── evaluate.py                      # RMSE / MAE / MAPE
│   ├── model_selector.py                # train, evaluate, pick best, generate forecasts
│   └── models/
│       ├── arima_model.py               # auto_arima + SARIMAX fallback
│       ├── prophet_model.py             # Prophet with US holidays
│       ├── xgboost_model.py             # XGBRegressor + iterative forecast
│       └── lstm_model.py                # 2-layer LSTM with dropout (PyTorch)
├── api/
│   └── main.py                          # FastAPI service
├── outputs/
│   ├── model_selection.csv              # per-state best model + RMSE/MAE
│   └── forecasts.csv                    # 8-week forecasts per state
├── models_saved/                        # serialized winning models per state
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-username>/forecasting-project.git
cd forecasting-project

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> Prophet requires a working C++ toolchain (the Stan backend builds via
> `cmdstanpy`). On macOS, `xcode-select --install` is usually enough.

## Run the Training Pipeline

From the project root:

```bash
python -m src.model_selector
```

This runs the full pipeline for all 43 states:

1. Load and resample the Excel data to W-SAT weekly frequency.
2. Train each of the 4 candidate models on `history[:-8]`.
3. Forecast 8 weeks and score on `history[-8:]` (RMSE primary, MAE
   secondary).
4. Pick the lowest-RMSE model per state, retrain on the full history,
   and forecast 8 weeks ahead of the last observation.
5. Persist:
   - `outputs/model_selection.csv` — `state, best_model, rmse, mae`
     plus `<Model>_rmse` / `<Model>_mae` for every candidate.
   - `outputs/forecasts.csv` — `state, week_number, forecast_date,
     predicted_sales, model_used`.
   - `models_saved/<State>__<Model>.joblib` (and `.pt` for LSTM).

The pipeline runs all four models on all 43 states. Expect roughly
**10–15 minutes** on a CPU-only machine (LSTM is the bottleneck).

> The pipeline pins thread counts (`OMP_NUM_THREADS=1`, etc.) at
> import time to avoid pthread/OpenMP races on macOS.

## API Usage

Start the server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `GET` | `/forecast?state=<name>&weeks=<1–8>` | Returns the 8-week forecast for a state, including which model produced it and its validation RMSE/MAE. |
| `GET` | `/models` | Returns the best model + metrics for every state. |
| `POST` | `/retrain?state=<name>` | Triggers an asynchronous per-state retrain in a background thread; updates `outputs/` in place when finished. |
| `GET` | `/retrain/status` | Lists states currently being retrained. |

### Example calls

```bash
# 1. Health check
curl -s http://localhost:8000/health
# → {"status":"ok"}

# 2. 8-week forecast for one state
curl -s 'http://localhost:8000/forecast?state=Alabama&weeks=8' | python -m json.tool

# 3. First few weeks only
curl -s 'http://localhost:8000/forecast?state=California&weeks=3' | python -m json.tool

# 4. Best model + metrics for every state
curl -s http://localhost:8000/models | python -m json.tool

# 5. Trigger an async retrain for a state
curl -s -X POST 'http://localhost:8000/retrain?state=Alabama'

# 6. Check retrain progress
curl -s http://localhost:8000/retrain/status
```

### Example response (`/forecast?state=Alabama&weeks=8`)

```json
{
  "state": "Alabama",
  "model_used": "LSTM",
  "validation_rmse": 16928884.25,
  "validation_mae": 15112636.48,
  "forecast": [
    {"week": 1, "date": "2023-12-16", "predicted_sales": 213083259.93},
    {"week": 2, "date": "2023-12-23", "predicted_sales": 216024255.84},
    {"week": 3, "date": "2023-12-30", "predicted_sales": 218586504.88},
    {"week": 4, "date": "2024-01-06", "predicted_sales": 220970538.78},
    {"week": 5, "date": "2024-01-13", "predicted_sales": 223256113.61},
    {"week": 6, "date": "2024-01-20", "predicted_sales": 225476038.98},
    {"week": 7, "date": "2024-01-27", "predicted_sales": 227640445.05},
    {"week": 8, "date": "2024-02-03", "predicted_sales": 229746111.34}
  ]
}
```

## Results Summary

After running on all 43 states with the strict last-8-weeks validation
window:

- **States forecasted:** 43
- **Forecast rows produced:** 344 (43 × 8 weeks)
- **Forecast horizon:** 2023-12-16 → 2024-02-03 (8 weeks ahead of the
  last observation 2023-12-09)
- **Best-model distribution:**
  - LSTM — 40 states
  - ARIMA — 2 states (Nebraska, South Dakota)
  - XGBoost — 1 state (Arkansas)

LSTM dominates because the per-state series have strong recent trend
and short-horizon autoregressive structure, which the recurrent model
captures more tightly than Prophet's additive components or ARIMA's
linear formulation on this data. Per-state metrics are in
`outputs/model_selection.csv`.

## Design Notes

- **No data leakage.** Every model is trained on `history[:-8]` and
  evaluated on `history[-8:]`. No random shuffling. No CV folds that
  cross train/test boundaries.
- **Sparse states.** If a state has fewer than 24 weekly rows after
  resampling, LSTM is skipped automatically. If every candidate fails
  for a state, the pipeline falls back to Prophet.
- **Recursive multi-step.** XGBoost and LSTM forecast iteratively: each
  predicted value is fed into the lag/rolling buffer (XGBoost) or LSTM
  input window for the next step.
- **API never retrains on the request path.** It serves the
  pre-computed `outputs/forecasts.csv`. `/retrain` runs in a background
  thread so the request returns immediately.
- **Saved models.** Every winning model is serialized to
  `models_saved/`. Sklearn-compatible models use `joblib`; LSTM uses
  PyTorch's `state_dict` plus a `joblib` sidecar for the scaler and
  metadata.
