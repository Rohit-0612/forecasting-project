"""Preprocessing for the weekly state-level beverages sales dataset.

Loads the Excel file, parses dates, resamples each state's series to a
strict weekly frequency, and fills missing weeks. Output is a dict of
{state: DataFrame[ds, y]} ready to feed Prophet (or any other model).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Forecasting_Case-_Study.xlsx"
WEEKLY_FREQ = "W-SAT"  # week ending Saturday


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Excel file and return a tidy DataFrame.

    Drops the constant Category column (only "Beverages").
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Total", "State"]).copy()
    df = df[["State", "Date", "Total"]]
    df.columns = ["state", "ds", "y"]
    df["y"] = df["y"].astype(float)
    return df


def _resample_state(df: pd.DataFrame, freq: str = WEEKLY_FREQ, fill: str = "interpolate") -> pd.DataFrame:
    """Resample one state's series to strict weekly frequency."""
    s = df.set_index("ds")["y"].sort_index()
    # If multiple raw rows fall in the same week, sum them (raw data is already
    # sales totals so summing is the right reduction for irregular reporting).
    s = s.resample(freq).sum(min_count=1)
    if fill == "ffill":
        s = s.ffill()
    elif fill == "interpolate":
        s = s.interpolate(method="time", limit_direction="both")
    s = s.ffill().bfill()  # safety net for endpoints
    out = s.reset_index()
    out.columns = ["ds", "y"]
    return out


def preprocess(
    path: str | Path = DEFAULT_DATA_PATH,
    freq: str = WEEKLY_FREQ,
    fill: str = "interpolate",
) -> Dict[str, pd.DataFrame]:
    """Return {state: DataFrame[ds, y]} resampled to weekly frequency."""
    raw = load_raw(path)
    out: Dict[str, pd.DataFrame] = {}
    for state, sdf in raw.groupby("state"):
        out[state] = _resample_state(sdf, freq=freq, fill=fill)
    return out


if __name__ == "__main__":
    series = preprocess()
    print(f"States: {len(series)}")
    sample_state = next(iter(series))
    print(f"Sample state: {sample_state}")
    print(series[sample_state].head())
    print(series[sample_state].tail())
    print(f"Rows per state (min/median/max): "
          f"{min(len(v) for v in series.values())}/"
          f"{sorted(len(v) for v in series.values())[len(series)//2]}/"
          f"{max(len(v) for v in series.values())}")
