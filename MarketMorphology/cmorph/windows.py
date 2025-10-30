from __future__ import annotations

import numpy as np
import pandas as pd


# --- Resampling --------------------------------------------------------------

def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to 5-minute OHLCV. Volume is summed if present.
    """
    x = df.set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in x.columns:
        agg["volume"] = "sum"
        cols = ["open", "high", "low", "close", "volume"]
    else:
        cols = ["open", "high", "low", "close"]

    ohlc = x[cols].resample("5min").agg(agg)
    ohlc = ohlc.dropna(subset=["close"]).reset_index()
    return ohlc


# --- Windowing ---------------------------------------------------------------

def make_windows(series: pd.Series, L: int = 12, S: int = 1) -> list[np.ndarray]:
    """
    Fixed-length sliding windows of length L and stride S (skip windows with NaN).
    """
    x = series.to_numpy(dtype=float)
    wins: list[np.ndarray] = []
    for start in range(0, len(x) - L + 1, S):
        w = x[start : start + L]
        if np.isnan(w).any():
            continue
        wins.append(w)
    return wins


# --- In-window normalization (shape curve) -----------------------------------

def normalize_window_returns_z(win_prices: np.ndarray) -> np.ndarray:
    """
    (1) log-returns, (2) z-score within window, (3) cumulative curve starting at 0.
    """
    lp = np.log(np.asarray(win_prices, dtype=float))
    r = np.diff(lp, prepend=lp[0])
    mu = r.mean()
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0.0:
        sd = 1.0
    rz = (r - mu) / sd
    curve = np.cumsum(rz)
    curve -= curve[0]
    return curve
