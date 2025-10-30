from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    g = df.set_index('timestamp').sort_index()
    ohlcv = g[['open','high','low','close','volume']].resample('5min').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    })
    ohlcv = ohlcv.dropna(subset=['close']).reset_index()
    return ohlcv

def make_windows(series: pd.Series, L: int = 12, S: int = 1) -> List[np.ndarray]:
    x = series.to_numpy(dtype=float)
    wins = []
    for start in range(0, len(x) - L + 1, S):
        w = x[start:start+L]
        if np.isnan(w).any():
            continue
        wins.append(w)
    return wins

def normalize_window_returns_z(win_prices: np.ndarray) -> np.ndarray:
    """
    1) log-returns, 2) z-score dans la fenêtre, 3) cumul (courbe de forme) démarrant à 0.
    """
    lp = np.log(np.asarray(win_prices, dtype=float))
    r = np.diff(lp, prepend=lp[0])
    mu = r.mean()
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    rz = (r - mu) / sd
    curve = np.cumsum(rz)
    curve -= curve[0]
    return curve
