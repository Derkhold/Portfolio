from __future__ import annotations
import numpy as np

def realized_vol_from_prices(prices: np.ndarray) -> float:
    """Volatilité réalisée sqrt(sum r^2) sur la fenêtre."""
    p = np.asarray(prices, dtype=float)
    r = np.diff(np.log(p))
    return float(np.sqrt(np.sum(r*r)))

def max_drawdown(prices: np.ndarray) -> float:
    p = np.asarray(prices, dtype=float)
    cummax = np.maximum.accumulate(p)
    dd = (p - cummax) / cummax
    return float(dd.min())
