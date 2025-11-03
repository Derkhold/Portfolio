from __future__ import annotations

import numpy as np


# --- Realized volatility -----------------------------------------------------

def realized_vol_from_prices(prices: np.ndarray) -> float:
    """
    Realized volatility (sqrt of sum of squared log-returns) over a window.
    """
    p = np.asarray(prices, dtype=float)
    r = np.diff(np.log(p))
    return float(np.sqrt(np.sum(r * r)))


# --- Maximum drawdown --------------------------------------------------------

def max_drawdown(prices: np.ndarray) -> float:
    """
    Maximum drawdown over a price window.
    """
    p = np.asarray(prices, dtype=float)
    cummax = np.maximum.accumulate(p)
    dd = (p - cummax) / cummax
    return float(dd.min())
