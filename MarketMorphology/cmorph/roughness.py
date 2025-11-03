from __future__ import annotations

from typing import Tuple

import numpy as np


# --- Detrended Fluctuation Analysis (DFA) -----------------------------------

def dfa_alpha(
    path: np.ndarray,
    orders: Tuple[int, ...] = (1,),
    scales: np.ndarray | None = None,
) -> float:
    """
    Compute DFA scaling exponent α for a 1D series.
    """
    x = np.asarray(path, float)
    x -= x.mean()
    y = np.cumsum(x)

    L = len(x)
    if scales is None:
        scales = np.unique(
            np.round(
                np.linspace(4, max(4, L // 2), num=min(8, max(2, L // 2)))
            ).astype(int)
        )
        scales = scales[scales >= 4]

    Fs, ns = [], []
    for n in scales:
        segments = len(y) // n
        if segments < 2:
            continue
        Z = y[: segments * n].reshape(segments, n)
        t = np.arange(n)
        F2 = []
        for seg in Z:
            a, b = np.polyfit(t, seg, deg=1)
            trend = a * t + b
            F2.append(np.mean((seg - trend) ** 2))
        Fs.append(np.sqrt(np.mean(F2)))
        ns.append(n)

    if len(Fs) < 2:
        return 0.5
    logn = np.log(ns)
    logF = np.log(Fs)
    alpha = np.polyfit(logn, logF, 1)[0]
    return float(alpha)


# --- Higuchi fractal dimension ----------------------------------------------

def higuchi_fd(path: np.ndarray, kmax: int = 8) -> float:
    """
    Compute Higuchi's fractal dimension for a 1D series.
    """
    x = np.asarray(path, float)
    N = len(x)
    ks = range(2, min(kmax, max(3, N // 3)) + 1)
    if N < 4:
        return 1.5

    Lk = []
    for k in ks:
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            xm = x[idx]
            Lm.append((N - 1) / (len(idx) * k) * np.sum(np.abs(np.diff(xm))))
        if Lm:
            Lk.append(np.mean(Lm))

    if len(Lk) < 2:
        return 1.5

    slope = -np.polyfit(np.log(list(ks)[: len(Lk)]), np.log(Lk), 1)[0]
    return float(np.clip(slope, 1.0, 2.0))
