from __future__ import annotations

from typing import Literal, Optional

import numpy as np


# --- DTW (banded) ------------------------------------------------------------

def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.1) -> float:
    """
    DTW distance with a Sakoe–Chiba band (fraction of max length).
    a, b: 1D arrays (ideally same length). Returns sqrt(min cost).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    w = int(max(1, band * max(n, m)))
    INF = 1e18

    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        jmin = max(1, i - w)
        jmax = min(m, i + w)
        ai = a[i - 1]
        for j in range(jmin, jmax + 1):
            cost = (ai - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def pairwise(
    paths: list[np.ndarray],
    metric: Literal["dtw"] = "dtw",
    band: float = 0.15,
) -> np.ndarray:
    """
    Compute an NxN distance matrix (symmetric, diag=0).
    Currently supports 'dtw'.
    """
    N = len(paths)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if metric == "dtw":
                d = dtw_distance(paths[i], paths[j], band=band)
            else:
                raise ValueError("Unsupported metric")
            D[i, j] = D[j, i] = d
    return D


# --- Soft-DTW (Cuturi & Blondel, 2017) --------------------------------------

def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    # stable log-sum-exp soft minimum over three values
    m = min(a, b, c)
    s = np.exp(-(a - m) / gamma) + np.exp(-(b - m) / gamma) + np.exp(-(c - m) / gamma)
    return float(-gamma * (np.log(s) + (-m / gamma)))


def soft_dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    gamma: float = 1.0,
    band: Optional[float] = None,
    normalize: bool = True,
) -> float:
    """
    Soft-DTW for two 1D arrays of equal length L (gamma > 0).
    If band is set, apply a Sakoe–Chiba band. If normalize, return cost / L.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    Lx, Ly = x.shape[0], y.shape[0]
    assert Lx == Ly > 0, "soft_dtw_distance expects equal, positive lengths"

    L = Lx
    C = (x[:, None] - y[None, :]) ** 2

    INF = 1e20
    D = np.full((L + 1, L + 1), INF, dtype=float)
    D[0, 0] = 0.0

    if band is None:
        lower = np.zeros(L, dtype=int)
        upper = np.full(L, L - 1, dtype=int)
    else:
        w = int(np.ceil(float(band) * L))
        idx = np.arange(L)
        lower = np.maximum(0, idx - w)
        upper = np.minimum(L - 1, idx + w)

    for i in range(1, L + 1):
        j0, j1 = lower[i - 1] + 1, upper[i - 1] + 1
        for j in range(j0, j1 + 1):
            c = C[i - 1, j - 1]
            d1 = D[i - 1, j]
            d2 = D[i, j - 1]
            d3 = D[i - 1, j - 1]
            D[i, j] = c + _softmin3(d1, d2, d3, gamma)

    val = D[L, L]
    return float(val / L) if normalize else float(val)


def pairwise_soft_dtw(
    paths: np.ndarray,  # shape: (N, L)
    gamma: float = 1.0,
    band: Optional[float] = 0.1,
    batch: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """
    Pairwise soft-DTW over (N, L) paths. Computes upper triangle then symmetrizes.
    """
    X = np.asarray(paths, dtype=float)
    N, _ = X.shape
    D = np.zeros((N, N), dtype=float)

    for i in range(N):
        xa = X[i]
        j_start = i + 1
        j_end = min(N, i + 1 + batch)
        while j_start < N:
            for j in range(j_start, j_end):
                D[i, j] = soft_dtw_distance(
                    xa, X[j], gamma=gamma, band=band, normalize=normalize
                )
            j_start = j_end
            j_end = min(N, j_end + batch)
        if (i % 50) == 0:
            pass

    D = D + D.T
    np.fill_diagonal(D, 0.0)
    return D


# --- Baselines ---------------------------------------------------------------

def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    corr = float(np.dot(x, y) / len(x))
    corr = float(np.clip(corr, -1.0, 1.0))
    return float(1.0 - corr)


def ncc_max_distance(a: np.ndarray, b: np.ndarray, max_lag: int | None = None) -> float:
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    L = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    if max_lag is None:
        max_lag = L - 1
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xv = x[lag:]
            yv = y[: L - lag]
        else:
            xv = x[: L + lag]
            yv = y[-lag:]
        if len(xv) < 3:
            continue
        c = float(np.dot(xv, yv) / len(xv))
        if c > best:
            best = c
    best = float(np.clip(best, -1.0, 1.0))
    return float(1.0 - best)


def frechet_1d_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Discrete Fréchet distance for 1D curves (Eiter & Mannila).
    """
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    n, m = len(x), len(y)
    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = abs(x[i] - y[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    return float(_c(n - 1, m - 1))


# --- Compatibility alias -----------------------------------------------------

def dtw(a: np.ndarray, b: np.ndarray, band: float = 0.1) -> float:
    return dtw_distance(a, b, band=band)
