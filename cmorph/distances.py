from __future__ import annotations
import numpy as np
from typing import Literal
from typing import Optional

def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.1) -> float:
    """
    Distance DTW avec bande de Sakoe–Chiba (fraction de la longueur max).
    - a, b : courbes (np.ndarray 1D) de même longueur L (recommandé).
    - band : largeur de bande (0.10 = 10% de L). Évite les alignements pathologiques.
    Retourne: sqrt(cost minimal).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    w = int(max(1, band * max(n, m)))  # demi-largeur de bande
    INF = 1e18

    # matrice cumulée (n+1 x m+1) initialisée à +inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        jmin = max(1, i - w)
        jmax = min(m, i + w)
        ai = a[i - 1]
        for j in range(jmin, jmax + 1):
            cost = (ai - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j],    # insertion
                                 D[i, j - 1],    # deletion
                                 D[i - 1, j - 1])# match
    return float(np.sqrt(D[n, m]))


def pairwise(
    paths: list[np.ndarray],
    metric: Literal["dtw"] = "dtw",
    band: float = 0.15,
) -> np.ndarray:
    """
    Calcule la matrice des distances NxN (symétrique, diag=0).
    - paths : liste de courbes (ex: celles de windows.pkl)
    - metric : pour l'instant "dtw" (banded)
    - band : largeur de bande pour DTW (par défaut 15%)
    """
    N = len(paths)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        # diagonale nulle
        for j in range(i + 1, N):
            if metric == "dtw":
                d = dtw_distance(paths[i], paths[j], band=band)
            else:
                raise ValueError("Metric non supportée")
            D[i, j] = D[j, i] = d
    return D

# ---------- SOFT-DTW (Cuturi & Blondel, 2017) ----------

def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    # softmin_gamma(x) = -γ log( exp(-x1/γ) + exp(-x2/γ) + exp(-x3/γ) )
    # fonctionne pour γ>0 ; quand γ->0, tend vers min(x).
    m = min(a, b, c)
    # log-sum-exp stable numériquement
    s = np.exp(-(a - m)/gamma) + np.exp(-(b - m)/gamma) + np.exp(-(c - m)/gamma)
    return float(-gamma * (np.log(s) + ( -m / gamma )))

def soft_dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    gamma: float = 1.0,
    band: Optional[float] = None,
    normalize: bool = True,
) -> float:
    """
    Soft-DTW entre deux courbes 1D (séquences de même longueur L).
    - a, b : np.ndarray de forme (L,)
    - gamma : lissage (γ>0). Plus γ est grand, plus la valeur est 'soft' (plus petite).
    - band : fraction de bande de Sakoe–Chiba (par ex. 0.1 => 10% de L). None = pas de bande.
    - normalize : renvoie la somme des coûts / L (utile pour comparer entre L).
    Retourne un coût >= 0. soft-DTW <= DTW, et soft-DTW -> DTW quand γ -> 0+.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    Lx, Ly = x.shape[0], y.shape[0]
    assert Lx == Ly > 0, "soft_dtw_distance suppose des longueurs égales >0"

    L = Lx
    # matrice locale des coûts (c_ij = (x_i - y_j)^2)
    # (on reste en simple précision pour perf si besoin, mais float64 ok)
    C = (x[:, None] - y[None, :]) ** 2

    # DP (L+1)x(L+1) initialisée à +inf, avec D[0,0]=0
    INF = 1e20
    D = np.full((L + 1, L + 1), INF, dtype=float)
    D[0, 0] = 0.0

    # bande de Sakoe–Chiba
    if band is None:
        lower = np.zeros(L, dtype=int)
        upper = np.full(L, L - 1, dtype=int)
    else:
        w = int(np.ceil(float(band) * L))
        idx = np.arange(L)
        lower = np.maximum(0, idx - w)
        upper = np.minimum(L - 1, idx + w)

    for i in range(1, L + 1):
        j0, j1 = lower[i-1] + 1, upper[i-1] + 1  # indices en DP (décalés de 1)
        # on laisse D[i, 0:j0] et D[i, j1+1:] à +inf (en dehors de la bande)
        for j in range(j0, j1 + 1):
            c = C[i - 1, j - 1]
            d1 = D[i - 1, j]      # insertion
            d2 = D[i, j - 1]      # suppression
            d3 = D[i - 1, j - 1]  # match
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
    Matrice pairwise soft-DTW sur un ensemble de chemins (N, L).
    Calcule uniquement la partie triangulaire puis symétrise.
    """
    X = np.asarray(paths, dtype=float)
    N, L = X.shape
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        xa = X[i]
        j_start = i + 1
        j_end = min(N, i + 1 + batch)
        while j_start < N:
            for j in range(j_start, j_end):
                D[i, j] = soft_dtw_distance(xa, X[j], gamma=gamma, band=band, normalize=normalize)
            j_start = j_end
            j_end = min(N, j_end + batch)
        if (i % 50) == 0:
            pass  # place pour un éventuel print de progression
    # symétrie + diag = 0
    D = D + D.T
    np.fill_diagonal(D, 0.0)
    return D

# ---------- BASELINES RAPIDES ----------

def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    corr = float(np.dot(x, y) / len(x))
    corr = float(np.clip(corr, -1.0, 1.0))     # <-- IMPORTANT
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
            xv = x[lag:]; yv = y[:L - lag]
        else:
            xv = x[:L + lag]; yv = y[-lag:]
        if len(xv) < 3:
            continue
        c = float(np.dot(xv, yv) / len(xv))
        if c > best:
            best = c
    best = float(np.clip(best, -1.0, 1.0))      # <-- IMPORTANT
    return float(1.0 - best)


def frechet_1d_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance de Fréchet discrète pour courbes 1D (Eiter & Mannila).
    Interprétation: 'lien' le plus court pour parcourir les deux courbes dans l'ordre.
    """
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    n, m = len(x), len(y)
    ca = np.full((n, m), -1.0)

    def _c(i, j):
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

# Alias pour compatibilité avec les scripts
def dtw(a: np.ndarray, b: np.ndarray, band: float = 0.1) -> float:
    return dtw_distance(a, b, band=band)
