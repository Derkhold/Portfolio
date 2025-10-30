from __future__ import annotations
import numpy as np
from typing import Dict

def saw_score(path: np.ndarray, bins: int = 20, eps: float | None = None) -> Dict[str, float]:
    """
    Score d'auto-évitation pour une courbe 1D:
    - Discrétise l'axe vertical en 'bins'
    - Compte les retouches (revisites) de niveaux déjà vus dans un corridor ±eps
    Retourne: S in [0,1], rho (ratio de nouveaux niveaux), I (nb de revisites)
    """
    x = np.asarray(path, dtype=float)
    if len(x) < 3:
        return {"S": 0.5, "rho": 0.5, "I": 0.0}

    if eps is None:
        eps = 0.5 * np.std(x) if np.std(x) > 0 else 1.0

    lo, hi = float(x.min()), float(x.max()) + 1e-12
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    visited = set()
    I = 0
    new_levels = 0

    for v in x:
        # trouve le(s) bin(s) dont le centre est dans le corridor de v
        close = np.where(np.abs(centers - v) <= eps)[0]
        corridor = set(int(i) for i in close.tolist())
        if visited & corridor:
            I += 1
        else:
            new_levels += 1
        visited |= corridor

    Imax = max(1, len(x))
    S = 1.0 - (I / Imax)
    rho = new_levels / len(x)
    return {"S": float(S), "rho": float(rho), "I": float(I)}

