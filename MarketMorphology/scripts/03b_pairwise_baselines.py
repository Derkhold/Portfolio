# scripts/03b_pairwise_baselines.py
# Compute pairwise distance matrices for baseline metrics

import sys
import time
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.distances import (
    correlation_distance,
    ncc_max_distance,
    frechet_1d_distance,
)


# --- Load windows ------------------------------------------------------------

wins = pickle.load(open("artifacts/windows.pkl", "rb"))
X = np.stack(wins)
N = X.shape[0]


# --- Pairwise computation ----------------------------------------------------

def pairwise(metric_fn, name: str):
    D = np.zeros((N, N), float)
    t0 = time.time()
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = metric_fn(X[i], X[j])
    D = D + D.T
    np.fill_diagonal(D, 0.0)

    out = f"artifacts/D_{name}.npy"
    np.save(out, D)
    print(f"Saved {out}  (t={time.time() - t0:.1f}s, shape={D.shape})")


# --- Run ---------------------------------------------------------------------

pairwise(correlation_distance, "corr")
pairwise(lambda a, b: ncc_max_distance(a, b, max_lag=4), "ncc")
pairwise(frechet_1d_distance, "frechet")
