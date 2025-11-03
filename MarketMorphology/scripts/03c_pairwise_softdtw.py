# scripts/03c_pairwise_softdtw.py
# Compute pairwise soft-DTW distance matrix

import sys
import time
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.distances import pairwise_soft_dtw, soft_dtw_distance, dtw


# --- Load windows ------------------------------------------------------------

wins = pickle.load(open("artifacts/windows.pkl", "rb"))  # list of 1D arrays
X = np.stack(wins)  # (N, L)
N, L = X.shape
print(f"Windows: N={N}, L={L}")

gamma = 0.5   # try 0.5, 1.0, 2.0...
band = 0.1    # 10% Sakoe–Chiba band


# --- Pairwise soft-DTW -------------------------------------------------------

t0 = time.time()
D_soft = pairwise_soft_dtw(X, gamma=gamma, band=band, batch=256, normalize=True)
t1 = time.time()
out = "artifacts/D_softdtw.npy"
np.save(out, D_soft)
print(f"Saved {out}  (shape={D_soft.shape}, t={t1 - t0:.1f}s)")


# --- Quick sanity checks -----------------------------------------------------

# (1) zero diagonal, symmetry
assert np.allclose(np.diag(D_soft), 0.0)
assert np.allclose(D_soft, D_soft.T)

# (2) soft-DTW <= DTW on a few random pairs
idx = np.random.default_rng(0).choice(N, size=min(10, N), replace=False)
for i in idx:
    j = (i + 1) % N
    d_soft = soft_dtw_distance(X[i], X[j], gamma=gamma, band=band)
    d_hard = dtw(X[i], X[j], band=band)
    if not (d_soft <= d_hard + 1e-9):
        print("Warning: soft-DTW > DTW on a sample pair.", d_soft, d_hard)

print("Sanity checks OK.")
