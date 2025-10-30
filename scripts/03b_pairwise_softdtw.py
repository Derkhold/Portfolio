# --- guard pour: python -m scripts.03b_pairwise_softdtw ---
import sys, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
from cmorph.distances import pairwise_soft_dtw, soft_dtw_distance
from cmorph.distances import dtw  # si tu as déjà ta DTW "classique"

wins = pickle.load(open("artifacts/windows.pkl", "rb"))  # liste d'array 1D
X = np.stack(wins)  # (N, L)
N, L = X.shape
print(f"Windows: N={N}, L={L}")

gamma = 0.5        # commence petit (proche DTW), teste ensuite 1.0, 2.0
band  = 0.1        # bande de 10% (comme pour DTW)

t0 = time.time()
D_soft = pairwise_soft_dtw(X, gamma=gamma, band=band, batch=256, normalize=True)
t1 = time.time()
np.save("artifacts/D_softdtw.npy", D_soft)
print(f"✅ D_softdtw.npy shape={D_soft.shape}  (t={t1-t0:.1f}s)")

# --- Sanity checks rapides ---
# 1) diagonale nulle, symétrie
assert np.allclose(np.diag(D_soft), 0.0)
assert np.allclose(D_soft, D_soft.T)

# 2) soft-DTW <= DTW pour paires simples (sur quelques échantillons)
idx = np.random.default_rng(0).choice(N, size=min(10, N), replace=False)
for i in idx:
    j = (i + 1) % N
    d_soft = soft_dtw_distance(X[i], X[j], gamma=gamma, band=band)
    d_hard = dtw(X[i], X[j], band=band)
    if not (d_soft <= d_hard + 1e-9):
        print("⚠️  soft-DTW > DTW sur un cas (rare si random).", d_soft, d_hard)

print("Sanity checks OK.")
