# --- python -m scripts.03b_fig_softdtw ---
import sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from cmorph.distances import dtw, soft_dtw_distance

# charge quelques fenêtres
wins = pickle.load(open("artifacts/windows.pkl","rb"))
X = np.stack(wins)
N = len(X)

# prends 3 paires représentatives
pairs = [
    (0, 1, "similar"),
    (0, 50, "moderate"),
    (0, 200, "different"),
]

gammas = np.logspace(-2, 1, 20)  # de 0.01 à 10
band = 0.1

for (i,j,label) in pairs:
    d_hard = dtw(X[i], X[j], band=band)
    vals = [soft_dtw_distance(X[i], X[j], gamma=g, band=band) for g in gammas]

    plt.figure()
    plt.semilogx(gammas, vals, marker="o", label="soft-DTW")
    plt.axhline(d_hard, color="r", linestyle="--", label="DTW")
    plt.xlabel("gamma")
    plt.ylabel("distance")
    plt.title(f"Soft-DTW vs gamma ({label} pair)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"artifacts/softdtw_curve_{label}.png", dpi=150)
    print(f"📈 saved artifacts/softdtw_curve_{label}.png")
