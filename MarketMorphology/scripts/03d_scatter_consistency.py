# scripts/03d_scatter_consistency.py
# Scatter consistency checks between distance matrices (Spearman ρ)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# --- Pairs to compare --------------------------------------------------------

pairs = [
    ("D_dtw.npy",     "D_softdtw.npy", "DTW vs soft-DTW"),
    ("D_dtw.npy",     "D_corr.npy",    "DTW vs Corr"),
    ("D_dtw.npy",     "D_ncc.npy",     "DTW vs NCC"),
    ("D_dtw.npy",     "D_frechet.npy", "DTW vs Frechet"),
]


# --- Helpers -----------------------------------------------------------------

def load_mat(name: str) -> np.ndarray:
    """Load matrix, enforce symmetry and zero diagonal."""
    D = np.load(f"artifacts/{name}")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


# --- Plots -------------------------------------------------------------------

for a, b, title in pairs:
    try:
        A, B = load_mat(a), load_mat(b)
    except FileNotFoundError:
        print(f"Skip: {title} (missing {a} or {b})")
        continue

    iu = np.triu_indices_from(A, k=1)
    x, y = A[iu], B[iu]

    # Shift if negatives exist (for log-like scales or robustness)
    if np.min(x) < 0:
        x = x - np.min(x) + 1e-12
    if np.min(y) < 0:
        y = y - np.min(y) + 1e-12

    rho, p = spearmanr(x, y)

    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, s=6)
    plt.xlabel(a)
    plt.ylabel(b)
    plt.title(f"{title}\nSpearman ρ={rho:.3f} (p={p:.1e})")
    plt.tight_layout()

    out = f"artifacts/scatter_{a[:-4]}_vs_{b[:-4]}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
