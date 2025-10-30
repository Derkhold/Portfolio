# scripts/03_pairwise_dtw.py
# Compute pairwise DTW distance matrix between normalized windows

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pickle
import numpy as np
from cmorph.distances import pairwise


# --- Load windows ------------------------------------------------------------

wins = pickle.load(open("artifacts/windows.pkl", "rb"))

# Optional: subsample for faster development
N = min(1000, len(wins))  # increase later if needed
X = wins[:N]

print(f"Computing pairwise DTW matrix on N={N} windows...")
D = pairwise(X, metric="dtw", band=0.15)

out_path = "artifacts/D_dtw.npy"
np.save(out_path, D)
print(f"Saved DTW matrix: {out_path} (shape={D.shape})")
