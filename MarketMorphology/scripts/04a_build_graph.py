# scripts/04a_build_graph.py
# Build mutual k-NN graph from DTW distance matrix

import sys
from pathlib import Path
import numpy as np
import pickle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.graph import mutual_knn, graph_stats

# --- Load distance matrix ----------------------------------------------------

D = np.load("artifacts/D_dtw.npy")

print("Building mutual k-NN graph...")
G = mutual_knn(D, k=15)

# --- Summary stats -----------------------------------------------------------

stats = graph_stats(G)
print("Graph built:", stats)

# --- Save --------------------------------------------------------------------

with open("artifacts/graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("Graph saved to artifacts/graph.pkl")
