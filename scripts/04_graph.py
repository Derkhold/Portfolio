# guard
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
from cmorph.graph import mutual_knn, graph_stats

D = np.load("artifacts/D_dtw.npy")

print("Construction graphe k-NN mutuel…")
G = mutual_knn(D, k=15)

stats = graph_stats(G)
print("✅ Graphe construit :", stats)

import pickle
with open("artifacts/graph.pkl", "wb") as f:
    pickle.dump(G, f)

