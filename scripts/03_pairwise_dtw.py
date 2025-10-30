# --- guard: permet de lancer "python scripts/03_pairwise_dtw.py"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------

import pickle
import numpy as np
from cmorph.distances import pairwise

# charge les fenêtres déjà générées à l'étape 01
wins = pickle.load(open("artifacts/windows.pkl", "rb"))

# pour développer plus vite, on peut sous-échantillonner :
N = min(1000, len(wins))    # augmente plus tard si tu veux
X = wins[:N]

print(f"Calcul de la matrice DTW pairwise sur N={N} fenêtres…")
D = pairwise(X, metric="dtw", band=0.15)

out_path = "artifacts/D_dtw.npy"
np.save(out_path, D)
print(f"✅ Matrice sauvegardée: {out_path} (shape={D.shape})")
