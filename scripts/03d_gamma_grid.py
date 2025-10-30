# --- python -m scripts.03d_gamma_grid ---
import sys, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from cmorph.distances import pairwise_soft_dtw

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

# ---------- utils ----------
def sanitize(D: np.ndarray) -> np.ndarray:
    """
    Rend la matrice de distances prête pour 'metric=\"precomputed\"':
    - symétrique
    - finie
    - >= 0
    - diagonale exactement 0 (après translation)
    """
    D = np.asarray(D, float)
    D = 0.5 * (D + D.T)

    # NaN/inf -> valeurs finies
    finite = np.isfinite(D)
    if not finite.all():
        repl = np.nanmax(D[finite]) if finite.any() else 0.0
        D = np.where(finite, D, repl)

    # translation si valeurs négatives, puis clamp
    mn = float(D.min())
    if mn < -1e-12:
        D = D - mn + 1e-12
    D = np.maximum(D, 0.0)

    # *** IMPORTANT: remettre la diagonale à 0 APRÈS translation
    np.fill_diagonal(D, 0.0)
    return D

def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    """S = exp(-D/sigma) avec sigma = médiane(D_ij>0)."""
    D = np.asarray(D, float)
    mask = (D > 0) & np.isfinite(D)
    sigma = np.median(D[mask]) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S

# ---------- data ----------
wins = pickle.load(open(ART / "windows.pkl", "rb"))
X = np.stack(wins)  # (N, L)

gammas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
K_RANGE = range(2, 7)

# ---------- grid ----------
rows = []
for g in gammas:
    t0 = time.time()
    D = pairwise_soft_dtw(X, gamma=g, band=0.1, batch=256, normalize=True)
    D = sanitize(D)
    S = distance_to_affinity(D)
    for k in K_RANGE:
        clus = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = clus.fit_predict(S)
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        rows.append({"gamma": g, "K": k, "silhouette": sil})
    print(f"gamma={g} → done in {time.time()-t0:.1f}s")

df = pd.DataFrame(rows)
df.to_csv(ART / "gamma_grid_results.csv", index=False)

# ---------- plots ----------
# Heatmap silhouette(gamma, K)
piv = df.pivot(index="gamma", columns="K", values="silhouette").reindex(index=gammas)
plt.figure(figsize=(6.5, 3.8))
plt.imshow(
    piv.values,
    aspect="auto",
    origin="lower",
    extent=[min(K_RANGE) - 0.5, max(K_RANGE) + 0.5, gammas[0] - 0.1, gammas[-1] + 0.1],
)
plt.colorbar(label="Silhouette")
plt.yticks(gammas)
plt.xticks(list(K_RANGE))
plt.title("Silhouette vs γ (soft-DTW) et K")
plt.xlabel("Amount of clusters K")
plt.ylabel("gamma")
plt.tight_layout()
plt.savefig(ART / "gamma_grid_heatmap.png", dpi=180)

# Courbe silhouette vs gamma au meilleur K moyen
bestK = int(df.groupby("K")["silhouette"].mean().idxmax())
sub = df[df["K"] == bestK].sort_values("gamma")
plt.figure(figsize=(5.2, 3.4))
plt.plot(sub["gamma"], sub["silhouette"], marker="o")
plt.xlabel("gamma")
plt.ylabel("Silhouette")
plt.title(f"Silhouette vs gamma (K={bestK})")
plt.tight_layout()
plt.savefig(ART / "gamma_sil_curve_Kstar.png", dpi=180)

print("📈 saved: gamma_grid_heatmap.png, gamma_sil_curve_Kstar.png, CSV")
