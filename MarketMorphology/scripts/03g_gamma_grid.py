# scripts/03g_gamma_grid.py
# Grid search over soft-DTW gamma and K (spectral clustering + silhouette)

import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from cmorph.distances import pairwise_soft_dtw

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


# --- Utils -------------------------------------------------------------------

def sanitize(D: np.ndarray) -> np.ndarray:
    """Make distance matrix symmetric, finite, >=0, with zero diagonal."""
    D = np.asarray(D, float)
    D = 0.5 * (D + D.T)

    finite = np.isfinite(D)
    if not finite.all():
        repl = np.nanmax(D[finite]) if finite.any() else 0.0
        D = np.where(finite, D, repl)

    mn = float(D.min())
    if mn < -1e-12:
        D = D - mn + 1e-12
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    """S = exp(-D / sigma) with sigma = median(D_ij > 0)."""
    D = np.asarray(D, float)
    mask = (D > 0) & np.isfinite(D)
    sigma = float(np.median(D[mask])) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S


# --- Data --------------------------------------------------------------------

wins = pickle.load(open(ART / "windows.pkl", "rb"))
X = np.stack(wins)  # (N, L)

gammas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
K_RANGE = range(2, 7)


# --- Grid --------------------------------------------------------------------

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
    print(f"gamma={g} done in {time.time() - t0:.1f}s")

df = pd.DataFrame(rows)
(df).to_csv(ART / "gamma_grid_results.csv", index=False)


# --- Plots -------------------------------------------------------------------

# Heatmap: silhouette(gamma, K)
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
plt.title("Silhouette vs γ (soft-DTW) and K")
plt.xlabel("Number of clusters K")
plt.ylabel("gamma")
plt.tight_layout()
plt.savefig(ART / "gamma_grid_heatmap.png", dpi=180)
plt.close()

# Curve: silhouette vs gamma at best average K
bestK = int(df.groupby("K")["silhouette"].mean().idxmax())
sub = df[df["K"] == bestK].sort_values("gamma")
plt.figure(figsize=(5.2, 3.4))
plt.plot(sub["gamma"], sub["silhouette"], marker="o")
plt.xlabel("gamma")
plt.ylabel("Silhouette")
plt.title(f"Silhouette vs gamma (K={bestK})")
plt.tight_layout()
plt.savefig(ART / "gamma_sil_curve_Kstar.png", dpi=180)
plt.close()

print("Saved: gamma_grid_heatmap.png, gamma_sil_curve_Kstar.png, gamma_grid_results.csv")
