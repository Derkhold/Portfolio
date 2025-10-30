# --- python -m scripts.03c_compare_metrics ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


# --------------------------- utils ---------------------------

def sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Nettoie une matrice de 'distance' pré-calculée pour un usage silhouette/precomputed.
    - Force la symétrie et met la diagonale à 0.
    - Si des valeurs négatives existent (soft-DTW ou corr/NCC), translate pour rendre D >= 0.
    - Élimine NaN/inf résiduels si présents (remplacés par max fini).
    Retourne une copie nettoyée (float64).
    """
    D = np.asarray(D, dtype=float)
    # symétrie & diag
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    # NaN / inf -> valeurs finies
    if not np.isfinite(D).all():
        finite = D[np.isfinite(D)]
        repl = np.nanmax(finite) if finite.size else 0.0
        D = np.where(np.isfinite(D), D, repl)
        np.fill_diagonal(D, 0.0)

    # clamp très léger sous 0 (erreurs numériques), sinon translate
    mn = float(np.min(D))
    if mn < -1e-12:
        D = D - mn + 1e-12
        np.fill_diagonal(D, 0.0)
    else:
        D = np.maximum(D, 0.0)

    # sécurités finales
    assert np.allclose(D, D.T), "Distance matrix must be symmetric"
    assert np.all(D.diagonal() == 0.0), "Diagonal must be zero"
    assert np.min(D) >= 0.0, "Distances must be non-negative for silhouette"
    return D


def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    """
    Transforme une distance en affinité positive pour SpectralClustering.
    On utilise S_ij = exp(-D_ij / sigma), avec sigma = médiane(D_ij > 0).
    """
    D = np.asarray(D, dtype=float)
    mask = (D > 0) & np.isfinite(D)
    sigma = np.median(D[mask]) if np.any(mask) else 1.0
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S


# --------------------------- main ----------------------------

candidates = {
    "dtw":      "artifacts/D_dtw.npy",
    "softdtw":  "artifacts/D_softdtw.npy",
    "corr":     "artifacts/D_corr.npy",
    "ncc":      "artifacts/D_ncc.npy",
    "frechet":  "artifacts/D_frechet.npy",
}

available = {name: p for name, p in candidates.items() if Path(p).exists()}
if not available:
    raise SystemExit("Aucune matrice trouvée. Lance d'abord 03_pairwise_dtw, 03b_pairwise_softdtw et/ou 03b_pairwise_baselines.")

rows = []
detail = []

K_RANGE = range(2, 9)  # K de 2 à 8

for name, path in available.items():
    D_raw = np.load(path)
    D = sanitize_distance_matrix(D_raw)

    # affinité pour spectral clustering
    S = distance_to_affinity(D)

    best_k = None
    best_sil = -1.0
    best_labels = None

    for k in K_RANGE:
        clus = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = clus.fit_predict(S)

        # silhouette avec distances pré-calculées
        sil = silhouette_score(D, labels, metric="precomputed")
        detail.append({"metric": name, "k": k, "silhouette": float(sil)})

        if sil > best_sil:
            best_sil = float(sil)
            best_k = int(k)
            best_labels = labels

    rows.append({"metric": name, "best_k": best_k, "silhouette": best_sil})
    print(f"{name:8s} -> best_k={best_k}  silhouette={best_sil:.3f}")

# Résumés
df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
df_detail = pd.DataFrame(detail).sort_values(["metric", "k"])

# Sauvegardes
out_md = ["# Metric benchmark (spectral clustering + silhouette)\n",
          "| metric | best_k | silhouette |",
          "|---|---:|---:|"]
for _, r in df.iterrows():
    out_md.append(f"| {r['metric']} | {int(r['best_k'])} | {r['silhouette']:.3f} |")

Path("artifacts/metric_benchmark.md").write_text("\n".join(out_md), encoding="utf-8")
df.to_csv("artifacts/metric_benchmark.csv", index=False)
df_detail.to_csv("artifacts/metric_benchmark_detail.csv", index=False)

print("📝 Rapport écrit -> artifacts/metric_benchmark.md")
print("📄 CSV -> artifacts/metric_benchmark.csv / metric_benchmark_detail.csv")
