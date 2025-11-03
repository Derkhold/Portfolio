# scripts/03e_compare_metrics.py
# Compare distance matrices via spectral clustering + silhouette

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


# --- Utils -------------------------------------------------------------------

def sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Make a 'distance' matrix usable with silhouette(metric='precomputed'):
    - enforce symmetry and zero diagonal
    - shift if negatives exist (make D >= 0)
    - replace NaN/inf by max finite
    """
    D = np.asarray(D, float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    if not np.isfinite(D).all():
        finite = D[np.isfinite(D)]
        repl = float(np.nanmax(finite)) if finite.size else 0.0
        D = np.where(np.isfinite(D), D, repl)
        np.fill_diagonal(D, 0.0)

    mn = float(np.min(D))
    if mn < -1e-12:
        D = D - mn + 1e-12
        np.fill_diagonal(D, 0.0)
    else:
        D = np.maximum(D, 0.0)

    assert np.allclose(D, D.T), "Distance matrix must be symmetric"
    assert np.all(D.diagonal() == 0.0), "Diagonal must be zero"
    assert np.min(D) >= 0.0, "Distances must be non-negative"
    return D


def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    """
    Convert distances to positive affinities for SpectralClustering:
    S_ij = exp(-D_ij / sigma), with sigma = median(D_ij > 0).
    """
    D = np.asarray(D, float)
    mask = (D > 0) & np.isfinite(D)
    sigma = float(np.median(D[mask])) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S


# --- Main --------------------------------------------------------------------

candidates = {
    "dtw":     "artifacts/D_dtw.npy",
    "softdtw": "artifacts/D_softdtw.npy",
    "corr":    "artifacts/D_corr.npy",
    "ncc":     "artifacts/D_ncc.npy",
    "frechet": "artifacts/D_frechet.npy",
}

available = {name: p for name, p in candidates.items() if Path(p).exists()}
if not available:
    raise SystemExit("No distance matrices found. Run 03_pairwise_dtw / 03b_pairwise_softdtw / 03b_pairwise_baselines first.")

rows, detail = [], []
K_RANGE = range(2, 9)

for name, path in available.items():
    D_raw = np.load(path)
    D = sanitize_distance_matrix(D_raw)
    S = distance_to_affinity(D)

    best_k, best_sil, best_labels = None, -1.0, None

    for k in K_RANGE:
        clus = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = clus.fit_predict(S)
        sil = silhouette_score(D, labels, metric="precomputed")
        detail.append({"metric": name, "k": k, "silhouette": float(sil)})

        if sil > best_sil:
            best_sil = float(sil)
            best_k = int(k)
            best_labels = labels

    rows.append({"metric": name, "best_k": best_k, "silhouette": best_sil})
    print(f"{name:8s} -> best_k={best_k}  silhouette={best_sil:.3f}")

# Summaries
df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
df_detail = pd.DataFrame(detail).sort_values(["metric", "k"])

# Outputs
md_lines = [
    "# Metric benchmark (spectral clustering + silhouette)",
    "| metric | best_k | silhouette |",
    "|---|---:|---:|",
]
for _, r in df.iterrows():
    md_lines.append(f"| {r['metric']} | {int(r['best_k'])} | {r['silhouette']:.3f} |")

Path("artifacts/metric_benchmark.md").write_text("\n".join(md_lines), encoding="utf-8")
df.to_csv("artifacts/metric_benchmark.csv", index=False)
df_detail.to_csv("artifacts/metric_benchmark_detail.csv", index=False)

print("Wrote: artifacts/metric_benchmark.md")
print("Wrote: artifacts/metric_benchmark.csv, artifacts/metric_benchmark_detail.csv")
