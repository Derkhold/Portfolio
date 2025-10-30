# --- python -m scripts.04c_cluster --K 2 --metric softdtw
import sys, argparse, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

def sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    if not np.isfinite(D).all():
        finite = D[np.isfinite(D)]
        repl = np.nanmax(finite) if finite.size else 0.0
        D = np.where(np.isfinite(D), D, repl)
        np.fill_diagonal(D, 0.0)

    mn = float(np.min(D))
    if mn < -1e-12:
        D = D - mn + 1e-12
        np.fill_diagonal(D, 0.0)
    else:
        D = np.maximum(D, 0.0)

    np.fill_diagonal(D, 0.0)
    assert np.allclose(D, D.T)
    assert np.all(D.diagonal() == 0.0)
    return D

def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    mask = (D > 0) & np.isfinite(D)
    sigma = np.median(D[mask]) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0: sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S

def cluster_and_score(D: np.ndarray, K: int, seed: int = 42):
    D = sanitize_distance_matrix(D)
    S = distance_to_affinity(D)
    model = SpectralClustering(n_clusters=K, affinity="precomputed",
                               assign_labels="kmeans", random_state=seed)
    labels = model.fit_predict(S)
    np.fill_diagonal(D, 0.0)   # sécurité silhouette
    sil = float(silhouette_score(D, labels, metric="precomputed"))
    return labels, sil, D

def medoids_by_cluster(D: np.ndarray, labels: np.ndarray, top: int = 5):
    """Retourne jusqu'à `top` médoines (indices) par cluster."""
    D = np.asarray(D, float)
    out = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            out[c] = []
            continue
        sub = D[np.ix_(idx, idx)]
        scores = sub.mean(axis=1)  # distance moyenne au cluster
        order = np.argsort(scores)
        out[c] = idx[order[:min(top, len(idx))]].tolist()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--metric", type=str, default="softdtw",
                    choices=["softdtw", "dtw", "frechet", "corr", "ncc"])
    args = ap.parse_args()

    # charge la matrice de distance choisie
    path = ART / f"D_{'softdtw' if args.metric=='softdtw' else args.metric}.npy"
    if not path.exists():
        raise SystemExit(f"Distance introuvable: {path}. Lance d'abord 03_pairwise…")
    D = np.load(path)

    # clustering + silhouette
    labels, sil, D_clean = cluster_and_score(D, K=args.K, seed=42)
    print(f"✅ clustering ({args.metric}, K={args.K}) — silhouette={sil:.3f}")

    # sauve labels
    lab_df = pd.DataFrame({"i": np.arange(len(labels)), "label": labels})
    lab_df.to_csv(ART / f"labels_{args.metric}_K{args.K}.csv", index=False)

    # exemplaires (médoines) et plots
    # on trace sur les courbes normalisées (windows.pkl) ou sur prices si tu préfères
    wins = pickle.load(open(ART / "windows.pkl", "rb"))  # liste d'array 1D
    X = np.stack(wins)

    medoids = medoids_by_cluster(D_clean, labels, top=5)
    for c, idxs in medoids.items():
        if len(idxs) == 0:
            continue
        plt.figure(figsize=(6.2, 3.6))
        for j, i in enumerate(idxs):
            plt.plot(X[i], alpha=0.9, label=f"ex{i}")
        plt.title(f"Exemplaires cluster {c} (metric={args.metric}, K={args.K})")
        plt.tight_layout()
        plt.savefig(ART / f"cluster_{args.metric}_K{args.K}_C{c}_exemplars.png", dpi=170)

    # petite table de taille de clusters
    sizes = (lab_df.groupby("label")["i"].count()
             .reset_index().rename(columns={"i": "count"}))
    sizes.to_csv(ART / f"cluster_sizes_{args.metric}_K{args.K}.csv", index=False)
    print("📄 labels/sizes et figures d’exemplaires écrits dans artifacts/")
