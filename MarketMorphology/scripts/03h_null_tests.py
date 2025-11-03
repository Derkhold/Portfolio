# scripts/03h_null_tests.py
# Null tests: compare real silhouettes vs phase-/block-shuffled surrogates

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
RNG = np.random.default_rng(42)


# --- Utils -------------------------------------------------------------------

def sanitize_distance_matrix(D: np.ndarray) -> np.ndarray:
    """Symmetric, finite, non-negative, with exact zero diagonal."""
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

    assert np.allclose(D, D.T), "Non-symmetric after sanitize"
    assert np.all(D.diagonal() == 0.0), "Non-zero diagonal after sanitize"
    assert np.min(D) >= 0.0, "Negative distances after sanitize"
    return D


def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    """S = exp(-D / sigma), with sigma = median(D_ij > 0)."""
    D = np.asarray(D, float)
    mask = (D > 0) & np.isfinite(D)
    sigma = float(np.median(D[mask])) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S


# --- Null generators ---------------------------------------------------------

def phase_shuffle(x: np.ndarray) -> np.ndarray:
    """Keep FFT magnitudes, randomize phases; return z-scored surrogate."""
    X = np.fft.rfft(x)
    mags = np.abs(X)
    phases = np.angle(X)
    rnd = RNG.uniform(-np.pi, np.pi, size=phases.shape)
    rnd[0] = phases[0]
    if rnd.size > 1:
        rnd[-1] = phases[-1]
    Y = mags * np.exp(1j * rnd)
    y = np.fft.irfft(Y, n=len(x)).real
    y = (y - y.mean()) / (y.std() + 1e-12)
    return y


def block_shuffle(x: np.ndarray, block: int = 3) -> np.ndarray:
    """Shuffle contiguous blocks; return z-scored surrogate."""
    L = len(x)
    b = max(1, int(block))
    blocks = [x[i:i + b] for i in range(0, L, b)]
    RNG.shuffle(blocks)
    y = np.concatenate(blocks)[:L]
    y = (y - y.mean()) / (y.std() + 1e-12)
    return y


# --- Core scoring ------------------------------------------------------------

def silhouette_of(X: np.ndarray, gamma: float = 0.4, K: int = 2) -> float:
    D = pairwise_soft_dtw(X, gamma=gamma, band=0.1, batch=256, normalize=True)
    D = sanitize_distance_matrix(D)
    S = distance_to_affinity(D)
    clus = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    labels = clus.fit_predict(S)
    np.fill_diagonal(D, 0.0)  # required for silhouette with precomputed distances
    return float(silhouette_score(D, labels, metric="precomputed"))


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    wins = pickle.load(open(ART / "windows.pkl", "rb"))
    X = np.stack(wins)  # (N, L)

    gamma = 0.4
    K = 2
    nrep = 5

    t0 = time.time()
    sil_real = silhouette_of(X, gamma=gamma, K=K)

    # Phase-shuffled
    s_phase = []
    for _ in range(nrep):
        Xp = np.stack([phase_shuffle(x) for x in X])
        s_phase.append(silhouette_of(Xp, gamma=gamma, K=K))
    sil_phase = float(np.mean(s_phase))

    # Block-shuffled
    s_block = []
    for _ in range(nrep):
        Xb = np.stack([block_shuffle(x, block=3) for x in X])
        s_block.append(silhouette_of(Xb, gamma=gamma, K=K))
    sil_block = float(np.mean(s_block))

    df = pd.DataFrame({
        "condition": ["real", "phase_shuffle", "block_shuffle"],
        "silhouette": [sil_real, sil_phase, sil_block],
    })
    df.to_csv(ART / "null_tests_silhouette.csv", index=False)

    # Bar plot
    plt.figure(figsize=(5.2, 3.4))
    plt.bar(df["condition"], df["silhouette"])
    for i, v in enumerate(df["silhouette"]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.ylabel("Silhouette")
    plt.title(f"Null tests (soft-DTW γ={gamma}, K={K}, n={nrep})")
    plt.tight_layout()
    plt.savefig(ART / "null_tests_bar.png", dpi=180)
    plt.close()

    print(f"Null tests done in {time.time() - t0:.1f}s")
    print("Saved: artifacts/null_tests_silhouette.csv")
    print("Saved: artifacts/null_tests_bar.png")
