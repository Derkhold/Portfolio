# scripts/05_percolation.py
# Usage:
#   python -m scripts.05a_percolation \
#       --metric softdtw \
#       --k 10 \
#       --mode quantile|weight|distance \
#       --taus "0.50:0.995:60"   # optional: "a:b:n" or comma list

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# project root for "cmorph"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.graph import mutual_knn  # backbone; weights overwritten

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


# -------------------- helpers --------------------
def load_distance(metric: str) -> np.ndarray:
    """
    Load artifacts/D_<metric>.npy and return a symmetric, non-negative matrix with zero diagonal.
    """
    p = ART / f"D_{metric}.npy"
    if not p.exists():
        raise SystemExit(f"Distance matrix not found: {p}")
    D = np.load(p).astype(float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise SystemExit(f"Invalid distance matrix shape: {D.shape}")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    mn = float(np.nanmin(D))
    if mn < 0:
        D = D - mn
        np.fill_diagonal(D, 0.0)
    return D


def build_graph_with_exp_weights(D: np.ndarray, k: int) -> nx.Graph:
    """
    Mutual-kNN backbone. Set:
      edge['weight'] = exp(-d / sigma)
      edge['dist']   = d
    with sigma = median(D_ij | D_ij > 0).
    """
    mask = (D > 0) & np.isfinite(D)
    sigma = float(np.median(D[mask])) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    G = mutual_knn(D, k=k, weight_mode="exp_median")
    for u, v, d in G.edges(data=True):
        dij = float(D[u, v])
        d["weight"] = float(np.exp(-dij / sigma))
        d["dist"] = dij
    return G


def largest_component_fraction(H: nx.Graph) -> float:
    n = H.number_of_nodes()
    if n == 0:
        return 0.0
    sizes = [len(c) for c in nx.connected_components(H)]
    return (max(sizes) / n) if sizes else 0.0


def susceptibility(H: nx.Graph) -> float:
    """
    Simple percolation proxy: sum(s^2)/sum(s) over components excluding the LCC.
    """
    sizes = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    if not sizes or len(sizes) == 1:
        return 0.0
    rest = sizes[1:]
    num = float(sum(s * s for s in rest))
    den = float(sum(rest))
    return (num / den) if den > 0 else 0.0


def parse_range_or_list(s: str) -> np.ndarray | None:
    """
    Accept "a:b:n" (linspace) or "t1,t2,...". Return None if empty.
    """
    if not s:
        return None
    s = s.strip()
    if ":" in s:
        a, b, n = s.split(":")
        a, b, n = float(a), float(b), int(float(n))
        return np.linspace(a, b, n)
    if "," in s:
        return np.array([float(x) for x in s.split(",") if x.strip()], float)
    return np.array([float(s)], float)


# -------------------- main --------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Percolation scan on morphology graph")
    ap.add_argument("--metric", default="softdtw", choices=["softdtw", "dtw", "frechet", "corr", "ncc"])
    ap.add_argument("--k", type=int, default=10, help="mutual-k parameter")
    ap.add_argument("--mode", default="quantile", choices=["quantile", "weight", "distance"],
                    help="threshold semantics")
    ap.add_argument("--taus", default="", help='tau grid: "a:b:n" or "t1,t2,..." (else auto)')
    args = ap.parse_args()

    D = load_distance(args.metric)
    G = build_graph_with_exp_weights(D, k=args.k)

    # Edge stats (for auto tau grids)
    ws = np.array([float(d.get("weight", 0.0)) for *_, d in G.edges(data=True)], float)
    ds = np.array([float(d.get("dist", 0.0)) for *_, d in G.edges(data=True)], float)
    ws = ws[np.isfinite(ws)]
    ds = ds[np.isfinite(ds)]

    # Tau grid
    user_taus = parse_range_or_list(args.taus)
    if user_taus is not None:
        taus = np.asarray(user_taus, float)
    else:
        if args.mode == "quantile":
            # scan where weights actually vary
            taus = np.quantile(ws, np.linspace(0.50, 0.995, 60)) if ws.size else np.linspace(0.5, 0.99, 60)
        elif args.mode == "weight":
            taus = np.linspace(float(ws.min()), float(ws.max()), 60) if ws.size else np.linspace(0.0, 1.0, 60)
        else:  # distance
            taus = np.quantile(ds, np.linspace(0.00, 0.99, 60)) if ds.size else np.linspace(0.0, 1.0, 60)

    # Scan
    rows = []
    for t in taus:
        thr = float(t)
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        if args.mode in ("quantile", "weight"):
            H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)
                             if float(d.get("weight", 0.0)) >= thr)
        else:  # distance
            H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True)
                             if float(d.get("dist", np.inf)) <= thr)

        rows.append({
            "tau": thr,
            "G": largest_component_fraction(H),
            "chi": susceptibility(H),
        })

    scan = pd.DataFrame(rows)
    idx = int(np.nanargmax(scan["chi"].to_numpy())) if len(scan) else 0
    tau_c = float(scan.iloc[idx]["tau"]) if len(scan) else float("nan")

    # Save CSV
    out_csv = ART / "percolation_scan.csv"
    scan.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(9.5, 3.6))
    ax2 = ax1.twinx()
    ax1.plot(scan["tau"], scan["G"], "-s", lw=2.0, ms=4, label="G(τ)")
    ax2.plot(scan["tau"], scan["chi"], "--o", lw=1.5, ms=3, alpha=0.9, label="χ(τ)")
    ax1.axvline(tau_c, color="crimson", ls=":", lw=1.6)

    ax1.set_title(f"Percolation (k={args.k}, mode={args.mode})")
    ax1.set_xlabel("τ" + (" (top-quantile kept)" if args.mode == "quantile"
                          else " (weight threshold)" if args.mode == "weight"
                          else " (distance threshold)"))
    ax1.set_ylabel("Relative size of largest component  G(τ)")
    ax2.set_ylabel("Susceptibility  χ(τ)")

    lines, labels = [], []
    for ax in (ax1, ax2):
        h, lab = ax.get_legend_handles_labels()
        lines += h
        labels += lab
    ax1.legend(lines, labels, loc="lower right", frameon=False)

    out_png = ART / "percolation_curves.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    print(f"Saved {out_png} | tau_c ≈ {tau_c:.4f}")


if __name__ == "__main__":
    main()
