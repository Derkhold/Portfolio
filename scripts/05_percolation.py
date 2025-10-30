# --- python -m scripts.05_percolation [--metric softdtw] [--k 10] [--mode quantile|weight|distance] [--taus "0.50:0.995:60"]
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cmorph.graph import mutual_knn  # k-NN backbone only; we set weights ourselves

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

# -------------------- helpers --------------------
def load_distance(metric: str) -> np.ndarray:
    p = ART / f"D_{metric}.npy"
    if not p.exists():
        raise SystemExit(f"Distance matrix not found: {p}")
    D = np.load(p).astype(float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise SystemExit(f"Invalid distance matrix shape: {D.shape}")
    # symmetrize + non-negative + diag=0
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    mn = np.nanmin(D)
    if mn < 0:
        D = D - mn           # shift so D >= 0
        np.fill_diagonal(D, 0.0)
    return D

def build_graph_with_exp_weights(D: np.ndarray, k: int) -> nx.Graph:
    """Mutual-kNN backbone; then set edge 'weight' = exp(-d/σ) in (0,1]."""
    # robust σ = median of positive off-diagonals
    mask = (D > 0) & np.isfinite(D)
    sigma = np.median(D[mask]) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    # ❗ use a supported mode to build the backbone, then overwrite weights
    G = mutual_knn(D, k=k, weight_mode="exp_median")

    for u, v, d in G.edges(data=True):
        dij = float(D[u, v])
        w = float(np.exp(-dij / sigma))  # ∈ (0,1]
        d["weight"] = w                  # overwrite
        d["dist"]   = dij                # keep raw distance for distance-mode thresholds

    return G

def largest_component_fraction(H: nx.Graph) -> float:
    N = H.number_of_nodes()
    if N == 0: return 0.0
    if H.number_of_edges() == 0: return 1.0 if N > 0 else 0.0
    sizes = [len(c) for c in nx.connected_components(H)]
    return max(sizes) / N if sizes else 0.0

def susceptibility(H: nx.Graph) -> float:
    """Variance-weighted size of non-giant components (simple percolation proxy)."""
    sizes = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    if not sizes: return 0.0
    if len(sizes) == 1: return 0.0
    rest = sizes[1:]
    num = sum(s * s for s in rest)
    den = sum(rest)
    return num / den if den > 0 else 0.0

def parse_range_or_list(s: str) -> np.ndarray | None:
    """Accept 'a:b:n' (inclusive linspace) or comma list; return None if s empty."""
    if not s: return None
    s = s.strip()
    if ":" in s:
        a, b, n = s.split(":")
        a, b, n = float(a), float(b), int(float(n))
        return np.linspace(a, b, n)
    elif "," in s:
        return np.array([float(x) for x in s.split(",") if x.strip()], float)
    else:
        return np.array([float(s)], float)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Percolation scan on the soft-DTW morphology graph")
    ap.add_argument("--metric", default="softdtw", choices=["softdtw","dtw","frechet","corr","ncc"])
    ap.add_argument("--k", type=int, default=10, help="mutual-kNN parameter (use 8–12 for diagnostics)")
    ap.add_argument("--mode", default="quantile", choices=["quantile","weight","distance"],
                    help="threshold meaning: keep top-quantile of weights, or weight>=τ, or distance<=τ")
    ap.add_argument("--taus", default="", help='optional τ grid: "a:b:n" or "t1,t2,..." (else auto)')
    args = ap.parse_args()

    D = load_distance(args.metric)
    G = build_graph_with_exp_weights(D, k=args.k)

    # collect edge stats
    w = np.array([float(d.get("weight", 0.0)) for *_ , d in G.edges(data=True)])
    dist = np.array([float(d.get("dist",   0.0)) for *_ , d in G.edges(data=True)])
    w = w[np.isfinite(w)]; dist = dist[np.isfinite(dist)]

    print("Edge weights exp(-d/σ) quantiles:",
          "min={:.6f}".format(w.min()) if w.size else "NA",
          "p50={:.6f}".format(np.quantile(w,0.50)) if w.size else "",
          "p90={:.6f}".format(np.quantile(w,0.90)) if w.size else "",
          "p95={:.6f}".format(np.quantile(w,0.95)) if w.size else "",
          "p99={:.6f}".format(np.quantile(w,0.99)) if w.size else "",
          "max={:.6f}".format(w.max()) if w.size else "", sep="  ")

    # τ grid
    user_taus = parse_range_or_list(args.taus)
    if user_taus is not None:
        taus = np.asarray(user_taus, float)
    else:
        if args.mode == "quantile":
            # scan where the mass of weights actually lies
            taus = np.quantile(w, np.linspace(0.50, 0.995, 60))
        elif args.mode == "weight":
            taus = np.linspace(float(w.min()), float(w.max()), 60)
        else:  # distance
            taus = np.quantile(dist, np.linspace(0.00, 0.99, 60))

    rows = []
    N = G.number_of_nodes()
    for t in taus:
        if args.mode == "quantile":
            thr = float(t)  # already a weight threshold from quantiles
            H = nx.Graph(); H.add_nodes_from(G.nodes())
            H.add_edges_from((u,v,d) for u,v,d in G.edges(data=True)
                             if float(d.get("weight",0.0)) >= thr)
        elif args.mode == "weight":
            thr = float(t)
            H = nx.Graph(); H.add_nodes_from(G.nodes())
            H.add_edges_from((u,v,d) for u,v,d in G.edges(data=True)
                             if float(d.get("weight",0.0)) >= thr)
        else:  # distance: keep edges with distance <= τ
            thr = float(t)
            H = nx.Graph(); H.add_nodes_from(G.nodes())
            H.add_edges_from((u,v,d) for u,v,d in G.edges(data=True)
                             if float(d.get("dist", np.inf)) <= thr)

        Grel = largest_component_fraction(H)
        chi  = susceptibility(H)
        rows.append({"tau": float(t), "G": float(Grel), "chi": float(chi)})

    scan = pd.DataFrame(rows)
    # τ_c at max χ
    idx = int(np.nanargmax(scan["chi"].to_numpy())) if len(scan) else 0
    tau_c = float(scan.iloc[idx]["tau"]) if len(scan) else float("nan")

    # save CSV
    out_csv = ART / "percolation_scan.csv"
    scan.to_csv(out_csv, index=False)
    print(f"📄 wrote {out_csv}")

    # plot
    fig, ax1 = plt.subplots(figsize=(9.5, 3.6))
    ax2 = ax1.twinx()
    ax1.plot(scan["tau"], scan["G"], "-s", lw=2.0, ms=4, label="G(τ)")
    ax2.plot(scan["tau"], scan["chi"], "--o", lw=1.5, ms=3, alpha=0.9, label="χ(τ)")

    ax1.axvline(tau_c, color="crimson", ls=":", lw=1.6)
    ax1.set_title(f"Percolation (k={args.k}, mode={args.mode})")
    ax1.set_xlabel("τ" + (" (top-quantile kept)" if args.mode=="quantile"
                           else " (weight threshold)" if args.mode=="weight"
                           else " (distance threshold)"))
    ax1.set_ylabel("Relative size of largest component  G(τ)")
    ax2.set_ylabel("Susceptibility  χ(τ)")

    # single legend
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_legend_handles_labels()
        lines += L[0]; labels += L[1]
    ax1.legend(lines, labels, loc="lower right", frameon=False)

    plt.tight_layout()
    out_png = ART / "percolation_curves.png"
    plt.savefig(out_png, dpi=180)
    print(f"✅ saved {out_png}  |  tau_c≈ {tau_c:.4f}")

if __name__ == "__main__":
    main()
