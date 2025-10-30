# --- python -m scripts.05_percolation_panel --metric softdtw --k 10 --out artifacts/fig_5_7_percolation_panel.png
import sys, argparse
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

from cmorph.graph import mutual_knn  # backbone only; we overwrite weights

# ----------------- distance & graph utils -----------------
def load_distance(metric: str) -> np.ndarray:
    p = ART / f"D_{metric}.npy"
    if not p.exists():
        raise SystemExit(f"Distance not found: {p}. Run 03* scripts first.")
    D = np.load(p).astype(float)
    # symmetric, non-negative, diag=0
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    mn = np.nanmin(D)
    if mn < 0:
        D = D - mn
        np.fill_diagonal(D, 0.0)
    return D

def build_graph_with_exp_weights(D: np.ndarray, k: int) -> nx.Graph:
    """
    Build mutual-kNN backbone with a supported mode, then overwrite:
      - edge['weight'] = exp(-d/σ) in (0,1]
      - edge['dist']   = raw distance d
    """
    # supported mode (old versions reject "none"/"distance")
    try:
        G = mutual_knn(D, k=k, weight_mode="exp_median")
    except (TypeError, ValueError):
        G = mutual_knn(D, k=k)

    mask = (D > 0) & np.isfinite(D)
    sigma = np.median(D[mask]) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    for u, v in G.edges():
        dij = float(D[u, v])
        G[u][v]["weight"] = float(np.exp(-dij / sigma))
        G[u][v]["dist"]   = dij
    return G

def comp_stats(H: nx.Graph, N: int) -> tuple[float, float]:
    if H.number_of_nodes() == 0:
        return 0.0, 0.0
    comps = [len(c) for c in nx.connected_components(H)]
    if not comps:
        return 0.0, 0.0
    L = max(comps)
    Grel = L / N
    rest = [s for s in comps if s != L]
    chi = (np.sum(np.square(rest)) / max(np.sum(rest), 1)) if rest else 0.0
    return float(Grel), float(chi)

# ----------------- scans -----------------
def scan_quantile(G: nx.Graph, taus: np.ndarray):
    N = G.number_of_nodes()
    ws = np.array([d.get("weight", 0.0) for *_, d in G.edges(data=True)], float)
    order = np.argsort(ws)
    Gvals, Chivals = [], []
    for q in taus:
        q = float(np.clip(q, 0.0, 1.0))
        k = int(np.ceil((1.0 - q) * len(order)))
        thr_index = order[k] if len(order) > 0 else 0
        thr = ws[thr_index] if len(ws) else np.inf
        H = nx.Graph(); H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= thr)
        g, c = comp_stats(H, N)
        Gvals.append(g); Chivals.append(c)
    Gvals, Chivals = np.array(Gvals), np.array(Chivals)
    tau_c = float(taus[int(np.argmax(Chivals))])
    return taus, Gvals, Chivals, tau_c

def scan_weight(G: nx.Graph, taus: np.ndarray):
    N = G.number_of_nodes()
    Gvals, Chivals = [], []
    for thr in taus:
        H = nx.Graph(); H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= float(thr))
        g, c = comp_stats(H, N)
        Gvals.append(g); Chivals.append(c)
    Gvals, Chivals = np.array(Gvals), np.array(Chivals)
    tau_c = float(taus[int(np.argmax(Chivals))])
    return taus, Gvals, Chivals, tau_c

def scan_distance(G: nx.Graph, taus: np.ndarray):
    N = G.number_of_nodes()
    Gvals, Chivals = [], []
    for thr in taus:
        H = nx.Graph(); H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d["dist"] <= float(thr))
        g, c = comp_stats(H, N)
        Gvals.append(g); Chivals.append(c)
    Gvals, Chivals = np.array(Gvals), np.array(Chivals)
    tau_c = float(taus[int(np.argmax(Chivals))])
    return taus, Gvals, Chivals, tau_c

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="softdtw")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=str(ART / "fig_5_7_percolation_panel.png"))
    args = ap.parse_args()

    D = load_distance(args.metric)
    G = build_graph_with_exp_weights(D, k=args.k)

    ws = np.array([d.get("weight", 0.0) for *_, d in G.edges(data=True)], float)
    ds = np.array([d.get("dist",   0.0) for *_, d in G.edges(data=True)], float)
    wq = np.quantile(ws, [0.60, 0.99])
    dq = np.quantile(ds, [0.01, 0.30])

    taus_q = np.linspace(0.60, 0.99, 50)
    taus_w = np.linspace(float(wq[0]), float(wq[1]), 60)
    taus_d = np.linspace(float(dq[0]), float(dq[1]), 60)

    tq, Gq, Cq, tc_q = scan_quantile(G, taus_q)
    tw, Gw, Cw, tc_w = scan_weight  (G, taus_w)
    td, Gd, Cd, tc_d = scan_distance(G, taus_d)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    def plot_one(ax, axr, t, Gs, Cs, tc, title, xlabel):
        ax.plot(t, Gs, marker="s", lw=2, label="G(τ)")
        axr.plot(t, Cs, marker="o", lw=1.6, ls="--", label="χ(τ)")
        ax.axvline(tc, color="crimson", ls=":", lw=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative size of largest component  G(τ)")
        axr.set_ylabel("Susceptibility  χ(τ)")
        ax.set_title(title)
        # joint legend
        lines, labels = [], []
        for a in (ax, axr):
            L = a.get_legend_handles_labels()
            lines += L[0]; labels += L[1]
        ax.legend(lines, labels, loc="best", frameon=False)

    plot_one(axs[0], axs[0].twinx(), tq, Gq, Cq, tc_q,
             f"Percolation (k={args.k}, mode=quantile)\nτc≈{tc_q:.3f}",
             "τ (top-quantile kept)")
    plot_one(axs[1], axs[1].twinx(), tw, Gw, Cw, tc_w,
             f"Percolation (k={args.k}, mode=weight)\nτc≈{tc_w:.3f}",
             "τ (weight threshold)")
    plot_one(axs[2], axs[2].twinx(), td, Gd, Cd, tc_d,
             f"Percolation (k={args.k}, mode=distance)\nτc≈{tc_d:.3f}",
             "τ (distance threshold)")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=180)
    print(f"✅ saved panel -> {out}")

if __name__ == "__main__":
    main()
