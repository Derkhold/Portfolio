# --- guard: permet de lancer "python -m scripts.09_inspect"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Optionnel selon ce que tu as déjà dans ton repo
try:
    from cmorph.percolation import scan, find_tau_c
except Exception:
    scan = find_tau_c = None

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"📈 saved: {path}")

# -----------------------------
# 1) DTW — heatmap + histogram
# -----------------------------
dtw_path = ART / "D_dtw.npy"
if dtw_path.exists():
    D = np.load(dtw_path)
    print(f"D_dtw.npy: shape={D.shape} min/median/max={D.min():.4g}/{np.median(D):.4g}/{D.max():.4g}")

    plt.figure(figsize=(6,5))
    plt.imshow(D, aspect="auto")
    plt.colorbar(label="DTW distance")
    plt.title("DTW distance matrix")
    savefig(ART / "dtw_heatmap.png")

    triu = D[np.triu_indices_from(D, k=1)]
    plt.figure(figsize=(5,4))
    plt.hist(triu, bins=40)
    plt.title("DTW distances (upper-tri histogram)")
    plt.xlabel("distance"); plt.ylabel("count")
    savefig(ART / "dtw_hist.png")
else:
    print("⚠️ artifacts/D_dtw.npy absent — lance scripts/03_pairwise_dtw d’abord.")

# -----------------------------
# 2) Graphe — stats + exports
# -----------------------------
g_pkl = ART / "graph.pkl"
if g_pkl.exists():
    with open(g_pkl, "rb") as f:
        G = pickle.load(f)
    degs = [d for _, d in G.degree()]
    stats = dict(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        deg_mean=float(np.mean(degs)) if degs else 0.0,
        deg_max=int(np.max(degs)) if degs else 0,
        n_comps=nx.number_connected_components(G),
        lcc_size=max((len(c) for c in nx.connected_components(G)), default=0),
    )
    print("Graph stats:", stats)

    # Exports
    nx.write_graphml(G, ART / "graph.graphml")
    nx.write_gexf(G,     ART / "graph.gexf")
    rows = [{"u": u, "v": v, "weight": d.get("weight", np.nan), "dist": d.get("dist", np.nan)}
            for u, v, d in G.edges(data=True)]
    pd.DataFrame(rows).to_csv(ART / "graph_edges.csv", index=False)
    print("📤 exported: graph.graphml / graph.gexf / graph_edges.csv")

    # Aperçu layout (sous-échantillon si > 400 nœuds)
    H = G
    if G.number_of_nodes() > 400:
        H = G.subgraph(list(G.nodes())[:400])
    pos = nx.spring_layout(H, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(H, pos, node_size=14, node_color="tab:blue", alpha=0.9)
    nx.draw_networkx_edges(H, pos, width=0.4, alpha=0.25)
    plt.axis("off"); plt.title("Mutual k-NN graph (sample)")
    savefig(ART / "graph_layout.png")
else:
    print("⚠️ artifacts/graph.pkl absent — lance scripts/04_graph d’abord.")

# ----------------------------------
# 3) Percolation (poids) — re-scan
# ----------------------------------
if g_pkl.exists() and scan is not None:
    # récupère les poids effectifs
    weights = np.array([d.get("weight", np.nan) for _, _, d in G.edges(data=True)], dtype=float)
    weights = weights[np.isfinite(weights)]
    if len(weights) >= 10:
        lo, hi = np.percentile(weights, [5, 95])
        lo = max(0.0, float(lo) - 1e-6); hi = min(1.0, float(hi) + 1e-6)
        taus = np.linspace(lo, hi, 51)

        results = scan(G, taus, mode="weight")
        sizes = [results[t][1] for t in taus]
        chis  = [results[t][2] for t in taus]
        tau_c = find_tau_c(results)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1); plt.plot(taus, sizes); plt.axvline(tau_c, color="r", ls="--"); plt.title("G(τ) (poids)")
        plt.subplot(1,2,2); plt.plot(taus, chis);  plt.axvline(tau_c, color="r", ls="--"); plt.title("χ(τ) (poids)")
        savefig(ART / "percolation_curves_inspect.png")

        # histogramme des poids
        plt.figure(figsize=(5,4))
        plt.hist(weights, bins=30)
        plt.title("Edge weights"); plt.xlabel("w"); plt.ylabel("count")
        savefig(ART / "edge_weights_hist.png")
        print(f"✅ Percolation re-scan: tau_c ≈ {tau_c:.3f} (range [{lo:.3f},{hi:.3f}])")
    else:
        print("⚠️ Trop peu d’arêtes ou poids non définis pour la percolation.")
else:
    print("ℹ️ Percolation skip: graphe absent ou cmorph.percolation indisponible.")

# --------------------------
# 4) SAW — séries & histos
# --------------------------
saw_csv = ART / "saw.csv"
if saw_csv.exists():
    saw = pd.read_csv(saw_csv)
    print("SAW describe:\n", saw.describe())

    plt.figure(figsize=(7,3))
    plt.plot(saw["S"].values, lw=1)
    plt.title("SAW S over windows"); plt.xlabel("window idx"); plt.ylabel("S")
    savefig(ART / "saw_series.png")

    plt.figure(figsize=(5,4))
    plt.hist(saw["S"].values, bins=20)
    plt.title("Histogram of SAW S")
    savefig(ART / "saw_hist.png")
else:
    print("⚠️ artifacts/saw.csv absent — lance scripts/06_saw.")

# --------------------------
# 5) TDA — distributions
# --------------------------
tda_csv = ART / "tda.csv"
if tda_csv.exists():
    tda = pd.read_csv(tda_csv)
    print("TDA describe (head):\n", tda.describe().head())

    for col in ["pers_max", "pers_sum", "pers_entropy", "n_pairs"]:
        if col in tda.columns:
            plt.figure(figsize=(5,4))
            vals = tda[col].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(vals) > 0:
                plt.hist(vals, bins=20)
            plt.title(f"Histogram of {col}")
            savefig(ART / f"tda_hist_{col}.png")
else:
    print("⚠️ artifacts/tda.csv absent — lance scripts/07_tda.")

# ------------------------------------------------------
# 6) Corrélations & scatters si metrics_merged.csv dispo
# ------------------------------------------------------
merged = ART / "metrics_merged.csv"
if merged.exists():
    df = pd.read_csv(merged)
    print("metrics_merged columns:", list(df.columns))
    # quelques scatter utiles
    pairs = [
        ("RV", "D_higuchi"),
        ("RV", "alpha_dfa"),
        ("RV", "S"),
        ("pers_max", "RV"),
        ("pers_sum", "RV"),
    ]
    for x, y in pairs:
        if x in df.columns and y in df.columns:
            plt.figure(figsize=(5,4))
            xv = df[x].values; yv = df[y].values
            mask = np.isfinite(xv) & np.isfinite(yv)
            plt.scatter(xv[mask], yv[mask], s=8)
            plt.xlabel(x); plt.ylabel(y); plt.title(f"{y} vs {x}")
            savefig(ART / f"scatter_{y}_vs_{x}.png")
else:
    print("ℹ️ metrics_merged.csv absent — lance scripts/08_validate pour le créer.")

# ----------------------------------------
# 7) Aperçu de quelques fenêtres (forme)
# ----------------------------------------
wins_pkl = ART / "windows.pkl"
if wins_pkl.exists():
    wins = pickle.load(open(wins_pkl, "rb"))
    for k in range(min(3, len(wins))):
        plt.figure(figsize=(5,3))
        plt.plot(wins[k], lw=1)
        plt.title(f"Window {k} (normalized shape)")
        savefig(ART / f"window_{k}.png")
else:
    print("ℹ️ artifacts/windows.pkl absent — lance scripts/01_prepare.")
