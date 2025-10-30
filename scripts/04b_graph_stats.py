# --- python -m scripts.04b_graph_stats --metric softdtw --k 15 --labels artifacts/labels_softdtw_K2.csv
import sys, argparse, pickle, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

# essaie d'importer les helpers de cmorph.graph ; fallback local si absent
try:
    from cmorph.graph import (
        mutual_knn, graph_stats, centrality_measures,
        robustness_stats, distance_to_affinity,  # peut ne pas être utilisé ici
        global_graph_stats as _global_graph_stats  # peut ne pas exister
    )
    HAS_GLOBAL = True
except Exception:
    from cmorph.graph import mutual_knn, graph_stats, centrality_measures, robustness_stats
    HAS_GLOBAL = False

from networkx.algorithms.community import modularity as nx_modularity

ART = Path("artifacts")

def load_labels(path: Path | None) -> np.ndarray | None:
    """
    Charge un vecteur de labels depuis CSV. Supporte:
      - 1 seule colonne -> labels
      - colonne 'label'
      - sinon dernière colonne
    """
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        lab = df.iloc[:, 0].to_numpy()
    elif "label" in df.columns:
        lab = df["label"].to_numpy()
    else:
        lab = df.iloc[:, -1].to_numpy()
    # force en int si possible
    try:
        lab = lab.astype(int)
    except Exception:
        pass
    return lab

# --- fallback local si global_graph_stats manquant ---
def _largest_cc(G):
    comps = list(nx.connected_components(G))
    if not comps:
        return G.copy()
    return G.subgraph(max(comps, key=len)).copy()

def _global_stats_fallback(G: nx.Graph, labels: np.ndarray | None) -> dict:
    H = _largest_cc(G)
    try:
        asp = nx.average_shortest_path_length(H, weight="dist")
    except Exception:
        asp = float("nan")
    assort = nx.degree_assortativity_coefficient(G)
    trans  = nx.transitivity(G)
    mod = float("nan")
    if labels is not None and len(labels) == G.number_of_nodes():
        comms = [set(np.where(labels==lab)[0].tolist()) for lab in np.unique(labels)]
        try:
            mod = nx_modularity(G, comms, weight="weight")
        except Exception:
            pass
    base = graph_stats(G)
    base.update({
        "assortativity_deg": float(assort) if np.isfinite(assort) else float("nan"),
        "transitivity": float(trans),
        "avg_shortest_path_LCC": float(asp),
        "modularity": float(mod),
    })
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["dtw","softdtw","corr","ncc","frechet"], default="softdtw")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--labels", type=str, default="")
    args = ap.parse_args()

    ART.mkdir(exist_ok=True)
    D_path = ART / f"D_{args.metric}.npy"
    if not D_path.exists():
        raise SystemExit(f"Distance introuvable: {D_path}. Lance d’abord les scripts 03*.")

    print(f"📂 Chargement distance: {D_path.name}")
    D = np.load(D_path)

    print("🧮 Construction graphe k-NN mutuel…")
    # on garde weight_mode="exp_median" comme précédemment
    G = mutual_knn(D, k=args.k, weight_mode="exp_median")
    st = graph_stats(G)
    print("✅ Graphe:", st)

    # Sauvegarde du graphe (même nom qu'avant)
    with open(ART / "graph.pkl","wb") as f:
        pickle.dump(G, f)

    # ---------- Centralités ----------
    print("📊 Centralités…")
    cent = centrality_measures(G)
    df_cent = pd.DataFrame(cent)
    df_cent.to_csv(ART / "graph_centrality.csv", index=True)

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6))
    cols = ["degree","strength","betweenness","eigenvector","clustering_coef"]
    for ax, col in zip(axs.flat, cols + [None]):
        if col is None:
            ax.axis("off");
            continue
        x = pd.Series(df_cent[col]).replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(x, bins=24); ax.set_title(col)
    plt.tight_layout(); plt.savefig(ART/"graph_centrality_hists.png", dpi=170)

    # ---------- Robustesse ----------
    print("🧱 Robustesse (edges)…")
    df_rob = pd.DataFrame(robustness_stats(G, steps=20))
    df_rob.to_csv(ART / "graph_robustness_edges.csv", index=False)

    plt.figure(figsize=(7,3.5))
    plt.plot(df_rob["tau"], df_rob["G"], marker="o")
    plt.xlabel("Fraction d'arêtes retirées (faibles poids → forts)")
    plt.ylabel("Taille relative de la LCC")
    plt.title("Robustesse du graphe (suppression d'arêtes)")
    plt.tight_layout(); plt.savefig(ART/"graph_robustness_edges_LCC.png", dpi=170)

    plt.figure(figsize=(7,3.5))
    plt.plot(df_rob["tau"], df_rob["chi"], marker="o")
    plt.xlabel("Fraction d'arêtes retirées")
    plt.ylabel("Susceptibilité (proxy)")
    plt.title("Fragmentation vs suppression d'arêtes")
    plt.tight_layout(); plt.savefig(ART/"graph_robustness_edges_components.png", dpi=170)

    # ---------- KPIs globaux ----------
    lbl_path = Path(args.labels) if args.labels else None
    labels = load_labels(lbl_path)
    if HAS_GLOBAL:
        kpis = _global_graph_stats(G, labels=labels)
        kpis.update(st)
    else:
        kpis = _global_stats_fallback(G, labels)
    (ART / "graph_global_stats.json").write_text(json.dumps(kpis, indent=2), encoding="utf-8")
    print("📄 KPIs globaux -> artifacts/graph_global_stats.json")

    # ---------- Layout (ne casse rien ; ajoute une version colorée si labels) ----------
    print("🗺️  Layout (échantillon)…")
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1/np.sqrt(max(1, G.number_of_nodes())))
    # 1) version neutre (comme avant)
    plt.figure(figsize=(6,6))
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=20, width=0.2, edge_color="0.6")
    plt.axis("off"); plt.tight_layout(); plt.savefig(ART/"graph_layout_sample.png", dpi=170)

    # 2) version colorée par labels (si fournis et compatibles)
    if labels is not None and len(labels) == G.number_of_nodes():
        labs = np.asarray(labels)
        uniq = np.unique(labs)
        # colormap robuste pour ≤ 20 groupes; au-delà on recycle
        base_cmap = mpl.cm.get_cmap("tab20")
        color_map = {lab: base_cmap(i % base_cmap.N) for i, lab in enumerate(uniq)}
        node_colors = [color_map[int(l)] for l in labs]

        plt.figure(figsize=(6.6,6.6))
        nx.draw_networkx(
            G, pos=pos, with_labels=False,
            node_size=24, width=0.25, edge_color="0.75",
            node_color=node_colors
        )
        # petite légende propre
        handles = [mpl.lines.Line2D([0],[0], marker='o', linestyle='',
                                    markersize=8, markerfacecolor=color_map[int(lab)],
                                    markeredgewidth=0, label=f"Cluster {lab}")
                   for lab in uniq]
        plt.legend(handles=handles, loc="lower left", frameon=False, title="Labels")
        plt.axis("off"); plt.tight_layout(); plt.savefig(ART/"graph_layout_bylabel.png", dpi=170)

    print("📁 Sorties dans artifacts/:")
    print(" - graph.pkl")
    print(" - graph_centrality.csv, graph_centrality_hists.png")
    print(" - graph_robustness_edges.csv, graph_robustness_edges_LCC.png, graph_robustness_edges_components.png")
    print(" - graph_global_stats.json")
    print(" - graph_layout_sample.png")
    if labels is not None and len(labels) == G.number_of_nodes():
        print(" - graph_layout_bylabel.png")

if __name__ == "__main__":
    main()
