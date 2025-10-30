from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Literal, Tuple, Dict, Any, List
from networkx.algorithms import community as nx_comm

# ---------- weights ----------

def _minmax_weights(dist: np.ndarray) -> np.ndarray:
    """
    Convert distances to weights in [0,1] with min-max on off-diagonal entries:
        w_ij = 1 - (d_ij - d_min)/(d_max - d_min)
    Diagonal forced to 0. If degenerate, fall back to 0.5 (diag=0).
    """
    D = np.asarray(dist, float)
    U = D.copy()
    np.fill_diagonal(U, np.nan)
    dmin = np.nanmin(U)
    dmax = np.nanmax(U)
    W = np.zeros_like(D, float)
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        W.fill(0.5)
        np.fill_diagonal(W, 0.0)
        return W
    Z = (D - dmin) / (dmax - dmin + 1e-12)
    W = 1.0 - Z
    W = np.clip(W, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    return W

# ---------- graph construction ----------

def mutual_knn(
    dist: np.ndarray,
    k: int = 15,
    weight_mode: Literal["minmax", "exp_median"] = "minmax",
) -> nx.Graph:
    """
    Mutual k-NN graph from an NxN distance matrix.
    Edge attributes:
      - 'weight' in [0,1]
      - 'dist'   original distance
    """
    dist = np.asarray(dist, float)
    N = dist.shape[0]

    # neighbor list (skip self)
    nn = np.argsort(dist, axis=1)[:, 1:k+1]

    # weights
    if weight_mode == "minmax":
        W = _minmax_weights(dist)
    elif weight_mode == "exp_median":
        m = np.median(dist[dist > 0])
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        W = np.exp(-dist / m)
        np.fill_diagonal(W, 0.0)
        W = np.clip(W, 0.0, 1.0)
    else:
        raise ValueError("Unknown weight_mode")

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in nn[i]:
            if i in nn[j]:
                G.add_edge(int(i), int(j),
                           weight=float(W[i, j]),
                           dist=float(dist[i, j]))
    return G

def graph_stats(G: nx.Graph) -> dict:
    degs = [d for _, d in G.degree()]
    comps = list(nx.connected_components(G))
    lcc = max((len(c) for c in comps), default=0)
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "deg_mean": float(np.mean(degs)) if degs else 0.0,
        "deg_max": int(np.max(degs)) if degs else 0,
        "n_comps": len(comps),
        "lcc_size": lcc,
    }

# ---------- helpers LCC ----------

def lcc_nodes(G: nx.Graph) -> list[int]:
    if G.number_of_nodes() == 0:
        return []
    comps = list(nx.connected_components(G))
    if not comps:
        return []
    return list(max(comps, key=len))

def eigenvector_centrality_lcc(G: nx.Graph) -> np.ndarray:
    """
    Eigenvector centrality calculée sur la LCC uniquement,
    0 ailleurs. Fallback Katz si l'eig ne converge pas.
    """
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    out = np.zeros(len(nodes), float)
    lcc = lcc_nodes(G)
    if not lcc:
        return out
    H = G.subgraph(lcc)
    try:
        vec = nx.eigenvector_centrality_numpy(H, weight="weight")
    except Exception:
        # fallback plus robuste
        vec = nx.katz_centrality_numpy(H, alpha=0.1, beta=1.0, weight="weight")
    for u, v in vec.items():
        out[idx[u]] = float(v)
    return out

def average_shortest_path_length_lcc(G: nx.Graph) -> float | None:
    lcc = lcc_nodes(G)
    if len(lcc) <= 1:
        return None
    H = G.subgraph(lcc)
    try:
        return float(nx.average_shortest_path_length(H, weight="dist"))
    except Exception:
        return None

# ---------- centralities ----------

def centrality_measures(G: nx.Graph) -> dict:
    """
    Returns dict of arrays (aligned to node order):
      degree, strength, betweenness, eigenvector, clustering_coef
    """
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    out = {}

    # degree
    deg = np.zeros(len(nodes), float)
    for n, d in G.degree():
        deg[idx[n]] = d
    out["degree"] = deg

    # strength (sum of weights)
    strg = np.zeros(len(nodes), float)
    for n, d in G.degree(weight="weight"):
        strg[idx[n]] = d
    out["strength"] = strg

    # betweenness (weighted by geodesic distance)
    btw = nx.betweenness_centrality(G, weight="dist", normalized=True)
    out["betweenness"] = np.array([btw.get(n, 0.0) for n in nodes], float)

    # eigenvector on LCC (0 elsewhere) with Katz fallback
    out["eigenvector"] = eigenvector_centrality_lcc(G)

    # clustering coefficient (weighted)
    cc = nx.clustering(G, weight="weight")
    out["clustering_coef"] = np.array([cc.get(n, 0.0) for n in nodes], float)

    return out

# ---------- robustness (edge removal) ----------

def _edge_list_sorted_by_weight(G: nx.Graph, ascending: bool = True) -> List[Tuple[int,int,float]]:
    edges = []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        edges.append((u, v, w))
    edges.sort(key=lambda x: x[2], reverse=not ascending)
    return edges

def robustness_stats(G: nx.Graph, steps: int = 20) -> dict:
    """
    Remove edges from lowest->highest weight in 'steps' quantiles.
    Returns dict with arrays:
      'tau' (fraction removed), 'G' (LCC ratio), 'chi' (susceptibility proxy)
    """
    H = G.copy()
    E = H.number_of_edges()
    if E == 0:
        return {"tau": np.array([0.0]), "G": np.array([1.0]), "chi": np.array([0.0])}

    edges = _edge_list_sorted_by_weight(H, ascending=True)
    batches = np.array_split(np.arange(len(edges)), steps)
    tau, Grel, chi = [], [], []

    N = H.number_of_nodes()
    removed = 0
    for b in batches:
        for idx in b:
            u, v, _ = edges[idx]
            if H.has_edge(u, v):
                H.remove_edge(u, v)
                removed += 1

        comps = [len(c) for c in nx.connected_components(H)]
        lcc = max(comps) if comps else 0
        rest = [c for c in comps if c != lcc]
        sus = (np.sum(np.array(rest, float)**2) / N) if rest else 0.0

        tau.append(removed / E)
        Grel.append(lcc / N if N else 0.0)
        chi.append(sus)

    return {"tau": np.array(tau), "G": np.array(Grel), "chi": np.array(chi)}

# ---------- affinity helper ----------

def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, float)
    mask = (D > 0) & np.isfinite(D)
    sigma = np.median(D[mask]) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S

# ---------- global graph quality stats ----------

def global_graph_stats(G: nx.Graph, labels: np.ndarray | None = None) -> dict:
    """
    Extra KPIs pour le papier/mémoire.
    - assortativity (degree, weighted)
    - transitivity (global clustering)
    - avg shortest path on LCC (weighted by 'dist')
    - modularity (si labels fournis)
    """
    out = {}
    try:
        out["assortativity_deg"] = float(nx.degree_assortativity_coefficient(G, weight="weight"))
    except Exception:
        out["assortativity_deg"] = np.nan
    try:
        out["transitivity"] = float(nx.transitivity(G))
    except Exception:
        out["transitivity"] = np.nan

    asp = average_shortest_path_length_lcc(G)
    out["avg_shortest_path_LCC"] = (float(asp) if asp is not None else np.nan)

    if labels is not None:
        # communautés à partir des labels
        comms = {}
        for i, c in enumerate(labels):
            comms.setdefault(int(c), []).append(i)
        communities = [set(v) for v in comms.values() if len(v) > 0]
        try:
            out["modularity"] = float(nx_comm.quality.modularity(G, communities, weight="weight"))
        except Exception:
            out["modularity"] = np.nan
    return out
