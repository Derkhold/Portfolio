from __future__ import annotations

from typing import Literal, Tuple, List

import networkx as nx
import numpy as np
from networkx.algorithms import community as nx_comm


# --- Weights -----------------------------------------------------------------

def _minmax_weights(dist: np.ndarray) -> np.ndarray:
    """
    Map distances to weights in [0, 1] using min–max on off-diagonal entries.
    Diagonal is forced to 0. If degenerate, return 0.5 off-diagonal.
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


# --- Graph construction ------------------------------------------------------

def mutual_knn(
    dist: np.ndarray,
    k: int = 15,
    weight_mode: Literal["minmax", "exp_median"] = "minmax",
) -> nx.Graph:
    """
    Build a mutual k-NN graph from an NxN distance matrix.
    Edge attrs: 'weight' in [0,1], 'dist' original distance.
    """
    dist = np.asarray(dist, float)
    N = dist.shape[0]

    nn = np.argsort(dist, axis=1)[:, 1 : k + 1]

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
                G.add_edge(int(i), int(j), weight=float(W[i, j]), dist=float(dist[i, j]))
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


# --- LCC helpers -------------------------------------------------------------

def lcc_nodes(G: nx.Graph) -> list[int]:
    if G.number_of_nodes() == 0:
        return []
    comps = list(nx.connected_components(G))
    if not comps:
        return []
    return list(max(comps, key=len))


def eigenvector_centrality_lcc(G: nx.Graph) -> np.ndarray:
    """
    Eigenvector centrality on the LCC, zeros elsewhere. Katz fallback on failure.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    out = np.zeros(len(nodes), float)
    lcc = lcc_nodes(G)
    if not lcc:
        return out
    H = G.subgraph(lcc)
    try:
        vec = nx.eigenvector_centrality_numpy(H, weight="weight")
    except Exception:
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


# --- Centralities ------------------------------------------------------------

def centrality_measures(G: nx.Graph) -> dict:
    """
    Return dict of arrays (aligned to node order):
    degree, strength, betweenness, eigenvector, clustering_coef.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    out = {}

    deg = np.zeros(len(nodes), float)
    for n, d in G.degree():
        deg[idx[n]] = d
    out["degree"] = deg

    strg = np.zeros(len(nodes), float)
    for n, d in G.degree(weight="weight"):
        strg[idx[n]] = d
    out["strength"] = strg

    btw = nx.betweenness_centrality(G, weight="dist", normalized=True)
    out["betweenness"] = np.array([btw.get(n, 0.0) for n in nodes], float)

    out["eigenvector"] = eigenvector_centrality_lcc(G)

    cc = nx.clustering(G, weight="weight")
    out["clustering_coef"] = np.array([cc.get(n, 0.0) for n in nodes], float)

    return out


# --- Robustness (edge removal) ----------------------------------------------

def _edge_list_sorted_by_weight(G: nx.Graph, ascending: bool = True) -> List[Tuple[int, int, float]]:
    edges = []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        edges.append((u, v, w))
    edges.sort(key=lambda x: x[2], reverse=not ascending)
    return edges


def robustness_stats(G: nx.Graph, steps: int = 20) -> dict:
    """
    Remove edges from lowest→highest weight in 'steps' batches.
    Returns:
      tau (fraction removed), G (LCC ratio), chi (susceptibility proxy).
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
        sus = (np.sum(np.array(rest, float) ** 2) / N) if rest else 0.0

        tau.append(removed / E)
        Grel.append(lcc / N if N else 0.0)
        chi.append(sus)

    return {"tau": np.array(tau), "G": np.array(Grel), "chi": np.array(chi)}


# --- Distance → affinity -----------------------------------------------------

def distance_to_affinity(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, float)
    mask = (D > 0) & np.isfinite(D)
    sigma = np.median(D[mask]) if np.any(mask) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    S = np.exp(-D / sigma)
    np.fill_diagonal(S, 1.0)
    return S


# --- Global graph quality ----------------------------------------------------

def global_graph_stats(G: nx.Graph, labels: np.ndarray | None = None) -> dict:
    """
    Return global KPIs:
      assortativity (degree, weighted), transitivity,
      avg shortest path (LCC, 'dist' weighted), modularity (if labels).
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
    out["avg_shortest_path_LCC"] = float(asp) if asp is not None else np.nan

    if labels is not None:
        groups = {}
        for i, c in enumerate(labels):
            groups.setdefault(int(c), []).append(i)
        communities = [set(v) for v in groups.values() if v]
        try:
            out["modularity"] = float(nx_comm.quality.modularity(G, communities, weight="weight"))
        except Exception:
            out["modularity"] = np.nan
    return out
