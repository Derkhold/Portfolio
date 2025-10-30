from __future__ import annotations

from typing import Dict, Tuple, Literal

import networkx as nx
import numpy as np


# --- Percolation sweep -------------------------------------------------------

def scan(
    G: nx.Graph,
    taus: np.ndarray,
    mode: Literal["weight", "dist"] = "weight",
) -> Dict[float, Tuple[nx.Graph, float, float]]:
    """
    Percolation sweep.
    mode="weight": keep edges with weight >= tau (tau in [0,1])
    mode="dist":   keep edges with dist   <= tau
    Return: {tau: (G_tau, LCC_ratio, chi)}, with chi = sum(s^2)/sum(s) off the LCC.
    """
    out: Dict[float, Tuple[nx.Graph, float, float]] = {}
    N = G.number_of_nodes()

    if mode == "weight":
        edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if "weight" in d]
        keep = lambda val, tau: val >= tau
    else:
        edges = [(u, v, d["dist"]) for u, v, d in G.edges(data=True) if "dist" in d]
        keep = lambda val, tau: val <= tau

    for tau in taus:
        t = float(tau)
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from([(u, v) for u, v, val in edges if keep(val, t)])

        comps = [len(c) for c in nx.connected_components(H)]
        if not comps:
            out[t] = (H, 0.0, 0.0)
            continue

        lcc = max(comps)
        rest = [s for s in comps if s != lcc]
        chi = (sum(s * s for s in rest) / sum(rest)) if rest else 0.0

        out[t] = (H, lcc / N if N else 0.0, float(chi))
    return out


def find_tau_c(results: dict, by: str = "peak_chi") -> float:
    """
    Critical threshold from sweep results:
      - by="peak_chi": argmax chi(tau)
      - otherwise: argmax |d(LCC_ratio)/d tau|
    """
    taus = list(results.keys())
    chis = [results[t][2] for t in taus]
    if by == "peak_chi":
        return float(taus[int(np.argmax(chis))])

    sizes = [results[t][1] for t in taus]
    d = np.gradient(sizes, taus)
    return float(taus[int(np.argmax(np.abs(d)))])
