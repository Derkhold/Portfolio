from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Literal, Dict, Tuple

def scan(G: nx.Graph, taus: np.ndarray, mode: Literal["weight","dist"]="weight"
        ) -> Dict[float, Tuple[nx.Graph, float, float]]:
    """
    Balayage de percolation.
    - mode="weight": on GARDE les arêtes avec weight >= tau (tau in [0,1])
    - mode="dist":   on GARDE les arêtes avec dist    <= tau
    Retour: dict[tau] = (G_tau, LCC/N, chi)  avec chi = sum(s^2)/sum(s) hors LCC
    """
    out = {}
    N = G.number_of_nodes()
    if mode == "weight":
        edges = [(u,v,d["weight"]) for u,v,d in G.edges(data=True) if "weight" in d]
        keep = lambda val, tau: val >= tau
    else:
        edges = [(u,v,d["dist"]) for u,v,d in G.edges(data=True) if "dist" in d]
        keep = lambda val, tau: val <= tau

    for tau in taus:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from([(u,v) for u,v,val in edges if keep(val, float(tau))])
        comps = [len(c) for c in nx.connected_components(H)]
        if not comps:
            out[float(tau)] = (H, 0.0, 0.0); continue
        lcc = max(comps)
        rest = [s for s in comps if s != lcc]
        chi = (sum(s*s for s in rest) / sum(rest)) if rest else 0.0
        out[float(tau)] = (H, lcc / N, float(chi))
    return out

def find_tau_c(results: dict, by: str = "peak_chi") -> float:
    taus = list(results.keys())
    chis = [results[t][2] for t in taus]
    if by == "peak_chi":
        return float(taus[int(np.argmax(chis))])
    sizes = [results[t][1] for t in taus]
    d = np.gradient(sizes, taus)
    return float(taus[int(np.argmax(np.abs(d)))])
