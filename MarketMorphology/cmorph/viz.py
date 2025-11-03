import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# --- Window plot -------------------------------------------------------------

def plot_window(win: np.ndarray, ax=None, title: str = "Window"):
    """
    Plot a single normalized price window (shape curve).
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(win, color="black")
    ax.set_title(title)
    return ax


# --- Graph visualization -----------------------------------------------------

def plot_graph(G: nx.Graph, n_nodes: int = 200, seed: int = 42, out_path: str | None = None):
    """
    Visualize a subsample of the similarity graph (for reports or diagnostics).
    """
    H = G.subgraph(list(G.nodes())[:n_nodes]) if G.number_of_nodes() > n_nodes else G
    pos = nx.spring_layout(H, seed=seed)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(H, pos, node_size=20, node_color="skyblue")
    nx.draw_networkx_edges(H, pos, alpha=0.3)
    plt.title("Mutual k-NN graph (sample)")
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"[saved] graph → {out_path}")


# --- Percolation plots -------------------------------------------------------

def plot_percolation(
    taus: np.ndarray,
    sizes: np.ndarray,
    chis: np.ndarray,
    tau_c: float,
    out_path: str | None = None,
):
    """
    Plot percolation results: LCC size and susceptibility χ(τ).
    """
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(taus, sizes, label="LCC size")
    plt.axvline(tau_c, color="r", ls="--")
    plt.title("Percolation G(τ)")

    plt.subplot(1, 2, 2)
    plt.plot(taus, chis, color="g", label="χ")
    plt.axvline(tau_c, color="r", ls="--")
    plt.title("Susceptibility χ(τ)")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"[saved] percolation plots → {out_path}")
