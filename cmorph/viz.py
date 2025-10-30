import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_window(win, ax=None, title="Window"):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(win, color="black")
    ax.set_title(title)
    return ax

def plot_graph(G, n_nodes=200, seed=42, out_path=None):
    """
    Visualise un sous-graphe (utile pour mémoire)
    """
    if G.number_of_nodes() > n_nodes:
        nodes = list(G.nodes())[:n_nodes]
        H = G.subgraph(nodes)
    else:
        H = G
    pos = nx.spring_layout(H, seed=seed)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(H, pos, node_size=20, node_color="skyblue")
    nx.draw_networkx_edges(H, pos, alpha=0.3)
    plt.title("k-NN mutual graph (sample)")
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"📈 Graph saved: {out_path}")

def plot_percolation(taus, sizes, chis, tau_c, out_path=None):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(taus, sizes, label="LCC size")
    plt.axvline(tau_c, color="r", ls="--")
    plt.title("Percolation G(tau)")
    plt.subplot(1,2,2)
    plt.plot(taus, chis, label="chi", color="g")
    plt.axvline(tau_c, color="r", ls="--")
    plt.title("Susceptibility χ(tau)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"📈 Percolation curves saved: {out_path}")

