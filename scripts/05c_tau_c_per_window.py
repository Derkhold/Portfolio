# --- python -m scripts.05c_tau_c_per_window --metric softdtw --k 15 --block 78 --step 1
# Sorties :
#   artifacts/tau_c_per_window.csv   (colonnes: i, start?, tau_c, chi_max, G_at_tau_c, k)
#   artifacts/tau_c_per_window.png   (série temporelle si 'start' dispo)

import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from cmorph.graph import mutual_knn   # déjà utilisé ailleurs dans ton pipeline
# (Pas de dépendance à cmorph.percolation : on fait une perco légère locale ici)

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def load_distance(metric: str) -> np.ndarray:
    p = ART / f"D_{metric}.npy"
    if not p.exists():
        raise SystemExit(f"Distance introuvable: {p}. Exécute d’abord les scripts 03* pour calculer la matrice {metric}.")
    D = np.load(p)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise SystemExit(f"Matrice de distance invalide: {p} (shape={D.shape})")
    return D

def weights_from_distances(D_block: np.ndarray) -> np.ndarray:
    """Convertit distances -> poids exp(-d/median(d)). Diagonale 0, symétrique."""
    d = D_block.copy().astype(float)
    np.fill_diagonal(d, np.nan)
    med = np.nanmedian(d)
    if not np.isfinite(med) or med <= 0:
        med = np.nanmean(d[np.isfinite(d)])  # fallback
        if not np.isfinite(med) or med <= 0:
            med = 1.0
    W = np.exp(-d / med)
    np.fill_diagonal(W, 0.0)
    # symétrise par sécurité
    W = np.maximum(W, W.T)
    return W

def build_mutual_knn_from_D(D_block: np.ndarray, k: int) -> nx.Graph:
    """Utilise ta fonction cmorph.graph.mutual_knn pour construire le graphe (poids exp_median en interne)."""
    # mutual_knn sait déjà pondérer avec weight_mode="exp_median"
    G = mutual_knn(D_block, k=k, weight_mode="exp_median")
    return G

def percolation_scan(G: nx.Graph, num_taus: int = 60) -> pd.DataFrame:
    """
    Balaye des seuils tau dans l'espace des poids d'arêtes (w_ij),
    calcule G(tau) = |LCC|/N et la susceptibilité chi(tau) (hors LCC).
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame({"tau": [], "G": [], "chi": []})

    # Récupération des poids
    ws = np.array([d.get("weight", 0.0) for _, _, d in G.edges(data=True)], dtype=float)
    ws = ws[np.isfinite(ws)]
    if ws.size == 0:
        return pd.DataFrame({"tau": [], "G": [], "chi": []})

    # Grille de seuils : quantiles pour bien couvrir la distribution locale
    qs = np.linspace(0.05, 0.99, num_taus)
    taus = np.quantile(ws, qs)

    N = G.number_of_nodes()
    rows = []
    for tau in taus:
        # filtre arêtes par poids >= tau
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0.0) >= float(tau)])

        comps = [len(c) for c in nx.connected_components(H)]
        if len(comps) == 0:
            Grel = 0.0
            chi = 0.0
        else:
            LCC = max(comps)
            Grel = LCC / N
            # susceptibilité (hors LCC)
            sizes = [s for s in comps if s != LCC]
            if len(sizes) == 0:
                chi = 0.0
            else:
                num = sum(s*s for s in sizes)
                den = sum(s for s in sizes)
                chi = num / den if den > 0 else 0.0
        rows.append({"tau": float(tau), "G": float(Grel), "chi": float(chi)})

    return pd.DataFrame(rows)

def find_tau_c(df_scan: pd.DataFrame) -> dict:
    """Retourne tau_c au pic de chi (avec metriques associées)."""
    if df_scan.empty:
        return {"tau_c": np.nan, "chi_max": np.nan, "G_at_tau_c": np.nan}
    idx = int(np.nanargmax(df_scan["chi"].to_numpy()))
    row = df_scan.iloc[idx]
    return {"tau_c": float(row["tau"]), "chi_max": float(row["chi"]), "G_at_tau_c": float(row["G"])}

def maybe_load_window_starts() -> pd.Series | None:
    """
    Essaie de récupérer les timestamps 'start' de fenêtres pour indexer la série τ_c :
    - artifacts/proxies_window.csv (produit par 06_liquidity_proxies)
    - sinon None (on sortira juste les 'i')
    """
    p = ART / "proxies_window.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=["start"])
            if "start" in df.columns:
                return df["start"]
        except Exception:
            pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["dtw","softdtw","corr","ncc","frechet"], default="softdtw")
    ap.add_argument("--k", type=int, default=15, help="k du graphe k-NN mutuel")
    ap.add_argument("--block", type=int, default=78, help="taille du bloc roulant (en fenêtres) ~ 78 ≈ 1 séance cash US")
    ap.add_argument("--step", type=int, default=1, help="pas entre centres de blocs (en fenêtres)")
    ap.add_argument("--num-taus", type=int, default=60, help="nombre de seuils dans le scan percolation")
    ap.add_argument("--out-prefix", type=str, default="tau_c_per_window", help="préfixe des fichiers de sortie")
    args = ap.parse_args()

    # 1) Charge distance globale
    D = load_distance(args.metric)
    N = D.shape[0]

    # 2) Prépare la liste des centres de bloc
    B = args.block
    if B < 20:
        print("⚠️  block < 20 : percolation risquée. Je recommande ~78 (1 séance) ou ≥60.")
    centers = list(range(B//2, N - (B - B//2) + 1, args.step))  # centre inclus, bloc centré si possible

    # 3) Timestamps 'start' (facultatif pour tracé)
    starts = maybe_load_window_starts()
    has_time = starts is not None and len(starts) == N

    # 4) Boucle rolling
    out_rows = []
    for c in centers:
        i0 = c - B//2
        i1 = i0 + B
        if i0 < 0 or i1 > N:  # garde-fou
            continue
        D_blk = D[i0:i1, i0:i1]

        # Graphe local
        G = build_mutual_knn_from_D(D_blk, k=args.k)
        # Perco
        scan = percolation_scan(G, num_taus=args.num_taus)
        res = find_tau_c(scan)
        row = {
            "i_center": int(c),
            "i0": int(i0),
            "i1": int(i1),
            "tau_c": res["tau_c"],
            "chi_max": res["chi_max"],
            "G_at_tau_c": res["G_at_tau_c"],
            "k": int(args.k),
            "block": int(B),
            "num_taus": int(args.num_taus),
        }
        if has_time:
            row["start"] = pd.to_datetime(starts.iloc[c])
        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values("i_center").reset_index(drop=True)
    if out.empty:
        raise SystemExit("Aucun bloc valide. Vérifie 'block' (trop grand?) vs N (taille de la matrice).")

    # 5) Option : étendre de 'i_center' à tous les i couverts ? (on garde la valeur au centre, c'est le plus propre)
    # Pour avoir un tau_c par fenêtre, on peut faire un forward/backward fill léger :
    tau_full = pd.Series(np.nan, index=np.arange(N), dtype=float)
    for _, r in out.iterrows():
        tau_full.iloc[int(r["i_center"])] = r["tau_c"]
    tau_full = tau_full.interpolate(limit_direction="both")  # lissage optionnel pour visualisation

    # 6) Sauvegardes
    out_csv = ART / f"{args.out_prefix}.csv".replace("-", "_")
    out.to_csv(out_csv, index=False)
    print(f"📄 τ_c (centres de blocs) -> {out_csv}  (nb={len(out)}/{N} centres)")

    # Sauvegarde aussi la série densifiée (même longueur que D)
    dense = pd.DataFrame({
        "i": np.arange(N, dtype=int),
        "tau_c": tau_full.values,
    })
    if has_time:
        dense["start"] = starts.values
    dense_csv = ART / f"{args.out_prefix}_dense.csv".replace("-", "_")
    dense.to_csv(dense_csv, index=False)
    print(f"📄 τ_c densifié (par fenêtre, interp.) -> {dense_csv}")

    # 7) Figure rapide si timestamps disponibles
    try:
        if has_time:
            plt.figure(figsize=(9.5, 3.2))
            plt.plot(dense["start"], dense["tau_c"], lw=1.2)
            plt.scatter(out["start"], out["tau_c"], s=12, alpha=0.7, label="centres")
            plt.title(r"$\tau_c$ (rolling percolation) — bloc={} fen., k={}".format(B, args.k))
            plt.xlabel("time")
            plt.ylabel(r"$\tau_c$")
            plt.legend()
            plt.tight_layout()
            figp = ART / f"{args.out_prefix}.png".replace("-", "_")
            plt.savefig(figp, dpi=170)
            print(f"🖼️  Figure -> {figp}")
    except Exception as e:
        print(f"⚠️  Impossible de tracer la série τ_c : {e}")

if __name__ == "__main__":
    main()
