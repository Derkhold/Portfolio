# --- python -m scripts.05b_perc_stability --metric softdtw --k 8 \
#     --block 60 --step 20 --taus "0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97,0.99" \
#     --tau-mode weight --tag k8_tau60_99 --ylim 0.6,1.0 --export-scans 3
import sys, argparse, pickle, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from cmorph.graph import mutual_knn

try:
    from cmorph.percolation import scan as perc_scan   # si dispo
except Exception:
    perc_scan = None

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

# ---------- utils parsing ----------
def parse_taus(arg: str | int | None, n_default: int = 50) -> np.ndarray:
    """Accepte entier (linspace) ou liste '0.6,0.7,...'."""
    if arg is None:
        return np.linspace(0.0, 1.0, n_default)
    s = str(arg).strip()
    if "," in s:
        vals = [float(x) for x in s.split(",") if x.strip()]
        return np.array(vals, float)
    return np.linspace(0.0, 1.0, int(float(s)))

# ---------- local percolation (fallback) ----------
def _weights_array(G: nx.Graph) -> np.ndarray:
    return np.array([float(d.get("weight", 0.0)) for *_, d in G.edges(data=True)], float)

def _subgraph_at_weight_threshold(G: nx.Graph, thr: float) -> nx.Graph:
    H = nx.Graph(); H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if float(d.get("weight", 0.0)) >= thr:
            H.add_edge(u, v, **d)
    return H

def _subgraph_keep_top_quantile(G: nx.Graph, q: float) -> nx.Graph:
    """Garde la fraction q des arêtes les plus fortes (0<q<=1)."""
    w = _weights_array(G)
    if w.size == 0:
        H = nx.Graph(); H.add_nodes_from(G.nodes()); return H
    q = float(np.clip(q, 0.0, 1.0))
    thr = np.quantile(w, 1.0 - q)  # garder top-q => poids >= thr
    return _subgraph_at_weight_threshold(G, thr)

def _susceptibility_from_sizes(sizes, N):
    if len(sizes) == 0:
        return 0.0
    s_sorted = sorted(sizes, reverse=True)
    non_giant = s_sorted[1:] if len(s_sorted) > 1 else []
    if not non_giant:
        return 0.0
    return float(np.sum(np.square(non_giant)) / max(N, 1))

def local_perc_scan(G: nx.Graph, taus: np.ndarray, mode: str):
    """mode='weight' => taus=seuils de poids ; mode='quantile' => taus=fractions gardées."""
    N = G.number_of_nodes()
    Gvals, Chivals = [], []
    for t in taus:
        if mode == "quantile":
            H = _subgraph_keep_top_quantile(G, float(t))
        else:
            H = _subgraph_at_weight_threshold(G, float(t))
        comps = [len(c) for c in nx.connected_components(H)]
        lcc = max(comps) if comps else 0
        Gvals.append(lcc / max(N, 1))
        Chivals.append(_susceptibility_from_sizes(comps, N))
    return np.asarray(taus, float), np.asarray(Gvals, float), np.asarray(Chivals, float)

# ---------- coercition percolation externe ----------
def coerce_scan_to_df(res, taus_hint=None, debug=False) -> pd.DataFrame:
    t = G = chi = None
    def _as1d(x):
        if x is None: return None
        a = np.asarray(x, dtype=float).ravel()
        return a if a.size else None
    def _looks_like_tau(a):
        return a is not None and a.size > 1 and a.min() >= -1e-9 and a.max() <= 1+1e-9

    if isinstance(res, dict):
        for k,v in res.items():
            lk = str(k).lower()
            if ("tau" in lk) and t is None: t = _as1d(v)
            elif (lk in ("g","g_tau") or "giant" in lk or "lcc" in lk) and G is None: G = _as1d(v)
            elif ("chi" in lk or "suscep" in lk) and chi is None: chi = _as1d(v)
    elif isinstance(res, (tuple, list)):
        parts = [_as1d(p) for p in res if isinstance(p, (list, tuple, np.ndarray))]
        parts = [p for p in parts if p is not None and p.size > 0]
        if len(parts) >= 2:
            cand_tau = [p for p in parts if _looks_like_tau(p)]
            if cand_tau:
                t = cand_tau[0]
                others = [p for p in parts if p is not t]
                idxG = int(np.argmin([abs(o.min()) + abs(o.max() - 1) for o in others]))
                G = others[idxG]
                rem = [o for i,o in enumerate(others) if i != idxG]
                chi = rem[0] if rem else None
            else:
                inc = [np.mean(np.diff(p) >= -1e-12) for p in parts]
                t = parts[int(np.argmax(inc))]
                others = [p for p in parts if p is not t]
                idxG = int(np.argmin([abs(o.min()) + abs(o.max() - 1) for o in others]))
                G = others[idxG]
                rem = [o for i,o in enumerate(others) if i != idxG]
                chi = rem[0] if rem else None
    else:
        for attr in ("tau","taus"):
            if hasattr(res, attr): t = _as1d(getattr(res, attr)); break
        for attr in ("G","G_tau","giant","lcc"):
            if hasattr(res, attr): G = _as1d(getattr(res, attr)); break
        for attr in ("chi","susceptibility"):
            if hasattr(res, attr): chi = _as1d(getattr(res, attr)); break

    if t is None and taus_hint is not None:
        t = _as1d(taus_hint)
    if t is None and G is not None:
        t = np.linspace(0.0, 1.0, len(G))
    if G is None:
        raise RuntimeError("percolation.scan: cannot coerce (G manquant)")

    if chi is None or len(chi) == 0:
        dt = np.gradient(t)
        dG = np.gradient(G, dt)
        chi = -(dG - dG.min())

    m = int(min(len(t), len(G), len(chi)))
    t, G, chi = t[:m], G[:m], chi[:m]
    return pd.DataFrame({"tau": t, "G": G, "chi": chi})

def find_tau_c(df: pd.DataFrame) -> float:
    i = int(np.nanargmax(df["chi"].to_numpy()))
    return float(df["tau"].iloc[i])

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", type=str, default="softdtw")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--block", type=int, default=60)
    ap.add_argument("--step",  type=int, default=20)
    ap.add_argument("--taus",  type=str, default="50", help='Entier (linspace) ou liste "0.6,0.7,..."')
    ap.add_argument("--tau-mode", choices=["weight","quantile"], default="weight",
                    help="Interprétation de τ : 'weight' = seuil de poids, 'quantile' = fraction d’arêtes gardées")
    ap.add_argument("--tag",   type=str, default="", help="suffixe des fichiers de sortie")
    ap.add_argument("--ylim",  type=str, default="", help='p.ex. "0.6,1.0"')
    ap.add_argument("--export-scans", type=int, default=0, help="sauver les courbes (tau,G,chi) pour les N premiers blocs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    dist_path = ART / f"D_{args.metric}.npy"
    if not dist_path.exists():
        alt = ART / "D_softdtw.npy"
        if args.metric == "softdtw" and alt.exists():
            dist_path = alt
        else:
            raise SystemExit(f"Distance matrix not found: {dist_path}")

    D = np.load(dist_path)
    N = D.shape[0]
    taus = parse_taus(args.taus, n_default=50)

    rows = []
    exported = 0
    for start in range(0, max(1, N - args.block + 1), args.step):
        end = min(N, start + args.block)
        idx = np.arange(start, end)

        Dsub = D[np.ix_(idx, idx)]
        G = mutual_knn(Dsub, k=args.k, weight_mode="exp_median")

        used_fallback = False
        if perc_scan is not None and args.tau_mode == "weight":
            # l’implémentation externe ne connaît pas le mode 'quantile'
            try:
                res = perc_scan(G, taus=taus)
                dfscan = coerce_scan_to_df(res, taus_hint=taus, debug=args.debug)
            except Exception as e:
                if args.debug:
                    print("perc_scan failed:", repr(e))
                used_fallback = True
        else:
            used_fallback = True

        if used_fallback:
            t, Gvals, Chivals = local_perc_scan(G, taus, mode=args.tau_mode)
            dfscan = pd.DataFrame({"tau": t, "G": Gvals, "chi": Chivals})

        tau_c = find_tau_c(dfscan)
        rows.append({
            "start": int(start),
            "end": int(end),
            "center": float(0.5 * (start + end)),
            "tau_c": tau_c,
            "taus_used": ";".join(f"{x:.4f}" for x in taus),
            "tau_mode": args.tau_mode,
            "k": args.k
        })

        # export des courbes pour les premiers blocs
        if exported < args.export_scans:
            tag = f"_{args.tag}" if args.tag else ""
            base = ART / f"perco_scan_{start}_{end}{tag}"
            dfscan.to_csv(f"{base}.csv", index=False)
            fig, ax1 = plt.subplots(figsize=(6.4,3.4))
            ax1.plot(dfscan["tau"], dfscan["G"], marker="o", label="G(τ)")
            ax1.set_xlabel("τ" + (" (quantile gardé)" if args.tau_mode=="quantile" else " (seuil de poids)"))
            ax1.set_ylabel("G(τ)")
            ax2 = ax1.twinx()
            ax2.plot(dfscan["tau"], dfscan["chi"], marker="s", alpha=0.7, label="χ(τ)", linestyle="--")
            ax2.set_ylabel("χ(τ)")
            ax1.axvline(tau_c, color="r", linestyle=":", label=r"$\tau_c$")
            ax1.set_title(f"Percolation bloc [{start},{end}]  (k={args.k}, mode={args.tau_mode})")
            fig.tight_layout(); fig.savefig(f"{base}.png", dpi=170)
            exported += 1

    out = pd.DataFrame(rows)
    tag = f"_{args.tag}" if args.tag else ""
    out_csv = ART / f"tau_c_rolling{tag}.csv"
    out_png = ART / f"tau_c_rolling{tag}.png"
    out.to_csv(out_csv, index=False)

    plt.figure(figsize=(6.6, 3.6))
    plt.plot(out["center"], out["tau_c"], marker="o", linewidth=1.6)
    plt.xlabel("index central du bloc"); plt.ylabel(r"$\tau_c$")
    plt.title(r"Série temporelle de $\tau_c$ (rolling)" + (f" — mode={args.tau_mode}" if args.tau_mode=="quantile" else ""))
    if args.ylim:
        lo, hi = [float(x) for x in args.ylim.split(",")]
        plt.ylim(lo, hi)
    plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out_png, dpi=180)
    print("📈 saved:", out_png, "and", out_csv)
