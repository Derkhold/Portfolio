# scripts/05c_tau_c_per_window.py
# Usage:
#   python -m scripts.05c_tau_c_per_window --metric softdtw --k 15 --block 78 --step 1
#
# Outputs:
#   artifacts/tau_c_per_window.csv        (columns: i_center, i0, i1, tau_c, chi_max, G_at_tau_c, k, block, num_taus, [start])
#   artifacts/tau_c_per_window_dense.csv  (columns: i, tau_c, [start])  — interpolated to one value per window
#   artifacts/tau_c_per_window.png        (time series plot if 'start' timestamps are available)

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.graph import mutual_knn  # mutual k-NN backbone; weights computed internally

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


# -------------------- I/O --------------------
def load_distance(metric: str) -> np.ndarray:
    p = ART / f"D_{metric}.npy"
    if not p.exists():
        raise SystemExit(f"Distance not found: {p}. Run 03* scripts first to compute {metric}.")
    D = np.load(p)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise SystemExit(f"Invalid distance matrix: {p} (shape={D.shape})")
    return D


def maybe_load_window_starts() -> pd.Series | None:
    """
    Try to load per-window start timestamps for plotting:
      - artifacts/proxies_window.csv must contain a 'start' column.
      - Returns None if missing/unreadable.
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


# -------------------- Graph & percolation --------------------
def build_mutual_knn_from_D(D_block: np.ndarray, k: int) -> nx.Graph:
    """Build mutual k-NN graph using cmorph.graph.mutual_knn (weight_mode='exp_median')."""
    return mutual_knn(D_block, k=k, weight_mode="exp_median")


def percolation_scan(G: nx.Graph, num_taus: int = 60) -> pd.DataFrame:
    """
    Scan thresholds τ over edge weights w_ij and compute:
      - G(τ): relative size of the largest component
      - χ(τ): susceptibility proxy on non-giant components
    τ grid is chosen via weight quantiles to adapt to local distribution.
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame({"tau": [], "G": [], "chi": []})

    ws = np.array([d.get("weight", 0.0) for *_ , d in G.edges(data=True)], dtype=float)
    ws = ws[np.isfinite(ws)]
    if ws.size == 0:
        return pd.DataFrame({"tau": [], "G": [], "chi": []})

    qs = np.linspace(0.05, 0.99, num_taus)
    taus = np.quantile(ws, qs)

    N = G.number_of_nodes()
    rows = []
    for tau in taus:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(
            (u, v, d) for u, v, d in G.edges(data=True)
            if float(d.get("weight", 0.0)) >= float(tau)
        )
        comps = [len(c) for c in nx.connected_components(H)]
        if not comps:
            rows.append({"tau": float(tau), "G": 0.0, "chi": 0.0})
            continue

        L = max(comps)
        Grel = L / N
        rest = [s for s in comps if s != L]
        if not rest:
            chi = 0.0
        else:
            num = sum(s * s for s in rest)
            den = sum(rest)
            chi = num / den if den > 0 else 0.0
        rows.append({"tau": float(tau), "G": float(Grel), "chi": float(chi)})

    return pd.DataFrame(rows)


def find_tau_c(df_scan: pd.DataFrame) -> dict:
    """Return τ_c at the χ peak, with associated χ_max and G(τ_c)."""
    if df_scan.empty:
        return {"tau_c": np.nan, "chi_max": np.nan, "G_at_tau_c": np.nan}
    idx = int(np.nanargmax(df_scan["chi"].to_numpy()))
    row = df_scan.iloc[idx]
    return {"tau_c": float(row["tau"]), "chi_max": float(row["chi"]), "G_at_tau_c": float(row["G"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["dtw", "softdtw", "corr", "ncc", "frechet"], default="softdtw")
    ap.add_argument("--k", type=int, default=15, help="mutual k-NN parameter")
    ap.add_argument("--block", type=int, default=78, help="rolling block length in windows (~78 ≈ one cash session)")
    ap.add_argument("--step", type=int, default=1, help="block step in windows")
    ap.add_argument("--num-taus", type=int, default=60, help="number of thresholds in the percolation scan")
    ap.add_argument("--out-prefix", type=str, default="tau_c_per_window", help="output file prefix")
    args = ap.parse_args()

    # 1) Distance and basic geometry
    D = load_distance(args.metric)
    N = D.shape[0]

    # 2) Rolling centers (keep block roughly centered on the index)
    B = args.block
    if B < 20:
        print("Warning: block < 20; percolation may be noisy. Prefer ~78 or ≥60.")
    centers = list(range(B // 2, N - (B - B // 2) + 1, args.step))

    # 3) Optional timestamps for plotting
    starts = maybe_load_window_starts()
    has_time = starts is not None and len(starts) == N

    # 4) Rolling percolation
    out_rows = []
    for c in centers:
        i0 = c - B // 2
        i1 = i0 + B
        if i0 < 0 or i1 > N:
            continue

        D_blk = D[i0:i1, i0:i1]
        G = build_mutual_knn_from_D(D_blk, k=args.k)

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
        raise SystemExit("No valid blocks. Check 'block' vs matrix size N.")

    # 5) Optional dense series: interpolate τ_c to one value per window index
    tau_full = pd.Series(np.nan, index=np.arange(N), dtype=float)
    for _, r in out.iterrows():
        tau_full.iloc[int(r["i_center"])] = r["tau_c"]
    tau_full = tau_full.interpolate(limit_direction="both")

    # 6) Save results
    out_csv = ART / f"{args.out_prefix}.csv".replace("-", "_")
    out.to_csv(out_csv, index=False)
    print(f"Wrote centers CSV: {out_csv}  (count={len(out)}/{N})")

    dense = pd.DataFrame({"i": np.arange(N, dtype=int), "tau_c": tau_full.values})
    if has_time:
        dense["start"] = starts.values
    dense_csv = ART / f"{args.out_prefix}_dense.csv".replace("-", "_")
    dense.to_csv(dense_csv, index=False)
    print(f"Wrote dense CSV: {dense_csv}")

    # 7) Plot (if timestamps available)
    if has_time:
        figp = ART / f"{args.out_prefix}.png".replace("-", "_")
        plt.figure(figsize=(9.5, 3.2))
        plt.plot(dense["start"], dense["tau_c"], lw=1.2)
        plt.scatter(out["start"], out["tau_c"], s=12, alpha=0.7, label="centers")
        plt.title(r"Rolling $\tau_c$ (block={} windows, k={})".format(B, args.k))
        plt.xlabel("time")
        plt.ylabel(r"$\tau_c$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figp, dpi=170)
        print(f"Wrote figure: {figp}")


if __name__ == "__main__":
    main()
