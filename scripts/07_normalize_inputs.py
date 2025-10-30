# %% scripts/07_normalize_inputs.py
# Standardise inputs for early-warning pipeline.
# Example:
# python -m scripts.07_normalize_inputs \
#   --tau-in artifacts/raw_tau.csv \
#   --proxies-in artifacts/raw_proxies.csv \
#   --time-col-tau start --time-col-proxies start \
#   --index-col-tau i --index-col-proxies i \
#   --tau-out artifacts/tau_c_per_window_dense.csv \
#   --proxies-out artifacts/proxies_window.csv

import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

def ensure_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def pick_col(df, pref, fallbacks):
    if pref and pref in df.columns:
        return pref
    for c in fallbacks:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-in", type=str, required=True)
    ap.add_argument("--proxies-in", type=str, required=True)
    ap.add_argument("--time-col-tau", type=str, default=None)
    ap.add_argument("--time-col-proxies", type=str, default=None)
    ap.add_argument("--index-col-tau", type=str, default=None, help="optional row-order align fallback")
    ap.add_argument("--index-col-proxies", type=str, default=None, help="optional row-order align fallback")
    ap.add_argument("--tau-out", type=str, default=str(ART/"tau_c_per_window_dense.csv"))
    ap.add_argument("--proxies-out", type=str, default=str(ART/"proxies_window.csv"))
    args = ap.parse_args()

    T = pd.read_csv(args.tau_in)
    P = pd.read_csv(args.proxies_in)

    # ---- time columns
    t_tau = pick_col(T, args.time_col_tau, ["start","Timestamp","timestamp","time","datetime","date","ts"])
    t_prx = pick_col(P, args.time_col_proxies, ["start","Timestamp","timestamp","time","datetime","date","ts"])

    if t_tau:
        T[t_tau] = ensure_utc(T[t_tau]); T = T.rename(columns={t_tau:"start"})
    if t_prx:
        P[t_prx] = ensure_utc(P[t_prx]); P = P.rename(columns={t_prx:"start"})

    # ---- tau column
    tau_col = None
    for c in T.columns:
        cl = str(c).lower()
        if cl in ("tau_c","tau","tauc") or cl.startswith("tau"):
            tau_col = c; break
    if tau_col is None:
        raise SystemExit(f"No tau column detected in {args.tau_in}. Columns: {list(T.columns)}")
    if tau_col != "tau_c":
        T = T.rename(columns={tau_col:"tau_c"})

    # ---- proxy columns (keep only known)
    lower = {str(c).lower(): c for c in P.columns}
    keep = []
    for name in ["parkinson","garman_klass","corwin_schultz","roll"]:
        if name in lower: keep.append((lower[name], name))
    if not keep:
        raise SystemExit(f"No expected proxies found in {args.proxies_in}. Columns: {list(P.columns)}")
    P = P.rename(columns=dict(keep))
    cols = ["start"] if "start" in P.columns else []
    P = P[cols + [n for _,n in keep]]

    # ---- If no 'start' in tau and proxies, fallback to order
    if "start" not in T.columns and "start" not in P.columns:
        if args.index_col_tau and args.index_col_tau in T.columns:
            T = T.sort_values(args.index_col_tau).reset_index(drop=True)
        if args.index_col_proxies and args.index_col_proxies in P.columns:
            P = P.sort_values(args.index_col_proxies).reset_index(drop=True)
        if len(T) != len(P):
            raise SystemExit("Order-alignment fallback requires same length.")
        print("⚠️  Falling back to order alignment: copying time from proxies to tau.")
        T["start"] = ensure_utc(P.index.to_series())  # placeholder
        if "start" in P.columns:
            T["start"] = P["start"]
        else:
            # build a dummy UTC timeline at 5-minute step if nothing else is available
            T["start"] = pd.to_datetime(np.arange(len(T)), unit="m", origin="unix", utc=True)

    # if only proxies has start, copy to tau by merge on index
    if "start" not in T.columns and "start" in P.columns:
        T = T.reset_index(drop=True); P = P.reset_index(drop=True)
        if len(T) != len(P):
            raise SystemExit("Cannot borrow 'start' from proxies: different lengths.")
        print("⚠️  Borrowing 'start' from proxies (row order).")
        T["start"] = P["start"]

    # if only tau has start, we keep proxies without time or warn
    if "start" not in P.columns and "start" in T.columns:
        print("⚠️  Proxies have no 'start' column; saving without it.")

    # ---- outputs
    T = T[["start","tau_c"]]
    Path(args.tau_out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(args.tau_out, index=False)
    print(f"✅ Normalized tau     → {args.tau_out}  (cols: {list(T.columns)})")

    if "start" in P.columns:
        P = P[["start"] + [n for _,n in keep]]
    else:
        P = P[[n for _,n in keep]]
    Path(args.proxies_out).parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(args.proxies_out, index=False)
    print(f"✅ Normalized proxies → {args.proxies_out}  (cols: {list(P.columns)})")

if __name__ == "__main__":
    main()
