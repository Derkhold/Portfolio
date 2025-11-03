# scripts/07_normalize_inputs.py
# Standardize inputs for the early-warning pipeline.
#
# Examples:
#   python -m scripts.07_normalize_inputs \
#     --tau-in artifacts/raw_tau.csv \
#     --proxies-in artifacts/raw_proxies.csv \
#     --time-col-tau start --time-col-proxies start \
#     --index-col-tau i --index-col-proxies i \
#     --tau-out artifacts/tau_c_per_window_dense.csv \
#     --proxies-out artifacts/proxies_window.csv

import sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_utc(s: pd.Series) -> pd.Series:
    """Coerce a series to timezone-aware UTC datetimes (tolerant to strings/naive)."""
    return pd.to_datetime(s, errors="coerce", utc=True)

def pick_col(df: pd.DataFrame, preferred: str | None, fallbacks: list[str]) -> str | None:
    """
    Pick a column name from df:
      1) return `preferred` if provided and present,
      2) otherwise return the first present in `fallbacks`,
      3) else None.
    """
    if preferred and preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    return None

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Normalize tau_c and liquidity proxies to a consistent schema."
    )
    ap.add_argument("--tau-in", type=str, required=True, help="Input CSV for tau series.")
    ap.add_argument("--proxies-in", type=str, required=True, help="Input CSV for window-level proxies.")
    ap.add_argument("--time-col-tau", type=str, default=None, help="Optional time column name in tau CSV.")
    ap.add_argument("--time-col-proxies", type=str, default=None, help="Optional time column name in proxies CSV.")
    ap.add_argument("--index-col-tau", type=str, default=None, help="Optional index column for row-order fallback.")
    ap.add_argument("--index-col-proxies", type=str, default=None, help="Optional index column for row-order fallback.")
    ap.add_argument("--tau-out", type=str, default=str(ART / "tau_c_per_window_dense.csv"),
                    help="Output CSV path for normalized tau.")
    ap.add_argument("--proxies-out", type=str, default=str(ART / "proxies_window.csv"),
                    help="Output CSV path for normalized proxies.")
    args = ap.parse_args()

    T = pd.read_csv(args.tau_in)
    P = pd.read_csv(args.proxies_in)

    # ---- Resolve time columns
    t_tau = pick_col(T, args.time_col_tau,
                     ["start","Timestamp","timestamp","time","datetime","date","ts"])
    t_prx = pick_col(P, args.time_col_proxies,
                     ["start","Timestamp","timestamp","time","datetime","date","ts"])

    if t_tau:
        T[t_tau] = ensure_utc(T[t_tau])
        T = T.rename(columns={t_tau: "start"})
    if t_prx:
        P[t_prx] = ensure_utc(P[t_prx])
        P = P.rename(columns={t_prx: "start"})

    # ---- Resolve tau column
    tau_col = None
    for c in T.columns:
        cl = str(c).lower()
        if cl in ("tau_c", "tau", "tauc") or cl.startswith("tau"):
            tau_col = c
            break
    if tau_col is None:
        raise SystemExit(f"No tau column detected in {args.tau_in}. Columns: {list(T.columns)}")
    if tau_col != "tau_c":
        T = T.rename(columns={tau_col: "tau_c"})

    # ---- Keep only known proxy columns (and start if present)
    lower = {str(c).lower(): c for c in P.columns}
    keep_pairs: list[tuple[str, str]] = []
    for name in ["parkinson", "garman_klass", "corwin_schultz", "roll"]:
        if name in lower:
            keep_pairs.append((lower[name], name))
    if not keep_pairs:
        raise SystemExit(f"No expected proxies found in {args.proxies_in}. Columns: {list(P.columns)}")
    P = P.rename(columns=dict(keep_pairs))
    ordered = (["start"] if "start" in P.columns else []) + [new for _, new in keep_pairs]
    P = P[ordered]

    # ---- If neither file has 'start', fall back to row order
    if "start" not in T.columns and "start" not in P.columns:
        if args.index_col_tau and args.index_col_tau in T.columns:
            T = T.sort_values(args.index_col_tau).reset_index(drop=True)
        if args.index_col_proxies and args.index_col_proxies in P.columns:
            P = P.sort_values(args.index_col_proxies).reset_index(drop=True)
        if len(T) != len(P):
            raise SystemExit("Order-alignment fallback requires the same number of rows in tau and proxies.")
        print("Warning: Falling back to order alignment; synthesizing a time axis.")
        # Prefer copying a real 'start' if proxies has it; else synthesize a dummy UTC timeline.
        if "start" in P.columns:
            T["start"] = ensure_utc(P["start"])
        else:
            T["start"] = pd.to_datetime(np.arange(len(T)), unit="m", origin="unix", utc=True)

    # ---- If only proxies has 'start', borrow it by row order (lengths must match)
    if "start" not in T.columns and "start" in P.columns:
        T = T.reset_index(drop=True)
        P = P.reset_index(drop=True)
        if len(T) != len(P):
            raise SystemExit("Cannot borrow 'start' from proxies: different lengths.")
        print("Warning: Borrowing 'start' from proxies using row order.")
        T["start"] = ensure_utc(P["start"])

    # ---- If only tau has 'start', proceed; proxies will be saved without time (with a warning)
    if "start" not in P.columns and "start" in T.columns:
        print("Warning: Proxies have no 'start' column; saving without it.")

    # ---- Outputs
    T = T[["start", "tau_c"]] if "start" in T.columns else T[["tau_c"]]
    Path(args.tau_out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(args.tau_out, index=False)
    print(f"Wrote normalized tau to {args.tau_out}  (cols: {list(T.columns)})")

    if "start" in P.columns:
        P = P[["start"] + [new for _, new in keep_pairs]]
    else:
        P = P[[new for _, new in keep_pairs]]
    Path(args.proxies_out).parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(args.proxies_out, index=False)
    print(f"Wrote normalized proxies to {args.proxies_out}  (cols: {list(P.columns)})")

if __name__ == "__main__":
    main()
