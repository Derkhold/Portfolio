# %% scripts/07d_brier_compare.py
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def brier(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean((y[m] - p[m])**2)) if m.any() else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="artifacts/tau_c_per_window_dense.csv")
    ap.add_argument("--labels", default="artifacts/labels_softdtw_K2.csv")
    ap.add_argument("--h", type=int, default=1, help="horizon (steps ahead)")
    ap.add_argument("--lb", type=int, default=78, help="look-back window for on-the-fly calibration (≈1 session)")
    ap.add_argument("--out", default="artifacts/fig_5_11_brier.png")
    args = ap.parse_args()

    T = pd.read_csv(args.tau); L = pd.read_csv(args.labels)
    # align by index (les deux fichiers produits dans ton pipeline ont même pas/ordre)
    if "label" not in L.columns:
        raise SystemExit("labels file must have column 'label' (0/1 stress).")
    y = L["label"].astype(int).to_numpy()
    tau = T["tau_c"].to_numpy()

    # signal → proba simple (calibration par déciles rolling)
    n=len(y); p_tau = np.full(n, np.nan)
    for t in range(args.lb, n-args.h):
        past = tau[t-args.lb:t]
        # déciles -> mapping proba(y=1|tau in bin) sur la fenêtre passée
        bins = np.nanquantile(past, np.linspace(0,1,11))
        idx = np.digitize(past, bins[1:-1], right=True)
        # taux locaux par bin
        ypast = y[t-args.lb:t]
        pr = np.array([np.nanmean(ypast[idx==k]) for k in range(10)])
        # proba pour l’instant t (basé sur tau[t])
        kt = np.digitize([tau[t]], bins[1:-1], right=True)[0]
        p_tau[t+args.h] = pr[kt] if np.isfinite(pr[kt]) else np.nan

    # baselines
    p_uncond = np.nanmean(y[:args.lb]) * np.ones(n)
    p_persist = np.r_[np.nan, y[:-1]]  # persistance naïve

    # Brier sur la partie valide (où p_tau est dispo)
    mask = np.isfinite(p_tau) & np.isfinite(y)
    b_tau = brier(y[mask], p_tau[mask])
    b_un  = brier(y[mask], p_uncond[mask])
    b_per = brier(y[mask], p_persist[mask])

    # plot
    vals = [b_tau, b_un, b_per]; names = ["τc (calibrated)", "unconditional", "persistence"]
    plt.figure(figsize=(5.2,4))
    plt.bar(range(3), vals, color=["C0","C2","C1"])
    plt.xticks(range(3), names, rotation=10); plt.ylabel("Brier score (lower is better)")
    plt.title(f"Brier comparison (h={args.h}, look-back={args.lb})")
    for i,v in enumerate(vals):
        plt.text(i, v+0.001, f"{v:.3f}", ha="center", va="bottom")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=180)
    print("🖼️  saved:", args.out)

if __name__ == "__main__":
    main()
