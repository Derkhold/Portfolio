# %% scripts/07b_ew_heatmap.py
# Build heatmaps: OR (centered at 1.0, diverging palette) and -log10(p).
# Input = CSV produced by 07a (artifacts/<out-prefix>.csv)
#
# Example:
# python -m scripts.07b_ew_heatmap \
#   --metrics artifacts/ew_q20_lowtail_eventHigh.csv \
#   --out artifacts/ew_heatmap_q20_lowtail_eventHigh.png

import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True)
    ap.add_argument("--out", type=str, default="artifacts/ew_heatmap.png")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    needed = {"proxy","lag","odds_ratio","fisher_p","fisher_p_holm"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in metrics: {missing}")

    # choose Holm-adjusted p if present
    p_use = df["fisher_p_holm"].copy()
    if p_use.isna().all(): p_use = df["fisher_p"]
    df = df.assign(p_use=p_use)

    order = ["roll","corwin_schultz","garman_klass","parkinson"]
    proxies = [p for p in order if p in df["proxy"].unique()] or sorted(df["proxy"].unique())
    lags = sorted(df["lag"].unique())

    OR = np.full((len(proxies), len(lags)), np.nan)
    LOGP = np.full_like(OR, np.nan, dtype=float)

    for i, pr in enumerate(proxies):
        for j, lg in enumerate(lags):
            sub = df[(df["proxy"]==pr) & (df["lag"]==lg)]
            if len(sub):
                OR[i,j] = float(sub.iloc[0]["odds_ratio"])
                pv = float(sub.iloc[0]["p_use"])
                LOGP[i,j] = -np.log10(max(pv, 1e-16)) if np.isfinite(pv) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    # OR heatmap (centered at 1.0)
    ax = axes[0]
    norm = TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.6)
    im = ax.imshow(OR, aspect="auto", origin="lower", norm=norm, cmap="RdBu_r")
    ax.set_xticks(range(len(lags))); ax.set_xticklabels(lags)
    ax.set_yticks(range(len(proxies))); ax.set_yticklabels([p.replace("_"," ") for p in proxies])
    ax.set_title("Odds ratio (S→proxy, by lag)")
    ax.set_xlabel("lag (steps)"); ax.set_ylabel("proxy")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("OR")

    # Significance stars over OR panel
    for i, pr in enumerate(proxies):
        for j, lg in enumerate(lags):
            pv = df[(df["proxy"]==pr) & (df["lag"]==lg)]["p_use"]
            if len(pv):
                pval = float(pv.iloc[0])
                star = "***" if pval <= 1e-3 else "**" if pval <= 1e-2 else "*" if pval <= 5e-2 else ""
                if star:
                    ax.text(j, i, star, ha="center", va="center", fontsize=11, color="black")

    # -log10(p) heatmap
    ax2 = axes[1]
    vmax = np.nanmax(LOGP) if np.isfinite(LOGP).any() else 3
    im2 = ax2.imshow(LOGP, aspect="auto", origin="lower", vmin=0, vmax=max(3, vmax), cmap="viridis")
    ax2.set_xticks(range(len(lags))); ax2.set_xticklabels(lags)
    ax2.set_yticks(range(len(proxies))); ax2.set_yticklabels([p.replace("_"," ") for p in proxies])
    ax2.set_title(r"$-\log_{10}(p)$ (Holm-adjusted if available)")
    ax2.set_xlabel("lag (steps)"); ax2.set_ylabel("proxy")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04); cbar2.set_label(r"$-\log_{10}(p)$")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print("🖼️  saved:", out)

if __name__ == "__main__":
    main()
