# scripts/07b_ew_heatmap.py
# Build side-by-side heatmaps:
#   (1) Odds ratio (centered at 1.0; diverging colormap)
#   (2) -log10(p) using Holm-adjusted p if available
#
# Input: CSV produced by 07a_early_warning.py
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
    ap = argparse.ArgumentParser(
        description="Render heatmaps for early-warning metrics: OR and -log10(p)."
    )
    ap.add_argument("--metrics", type=str, required=True,
                    help="CSV produced by 07a_early_warning (must include proxy, lag, odds_ratio, fisher_p, fisher_p_holm).")
    ap.add_argument("--out", type=str, default="artifacts/ew_heatmap.png",
                    help="Output PNG path.")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    required = {"proxy", "lag", "odds_ratio", "fisher_p", "fisher_p_holm"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in metrics file: {missing}")

    # Choose Holm-adjusted p if any are present; otherwise, use raw Fisher p.
    p_use = df["fisher_p_holm"].copy()
    if p_use.isna().all():
        p_use = df["fisher_p"]
    df = df.assign(p_use=p_use)

    # Stable proxy ordering (roll → CS → GK → Parkinson) if available, else alphabetical.
    preferred = ["roll", "corwin_schultz", "garman_klass", "parkinson"]
    proxies = [p for p in preferred if p in df["proxy"].unique()]
    if not proxies:
        proxies = sorted(df["proxy"].unique())

    lags = sorted(df["lag"].unique())

    # Build matrices
    OR = np.full((len(proxies), len(lags)), np.nan, dtype=float)
    LOGP = np.full_like(OR, np.nan)

    # Fill matrices
    for i, pr in enumerate(proxies):
        for j, lg in enumerate(lags):
            sub = df[(df["proxy"] == pr) & (df["lag"] == lg)]
            if len(sub):
                OR[i, j] = float(sub.iloc[0]["odds_ratio"])
                pv = float(sub.iloc[0]["p_use"])
                LOGP[i, j] = -np.log10(max(pv, 1e-16)) if np.isfinite(pv) else np.nan

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    # (1) OR heatmap (diverging, centered at 1.0)
    ax1 = axes[0]
    norm = TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.6)
    im1 = ax1.imshow(OR, aspect="auto", origin="lower", norm=norm, cmap="RdBu_r")
    ax1.set_xticks(range(len(lags))); ax1.set_xticklabels(lags)
    ax1.set_yticks(range(len(proxies))); ax1.set_yticklabels([p.replace("_", " ") for p in proxies])
    ax1.set_title("Odds ratio (S → proxy, by lag)")
    ax1.set_xlabel("lag (windows)")
    ax1.set_ylabel("proxy")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("OR")

    # Overlay significance stars on the OR panel
    for i, pr in enumerate(proxies):
        for j, lg in enumerate(lags):
            pv_series = df[(df["proxy"] == pr) & (df["lag"] == lg)]["p_use"]
            if len(pv_series):
                pval = float(pv_series.iloc[0])
                if pval <= 1e-3:
                    star = "***"
                elif pval <= 1e-2:
                    star = "**"
                elif pval <= 5e-2:
                    star = "*"
                else:
                    star = ""
                if star:
                    ax1.text(j, i, star, ha="center", va="center", fontsize=11, color="black")

    # (2) -log10(p) heatmap
    ax2 = axes[1]
    vmax = np.nanmax(LOGP) if np.isfinite(LOGP).any() else 3.0
    im2 = ax2.imshow(LOGP, aspect="auto", origin="lower", vmin=0.0, vmax=max(3.0, float(vmax)), cmap="viridis")
    ax2.set_xticks(range(len(lags))); ax2.set_xticklabels(lags)
    ax2.set_yticks(range(len(proxies))); ax2.set_yticklabels([p.replace("_", " ") for p in proxies])
    ax2.set_title(r"$-\log_{10}(p)$ (Holm-adjusted if available)")
    ax2.set_xlabel("lag (windows)")
    ax2.set_ylabel("proxy")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"$-\log_{10}(p)$")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print(f"Saved heatmaps to {out}")

if __name__ == "__main__":
    main()
