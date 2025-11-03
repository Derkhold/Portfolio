# scripts/04e_plot_cluster_features.py
# Scatter of DFA vs Higuchi features, optionally colored by cluster labels

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ART = Path("artifacts")
ART.mkdir(exist_ok=True)
FIGDIR = ART / "figures"
FIGDIR.mkdir(exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ART / "cluster_features.csv"))
    ap.add_argument("--labels", default=str(ART / "labels_softdtw_K2.csv"))
    ap.add_argument("--out", default=str(FIGDIR / "fig_5_2f_features_scatter.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    need = {"alpha_dfa", "D_higuchi"}
    if not need.issubset(df.columns):
        raise SystemExit("cluster_features.csv must contain columns: alpha_dfa, D_higuchi")

    # optional labels
    labels = None
    try:
        L = pd.read_csv(args.labels)
        labels = (L["label"].astype(int).to_numpy()
                  if "label" in L.columns else L.iloc[:, -1].astype(int).to_numpy())
        if len(labels) != len(df):
            print("Labels length != features length — ignoring labels.")
            labels = None
    except Exception:
        print("Labels not found — plotting in single color.")

    x = df["alpha_dfa"].to_numpy()
    y = df["D_higuchi"].to_numpy()

    plt.figure(figsize=(6.2, 4.6))
    if labels is None:
        plt.scatter(x, y, s=8, alpha=0.6)
    else:
        for c in np.unique(labels):
            m = labels == c
            plt.scatter(x[m], y[m], s=10, alpha=0.65, label=f"Cluster {int(c)}")
        plt.legend(title="Regime")

    plt.xlabel(r"DFA slope $\alpha_{\mathrm{DFA}}$")
    plt.ylabel(r"Higuchi fractal dimension $D_{H}$")
    plt.title("Figure 5.2f – Morphological feature space", loc="left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
