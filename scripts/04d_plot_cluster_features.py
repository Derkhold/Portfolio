# --- python -m scripts.04d_plot_cluster_features --labels artifacts/labels_softdtw_K2.csv
import sys, argparse
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
FIGDIR = ART / "figures"; FIGDIR.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ART / "cluster_features.csv"))
    ap.add_argument("--labels",   default=str(ART / "labels_softdtw_K2.csv"))
    ap.add_argument("--out",      default=str(FIGDIR / "fig_5_2f_features_scatter.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    if "alpha_dfa" not in df.columns or "D_higuchi" not in df.columns:
        raise SystemExit("cluster_features.csv doit contenir alpha_dfa et D_higuchi")

    # Essaie de charger les labels (K=2 par défaut)
    labs = None
    try:
        L = pd.read_csv(args.labels)
        labs = (L.iloc[:, -1].astype(int).to_numpy()
                if "label" not in L.columns else L["label"].astype(int).to_numpy())
        if len(labs) != len(df):
            print("⚠️  labels et features de tailles différentes ; on ignore les labels.")
            labs = None
    except Exception:
        print("ℹ️  labels non trouvés ; on colorie en unicolore.")

    x, y = df["alpha_dfa"].to_numpy(), df["D_higuchi"].to_numpy()

    plt.figure(figsize=(6.2, 4.6))
    if labs is None:
        plt.scatter(x, y, s=8, alpha=0.6)
    else:
        for c in np.unique(labs):
            m = labs == c
            plt.scatter(x[m], y[m], s=10, alpha=0.65, label=f"Cluster {int(c)}")
        plt.legend(title="Regime")

    plt.xlabel(r"DFA slope $\alpha_{\mathrm{DFA}}$")
    plt.ylabel(r"Higuchi fractal dimension $D_{H}$")
    plt.title("Figure 5.2f – Morphological feature space (colored by cluster)", loc="left")
    plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(args.out, dpi=200); plt.close()
    print("📈 saved:", args.out)

if __name__ == "__main__":
    main()
