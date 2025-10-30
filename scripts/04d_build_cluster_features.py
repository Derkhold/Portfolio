# --- python -m scripts.04d_build_cluster_features --labels artifacts/labels_softdtw_K2.csv
import sys, argparse, pickle
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

# tes fonctions de rugosité
from cmorph.roughness import dfa_alpha, higuchi_fd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ART / "labels_softdtw_K2.csv"))
    ap.add_argument("--wins",   default=str(ART / "windows.pkl"))
    ap.add_argument("--out",    default=str(ART / "cluster_features.csv"))
    args = ap.parse_args()

    # 1) charge les fenêtres normalisées
    wins = pickle.load(open(args.wins, "rb"))
    X = np.stack(wins)  # (N, L)

    # 2) calcule les features
    rows = []
    for i, w in enumerate(X):
        rows.append({
            "i": i,
            "alpha_dfa": float(dfa_alpha(w)),
            "D_higuchi": float(higuchi_fd(w)),
        })
    df = pd.DataFrame(rows)

    # 3) ajoute les labels si fournis
    try:
        lab = pd.read_csv(args.labels)
        if "label" not in lab.columns:
            lab = lab.rename(columns={lab.columns[-1]: "label"})
        lab = lab[["label"]].reset_index(drop=True)
        if len(lab) == len(df):
            df["label"] = lab["label"].astype(int).values
        else:
            print(f"⚠️ labels({len(lab)}) != features({len(df)}) — on n’ajoute pas les labels.")
    except Exception as e:
        print("ℹ️ labels non trouvés / non lus :", e)

    df.to_csv(args.out, index=False)
    print("✅ écrit:", args.out, "colonnes:", list(df.columns))

if __name__ == "__main__":
    main()

