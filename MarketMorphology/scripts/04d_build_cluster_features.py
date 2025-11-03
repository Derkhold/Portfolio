# scripts/04d_build_cluster_features.py
# Compute DFA/Higuchi features per window and attach cluster labels

import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.roughness import dfa_alpha, higuchi_fd

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ART / "labels_softdtw_K2.csv"))
    ap.add_argument("--wins", default=str(ART / "windows.pkl"))
    ap.add_argument("--out", default=str(ART / "cluster_features.csv"))
    args = ap.parse_args()

    # 1) Load normalized windows
    wins = pickle.load(open(args.wins, "rb"))
    X = np.stack(wins)

    # 2) Compute roughness features
    rows = [
        {
            "i": i,
            "alpha_dfa": float(dfa_alpha(w)),
            "D_higuchi": float(higuchi_fd(w)),
        }
        for i, w in enumerate(X)
    ]
    df = pd.DataFrame(rows)

    # 3) Add labels if provided
    try:
        lab = pd.read_csv(args.labels)
        if "label" not in lab.columns:
            lab = lab.rename(columns={lab.columns[-1]: "label"})
        lab = lab[["label"]].reset_index(drop=True)
        if len(lab) == len(df):
            df["label"] = lab["label"].astype(int).values
        else:
            print(f"label count ({len(lab)}) != features ({len(df)}) — skipping label merge.")
    except Exception as e:
        print("Labels not found or unreadable:", e)

    # 4) Save
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} | Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
