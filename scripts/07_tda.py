# guard
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import pickle, pandas as pd
from cmorph.tda import tda_features_lower_star

wins = pickle.load(open("artifacts/windows.pkl","rb"))
rows = []
for i, w in enumerate(wins):
    feats = tda_features_lower_star(w)
    feats["i"] = i
    rows.append(feats)

out = pd.DataFrame(rows)
out.to_csv("artifacts/tda.csv", index=False)
print("✅ artifacts/tda.csv écrit — colonnes:", list(out.columns))
print(out.describe())
