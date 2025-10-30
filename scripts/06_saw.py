# guard
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import pickle, pandas as pd
from cmorph.saw import saw_score

wins = pickle.load(open("artifacts/windows.pkl","rb"))

rows = []
for i, w in enumerate(wins):
    s = saw_score(w, bins=20, eps=None)
    s["i"] = i
    rows.append(s)

out = pd.DataFrame(rows)
out.to_csv("artifacts/saw.csv", index=False)
print("✅ artifacts/saw.csv écrit — colonnes: i, S, rho, I")
print(out.describe())
