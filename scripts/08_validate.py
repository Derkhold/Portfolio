# guard
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from cmorph.validate import realized_vol_from_prices, max_drawdown

# charge artefacts
wins_price = pickle.load(open("artifacts/windows_price.pkl","rb"))
rough = pd.read_csv("artifacts/roughness.csv")      # i, alpha_dfa, D_higuchi
saw   = pd.read_csv("artifacts/saw.csv")            # i, S, rho, I
tda   = pd.read_csv("artifacts/tda.csv")            # i, pers_max, pers_sum, pers_entropy, n_pairs

# métriques marché par fenêtre (RV, MDD)
rv = []
mdd = []
for w in wins_price:
    rv.append(realized_vol_from_prices(w))
    mdd.append(max_drawdown(w))
market = pd.DataFrame({"i": np.arange(len(wins_price)), "RV": rv, "MDD": mdd})

# merge
df = market.merge(rough, on="i", how="left").merge(saw, on="i", how="left").merge(tda, on="i", how="left")
df.to_csv("artifacts/metrics_merged.csv", index=False)
print("✅ artifacts/metrics_merged.csv écrit")

# corrélations rapides
cols = ["RV","MDD","alpha_dfa","D_higuchi","S","rho","pers_max","pers_sum","pers_entropy"]
print("\nCorrélations (Pearson):\n", df[cols].corr())

# plots simples (sans styles)
plt.figure(); plt.scatter(df["RV"], df["D_higuchi"], s=8); plt.xlabel("RV"); plt.ylabel("D_higuchi"); plt.tight_layout()
plt.savefig("artifacts/scatter_RV_vs_Dhiguchi.png", dpi=150)

plt.figure(); plt.scatter(df["RV"], df["alpha_dfa"], s=8); plt.xlabel("RV"); plt.ylabel("alpha_dfa"); plt.tight_layout()
plt.savefig("artifacts/scatter_RV_vs_alpha.png", dpi=150)

plt.figure(); plt.scatter(df["RV"], df["S"], s=8); plt.xlabel("RV"); plt.ylabel("SAW S"); plt.tight_layout()
plt.savefig("artifacts/scatter_RV_vs_SAW.png", dpi=150)

print("📈 Figures sauvegardées dans artifacts/:")
print(" - scatter_RV_vs_Dhiguchi.png")
print(" - scatter_RV_vs_alpha.png")
print(" - scatter_RV_vs_SAW.png")
