# --- python -m scripts.07b_summary_early_warning
import sys
from pathlib import Path
import pandas as pd

ART = Path("artifacts")

def main():
    files = sorted(ART.glob("ew_*.csv"))
    if not files:
        print("⚠️  Aucun fichier ew_*.csv trouvé dans artifacts/")
        return

    all_runs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["run"] = f.stem  # ajoute le nom du fichier comme identifiant
            all_runs.append(df)
        except Exception as e:
            print(f"⚠️  Impossible de lire {f}: {e}")

    if not all_runs:
        print("⚠️  Aucun fichier lisible.")
        return

    big = pd.concat(all_runs, ignore_index=True)

    # on ne garde que les colonnes principales
    keep = [
        "run", "proxy", "lag", "n",
        "odds_ratio", "fisher_p", "fisher_p_holm",
        "lift", "accuracy", "mcc"
    ]
    cols = [c for c in keep if c in big.columns]
    big = big[cols]

    # tri par proxy, lag, puis run
    big = big.sort_values(["proxy", "lag", "run"])

    # sauvegarde
    out_csv = ART / "ew_all_runs_summary.csv"
    out_md  = ART / "ew_all_runs_summary.md"
    big.to_csv(out_csv, index=False)
    big.to_markdown(out_md.open("w", encoding="utf-8"), index=False)

    print(f"✅ Résumé sauvegardé → {out_csv} & {out_md}")
    print(big.head(15))

if __name__ == "__main__":
    main()
