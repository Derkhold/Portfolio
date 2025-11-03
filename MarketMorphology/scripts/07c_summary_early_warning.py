# scripts/07c_summary_early_warning.py
# Aggregate all early-warning runs (artifacts/ew_*.csv) into a single summary.
#
# Outputs:
#   artifacts/ew_all_runs_summary.csv
#   artifacts/ew_all_runs_summary.md
#
# Columns kept when available:
#   run, proxy, lag, n, odds_ratio, fisher_p, fisher_p_holm, lift, accuracy, mcc

import sys
from pathlib import Path
import pandas as pd

ART = Path("artifacts")

def main():
    files = sorted(ART.glob("ew_*.csv"))
    if not files:
        print("No ew_*.csv files found in artifacts/.")
        return

    all_runs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["run"] = f.stem  # identify source file
            all_runs.append(df)
        except Exception as e:
            print(f"Could not read {f}: {e}")

    if not all_runs:
        print("No readable ew_*.csv files.")
        return

    big = pd.concat(all_runs, ignore_index=True)

    # Keep only main columns when present
    keep = [
        "run", "proxy", "lag", "n",
        "odds_ratio", "fisher_p", "fisher_p_holm",
        "lift", "accuracy", "mcc",
    ]
    cols = [c for c in keep if c in big.columns]
    if not cols:
        print("No expected columns found in input files.")
        return
    big = big[cols]

    # Sort for readability
    sort_cols = [c for c in ["proxy", "lag", "run"] if c in big.columns]
    if sort_cols:
        big = big.sort_values(sort_cols).reset_index(drop=True)

    # Save
    out_csv = ART / "ew_all_runs_summary.csv"
    out_md  = ART / "ew_all_runs_summary.md"
    big.to_csv(out_csv, index=False)
    with out_md.open("w", encoding="utf-8") as f:
        f.write(big.to_markdown(index=False))

    print(f"Saved summary to: {out_csv} and {out_md}")
    # Show a small preview in stdout (optional)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(big.head(15))

if __name__ == "__main__":
    main()
