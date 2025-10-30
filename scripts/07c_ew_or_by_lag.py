# %% scripts/07c_ew_or_by_lag.py
import sys, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="CSV from 07a_early_warning")
    ap.add_argument("--out", default="artifacts/fig_5_11_odds_by_lag.png")
    ap.add_argument("--per-proxy", action="store_true", help="One curve per proxy; else best-only")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    if not {"proxy","lag","odds_ratio","fisher_p","fisher_p_holm"} <= set(df.columns):
        raise SystemExit("Missing columns in metrics CSV.")

    df = df.sort_values(["proxy","lag"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5,4.2))

    if args.per_proxy:
        for pr, sub in df.groupby("proxy"):
            ax.plot(sub["lag"], sub["odds_ratio"], "-o", label=pr.replace("_"," "))
        ax.legend(frameon=False)
    else:
        best = df.sort_values(["proxy","fisher_p","odds_ratio"], ascending=[True,True,False]) \
                 .groupby("proxy", as_index=False).first()
        ax.plot(best["lag"], best["odds_ratio"], "o-")
        for _, r in best.iterrows():
            ax.annotate(r["proxy"].replace("_"," "), (r["lag"], r["odds_ratio"]), xytext=(5,5),
                        textcoords="offset points", fontsize=9)

    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xlabel("lag (steps)"); ax.set_ylabel("odds ratio")
    ax.set_title("Early-warning: odds ratio by horizon")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=180)
    print("🖼️  saved:", args.out)

if __name__ == "__main__":
    main()
