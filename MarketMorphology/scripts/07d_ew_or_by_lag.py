# scripts/07d_ew_or_by_lag.py
# Plot odds ratio vs. horizon (lag) from early-warning metrics (07a output).

# Examples:
#   python -m scripts.07d_ew_or_by_lag --metrics artifacts/ew_q20_lowtail_eventHigh.csv
#   python -m scripts.07d_ew_or_by_lag --metrics artifacts/ew_q20_lowtail_eventHigh.csv --per-proxy

# Notes:
# - If --per-proxy is set, draws one curve per proxy across lags.
# - Otherwise, selects the "best" point per proxy using Holm-adjusted p-values
#   when available (fallback: raw Fisher p), then plots those best points.

import sys
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _pick_puse_column(df: pd.DataFrame) -> pd.Series:
    """Prefer Holm-adjusted p if present, else raw Fisher p."""
    if "fisher_p_holm" in df.columns and not df["fisher_p_holm"].isna().all():
        return df["fisher_p_holm"]
    return df["fisher_p"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="CSV produced by 07a_early_warning")
    ap.add_argument("--out", default="artifacts/fig_5_11_odds_by_lag.png")
    ap.add_argument("--per-proxy", action="store_true", help="One curve per proxy; else best-only")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    required = {"proxy", "lag", "odds_ratio", "fisher_p"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Missing columns in metrics CSV. Required at least: {sorted(required)}")

    # Choose the p-value column to rank by (Holm if available)
    df = df.copy()
    df["p_use"] = _pick_puse_column(df)

    df = df.sort_values(["proxy", "lag"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    if args.per_proxy:
        # One curve per proxy across lags
        for pr, sub in df.groupby("proxy"):
            ax.plot(sub["lag"], sub["odds_ratio"], "-o", label=pr.replace("_", " "))
        ax.legend(frameon=False)
    else:
        # Best point per proxy: lowest p (Holm if present), then highest OR as tiebreaker
        best = (
            df.sort_values(["proxy", "p_use", "odds_ratio"], ascending=[True, True, False])
              .groupby("proxy", as_index=False)
              .first()
        )
        ax.plot(best["lag"], best["odds_ratio"], "o-")
        for _, r in best.iterrows():
            ax.annotate(
                r["proxy"].replace("_", " "),
                (r["lag"], r["odds_ratio"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("odds ratio")
    ax.set_title("Early-warning: odds ratio by horizon")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
