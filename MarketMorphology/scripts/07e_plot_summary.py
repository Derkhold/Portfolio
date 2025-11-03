# scripts/07e_plot_summary.py
# Aggregate ew_*.csv results, build a "best-of" table per (proxy, lag),
# and generate odds-ratio heatmaps by proxy × lag × quantile scope.

# Example:
#   python -m scripts.07e_plot_summary \
#       --artifacts artifacts \
#       --pattern "ew_*.csv" \
#       --out-prefix ew_all_runs_summary \
#       --filter-proxies roll parkinson

from __future__ import annotations
import sys
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# ----------------------
# Helpers
# ----------------------

def infer_scope_from_name(name: str) -> str:
    """
    Infer quantile-scope shorthand from the filename stem.
    Priority: expanding-bucket > expanding > global > baseline > other.
    """
    n = name.lower()
    if "expbucket" in n:
        m = re.search(r"ex[bp]?bucket[_\-]?(\d+)(m|min)?", n)
        if m:
            return f"expbucket_{m.group(1)}m"
        return "expbucket"
    if "expanding" in n:
        return "expanding"
    if "global" in n:
        return "global"
    if "metrics" in n:
        return "baseline"
    return "other"

def nice_scope_order(scopes: list[str]) -> list[str]:
    """
    Provide a stable, human-friendly ordering of scopes on the x-axis.
    """
    pref = ["baseline", "global", "expanding",
            "expbucket_30m", "expbucket_60m", "expbucket", "other"]
    seen = []
    for s in pref:
        if s in scopes:
            seen.append(s)
    for s in sorted(set(scopes) - set(seen)):
        seen.append(s)
    return seen

def load_all_runs(art_dir: Path, pattern: str = "ew_*.csv") -> pd.DataFrame:
    """
    Load all ew_*.csv files, tag each with its run name and inferred scope.
    Ensures presence of the expected columns; prefers Holm-adjusted p if present.
    """
    rows = []
    for p in sorted(art_dir.glob(pattern)):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "proxy" not in df.columns or "lag" not in df.columns:
            continue
        df["run"] = p.stem
        df["scope"] = infer_scope_from_name(p.stem)
        rows.append(df)
    if not rows:
        raise SystemExit(f"No files matching {pattern} in {art_dir}")
    out = pd.concat(rows, ignore_index=True)

    expected = ["proxy", "lag", "n", "odds_ratio", "fisher_p",
                "lift", "accuracy", "mcc", "run", "scope"]
    for c in expected:
        if c not in out.columns:
            out[c] = np.nan

    # Prefer Holm-adjusted p if available; otherwise copy raw Fisher p
    if "fisher_p_holm" not in out.columns:
        out["fisher_p_holm"] = out["fisher_p"]

    return out

def pick_best(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best run per (proxy, lag):
      1) minimal Holm-adjusted p-value (fisher_p_holm),
      2) tie-break by |log(OR)| descending.
    """
    def _score(row):
        orv = row.get("odds_ratio", np.nan)
        if not np.isfinite(orv) or orv <= 0:
            return -np.inf
        return abs(np.log(orv))

    df = df.copy()
    df["abs_log_or"] = df.apply(_score, axis=1)
    df["p_rank"] = df.groupby(["proxy", "lag"])["fisher_p_holm"].rank("min")
    df["tie_rank"] = (-df["abs_log_or"]).groupby([df["proxy"], df["lag"]]).rank("min")

    best = df[df["p_rank"] == 1].copy()
    best = best[best.groupby(["proxy", "lag"])["tie_rank"].transform("min") == best["tie_rank"]]

    keep = ["proxy", "lag", "n", "run", "scope", "odds_ratio",
            "fisher_p", "fisher_p_holm", "lift", "accuracy", "mcc"]
    for c in ["or_ci_lo", "or_ci_hi"]:
        if c in df.columns and c not in keep:
            keep.append(c)

    return best[keep].sort_values(["proxy", "lag"]).reset_index(drop=True)

def plot_heatmaps(full: pd.DataFrame, out_prefix: str = "ew_heatmap"):
    """
    For each proxy, plot an OR heatmap:
      - rows  = lags
      - cols  = scopes
      - values = median OR across runs for that (lag, scope)
    Saves one PNG per proxy.
    """
    proxies = sorted(full["proxy"].dropna().unique())
    scopes = nice_scope_order(list(full["scope"].dropna().unique()))

    for proxy in proxies:
        sub = full[full["proxy"] == proxy].copy()
        if sub.empty:
            continue

        # Pivot OR by (lag × scope) using median across runs
        piv = sub.pivot_table(
            index="lag",
            columns="scope",
            values="odds_ratio",
            aggfunc="median",
        )
        piv = piv.reindex(columns=[c for c in scopes if c in piv.columns])

        # Setup figure (diverging palette centered at OR=1.0)
        fig_w = 1.8 + 1.1 * max(1, len(piv.columns))
        fig_h = 0.8 + 0.8 * max(1, len(piv.index))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        norm = TwoSlopeNorm(vmin=max(0.5, np.nanmin(piv.values) if np.isfinite(piv.values).any() else 0.5),
                            vcenter=1.0,
                            vmax=min(2.0, np.nanmax(piv.values) if np.isfinite(piv.values).any() else 1.5))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", norm=norm, cmap="RdBu_r")

        # Ticks and labels
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=30, ha="right")
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_title(f"Odds ratio by scope (proxy={proxy})")
        ax.set_xlabel("scope")
        ax.set_ylabel("lag")

        # Cell annotations
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                txt = "" if not np.isfinite(val) else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("odds ratio", rotation=90)

        fig.tight_layout()
        out_path = ART / f"{out_prefix}_{proxy}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved heatmap: {out_path}")

# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=str(ART), help="Directory containing ew_*.csv files")
    ap.add_argument("--pattern", type=str, default="ew_*.csv", help="Glob pattern to aggregate")
    ap.add_argument("--out-prefix", type=str, default="ew_all_runs_summary", help="Output prefix for summary files")
    ap.add_argument("--filter-proxies", type=str, nargs="*", default=None,
                    help="Optional list of proxies to keep (e.g., roll parkinson)")
    args = ap.parse_args()

    art_dir = Path(args.artifacts)
    full = load_all_runs(art_dir, args.pattern)

    if args.filter_proxies:
        wanted = set(w.lower() for w in args.filter_proxies)
        full = full[full["proxy"].str.lower().isin(wanted)].copy()
        if full.empty:
            raise SystemExit("Proxy filter returned no rows. Check proxy names.")

    # Save concatenated raw aggregate
    all_csv = art_dir / f"{args.out_prefix}.csv"
    full.to_csv(all_csv, index=False)
    print(f"Saved aggregate: {all_csv}")

    # Best-of per (proxy, lag)
    best = pick_best(full)
    best_csv = art_dir / f"{args.out_prefix}_bestof.csv"
    best_md  = art_dir / f"{args.out_prefix}_bestof.md"
    best.to_csv(best_csv, index=False)
    with best_md.open("w", encoding="utf-8") as f:
        f.write("# Best-of (per proxy × lag)\n\n")
        f.write(best.to_markdown(index=False))
    print(f"Saved best-of: {best_csv} and {best_md}")

    # Heatmaps per proxy
    plot_heatmaps(full, out_prefix="ew_heatmap")

if __name__ == "__main__":
    main()
