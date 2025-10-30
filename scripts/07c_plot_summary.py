# --- python -m scripts.07c_plot_summary
# Concatène les résultats ew_* dans artifacts/, crée une table "best-of"
# et des heatmaps OR par proxy × lag × scope.

from __future__ import annotations
import sys, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# ----------------------
# Helpers
# ----------------------

def infer_scope_from_name(name: str) -> str:
    n = name.lower()
    # ordre spécifique → générique
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
    pref = ["baseline", "global", "expanding",
            "expbucket_30m", "expbucket_60m", "expbucket", "other"]
    seen = []
    # d’abord ceux du pref s’ils existent, puis le reste trié alpha
    for s in pref:
        if s in scopes: seen.append(s)
    for s in sorted(set(scopes) - set(seen)):
        seen.append(s)
    return seen

def load_all_runs(art_dir: Path, pattern: str = "ew_*.csv") -> pd.DataFrame:
    rows = []
    for p in sorted(art_dir.glob(pattern)):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "proxy" not in df.columns or "lag" not in df.columns:
            continue
        df["run"] = p.stem  # ex: ew_global_q80
        df["scope"] = infer_scope_from_name(p.stem)
        rows.append(df)
    if not rows:
        raise SystemExit(f"Aucun fichier {pattern} dans {art_dir}")
    out = pd.concat(rows, ignore_index=True)
    # colonnes attendues, on force leur présence
    needed = ["proxy","lag","n","odds_ratio","fisher_p","lift","accuracy","mcc","run","scope"]
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan
    # p Holm si dispo
    if "fisher_p_holm" not in out.columns:
        out["fisher_p_holm"] = out["fisher_p"]
    return out

def pick_best(df: pd.DataFrame) -> pd.DataFrame:
    """
    Meilleur run par (proxy, lag) :
      1) min p_holm
      2) tie-break par |log(OR)| décroissant
    """
    def _score(row):
        orv = row.get("odds_ratio", np.nan)
        if not np.isfinite(orv) or orv <= 0:
            return -np.inf
        return abs(np.log(orv))
    df = df.copy()
    df["abs_log_or"] = df.apply(_score, axis=1)
    df["p_rank"] = df.groupby(["proxy","lag"])["fisher_p_holm"].rank("min")
    df["tie_rank"] = (-df["abs_log_or"]).groupby([df["proxy"], df["lag"]]).rank("min")
    # on filtre p_rank=1, puis tie_rank=1
    best = df[df["p_rank"] == 1].copy()
    best = best[best.groupby(["proxy","lag"])["tie_rank"].transform("min") == best["tie_rank"]]
    # colonnes finales compactes
    keep = ["proxy","lag","n","run","scope","odds_ratio","fisher_p","fisher_p_holm",
            "lift","accuracy","mcc"]
    # garder CI si dispo
    for c in ["or_ci_lo","or_ci_hi"]:
        if c in df.columns and c not in keep:
            keep.append(c)
    return best[keep].sort_values(["proxy","lag"]).reset_index(drop=True)

def plot_heatmaps(full: pd.DataFrame, out_prefix: str = "ew_heatmap"):
    """
    Une heatmap OR par proxy :
      - lignes = lags
      - colonnes = scopes
    Sauvegarde PNG par proxy.
    """
    proxies = sorted(full["proxy"].dropna().unique())
    scopes = nice_scope_order(list(full["scope"].dropna().unique()))
    for proxy in proxies:
        sub = full[full["proxy"] == proxy].copy()
        if sub.empty: continue
        # pivot OR
        piv = sub.pivot_table(index="lag", columns="scope", values="odds_ratio", aggfunc="median")
        # aligner colonnes dans ordre scopes
        piv = piv.reindex(columns=[c for c in scopes if c in piv.columns])
        # plot simple avec matplotlib (pas seaborn, pour compat)
        fig, ax = plt.subplots(figsize=(1.8 + 1.1*len(piv.columns), 0.6 + 0.8*len(piv.index)))
        im = ax.imshow(piv.values, aspect="auto")
        # ticks
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=30, ha="right")
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index)
        ax.set_title(f"Odds ratio by scope (proxy={proxy})")
        # annotations
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                txt = "" if not np.isfinite(val) else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("odds ratio", rotation=90)
        fig.tight_layout()
        out_path = ART / f"{out_prefix}_{proxy}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"🖼️  Heatmap -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=str(ART), help="Répertoire des CSV ew_*")
    ap.add_argument("--pattern", type=str, default="ew_*.csv", help="Pattern de fichiers à agréger")
    ap.add_argument("--out-prefix", type=str, default="ew_all_runs_summary", help="Prefix pour les sorties")
    ap.add_argument("--filter-proxies", type=str, nargs="*", default=None,
                    help="Limiter à certains proxys (ex: roll parkinson)")
    args = ap.parse_args()

    art_dir = Path(args.artifacts)
    full = load_all_runs(art_dir, args.pattern)

    if args.filter_proxies:
        wanted = set([w.lower() for w in args.filter_proxies])
        full = full[full["proxy"].str.lower().isin(wanted)].copy()
        if full.empty:
            raise SystemExit("Filtre proxy vide — vérifie les noms de colonnes 'proxy'.")

    # Sauvegarde concat brute
    all_csv = art_dir / f"{args.out_prefix}.csv"
    full.to_csv(all_csv, index=False)
    print(f"📄 Agrégat complet -> {all_csv}")

    # Best-of
    best = pick_best(full)
    best_csv = art_dir / f"{args.out_prefix}_bestof.csv"
    best_md  = art_dir / f"{args.out_prefix}_bestof.md"
    best.to_csv(best_csv, index=False)
    with best_md.open("w", encoding="utf-8") as f:
        f.write("# Best-of (par proxy × lag)\n\n")
        f.write(best.to_markdown(index=False))
    print(f"📄 Best-of -> {best_csv} & {best_md}")

    # Heatmaps par proxy
    plot_heatmaps(full, out_prefix="ew_heatmap")

if __name__ == "__main__":
    main()
