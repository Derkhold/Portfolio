# --- python -m scripts.06b_align_vs_clusters --labels artifacts/labels_softdtw_K2.csv
import sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.io import load_ohlcv, sessionize, resample_ohlc  # utilisé seulement si --ohlc est fourni

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# ---------- utilitaires ----------

def load_labels_any(path: str | Path) -> np.ndarray:
    """
    Charge un fichier de labels flexible:
    - 1 colonne: directement les labels
    - plusieurs colonnes: prend 'label' si présent sinon la dernière colonne numérique
    - sans header: essaie de lire comme une colonne
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Labels introuvables: {p}")

    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.read_csv(p, header=None)

    if df.shape[1] == 1:
        arr = df.iloc[:, 0].to_numpy()
    else:
        if "label" in df.columns:
            arr = df["label"].to_numpy()
        else:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            arr = (df[num_cols[-1]] if num_cols else df.iloc[:, -1]).to_numpy()

    try:
        arr = arr.astype(int)
    except Exception:
        pass
    return arr

def effect_sizes(x, y):
    """Retourne Cohen's d (Welch) et Cliff's delta."""
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    nx, ny = len(x), len(y)
    sp2 = (vx / nx) + (vy / ny)
    d = (mx - my) / np.sqrt(sp2) if (sp2 > 0 and np.isfinite(sp2)) else np.nan
    # Cliff's delta (fallback sans dépendance externe)
    xy = pd.Series(np.concatenate([x, y]))
    ranks = xy.rank(method="average")
    rx = ranks.iloc[:len(x)].sum()
    n, m = len(x), len(y)
    cd = (rx - n * (n + 1) / 2) / (n * m) * 2 - 1
    return float(d), float(cd)

def stars(p):
    if not np.isfinite(p):
        return ""
    return "★★★" if p < 1e-3 else "★★" if p < 1e-2 else "★" if p < 5e-2 else ""

def safe_ttest(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        return float(stats.ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        return np.nan

def safe_mwu(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 1 or len(y) < 1:
        return np.nan
    try:
        return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def load_proxies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start", "end"])
    df = df.sort_values(["start", "end"]).reset_index(drop=True)
    df["i"] = np.arange(len(df))
    return df

def regen_index_from_ohlc(ohlc_path: str, L: int, S: int) -> pd.DataFrame:
    """
    Régénère l'index (start,end,i) des fenêtres à partir de l'OHLC,
    pour sécuriser l'alignement.
    """
    df = load_ohlcv(ohlc_path)
    df = sessionize(df, tz="Europe/Paris", start="09:30", end="16:00")
    df = resample_ohlc(df, rule="5min")
    n = len(df)
    starts, ends = [], []
    for s in range(0, n - L + 1, S):
        w = df.iloc[s:s + L]
        if len(w) == L:
            starts.append(w["timestamp"].iloc[0])
            ends.append(w["timestamp"].iloc[-1])
    idx = pd.DataFrame({"start": starts, "end": ends})
    idx["i"] = np.arange(len(idx))
    return idx

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=str, help="Chemin labels (csv)")
    ap.add_argument("--proxies", type=str, default=str(ART / "proxies_window.csv"))
    ap.add_argument("--ohlc", type=str, default="", help="(Optionnel) OHLC source pour régénérer l’index fenêtrage")
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--out-prefix", type=str, default="cluster_vs_proxies")
    args = ap.parse_args()

    # 1) Labels
    labels = load_labels_any(args.labels)
    n_labels = len(labels)

    # 2) Proxies
    proxies_path = Path(args.proxies)
    if not proxies_path.exists():
        raise FileNotFoundError(f"Fichier proxies introuvable: {proxies_path}")
    P = load_proxies(proxies_path)

    # 3) Alignement temporel optionnel via OHLC
    if args.ohlc:
        idx = regen_index_from_ohlc(args.ohlc, args.L, args.S)
        P = P.merge(idx, on=["start", "end"], how="inner", suffixes=("", "_regen"))
        if "i_regen" in P.columns:
            P["i"] = P["i_regen"]
            P = P.drop(columns=["i_regen"])
        P = P.sort_values("i").reset_index(drop=True)
    else:
        P = P.sort_values("start").reset_index(drop=True)
        P["i"] = np.arange(len(P))

    # 4) Trim/synchronise
    n = min(n_labels, len(P))
    if n_labels != len(P):
        print(f"⚠️  Attention: labels={n_labels}, proxies={len(P)} → on tronque à n={n}")
    P = P.iloc[:n].copy()
    labels = labels[:n]
    P["label"] = labels.astype(int)

    # 5) Stats par cluster
    metrics = ["roll", "corwin_schultz", "garman_klass", "parkinson"]
    metrics = [m for m in metrics if m in P.columns]
    if not metrics:
        raise SystemExit("Aucune colonne proxy attendue trouvée dans proxies_window.csv")

    groups = sorted(P["label"].unique())
    if len(groups) < 2:
        print("⚠️  Un seul cluster détecté; tests bivariés non calculables.")

    # Moyennes / écarts-types par cluster
    res_rows = []
    for g in groups:
        sub = P[P["label"] == g]
        row = {"label": int(g), "n": int(len(sub))}
        for m in metrics:
            row[f"{m}_mean"] = float(np.nanmean(sub[m]))
        for m in metrics:
            row[f"{m}_std"] = float(np.nanstd(sub[m], ddof=1))
        res_rows.append(row)
    res = pd.DataFrame(res_rows)

    # p-values et tailles d'effet (si 2 groupes)
    p_row_t = {"label": "p_value_t"}
    p_row_mwu = {"label": "p_value_mwu"}
    eff_row_d = {"label": "effect_cohen_d"}
    eff_row_cd = {"label": "effect_cliffs_delta"}

    if len(groups) == 2:
        g0, g1 = groups[0], groups[1]
        A = P[P["label"] == g0]
        B = P[P["label"] == g1]
        for m in metrics:
            p_t = safe_ttest(A[m], B[m])
            p_u = safe_mwu(A[m], B[m])
            d, cd = effect_sizes(A[m], B[m])
            p_row_t[m] = p_t
            p_row_mwu[m] = p_u
            eff_row_d[m] = d
            eff_row_cd[m] = cd

    res_p = pd.DataFrame([p_row_t, p_row_mwu])
    res_e = pd.DataFrame([eff_row_d, eff_row_cd])

    # Ordre des colonnes
    ordered_cols = ["label", "n"]
    for m in metrics:
        ordered_cols += [f"{m}_mean"]
    for m in metrics:
        ordered_cols += [f"{m}_std"]
    res = res[ordered_cols]

    # Concat et sorties
    out_csv = ART / f"{args.out_prefix}.csv".replace("-", "_")
    out_md  = ART / f"{args.out_prefix}.md".replace("-", "_")
    out_all = pd.concat([res, res_p, res_e], ignore_index=True, sort=False)
    out_all.to_csv(out_csv, index=False)

    # --- MD formaté (cast en object pour éviter FutureWarning) ---
    pretty = out_all.copy()
    for m in metrics:
        pretty[m] = pretty[m].astype(object)

    for row_idx in range(len(pretty)):
        label_val = pretty.loc[row_idx, "label"]
        if isinstance(label_val, str) and label_val.startswith("p_value"):
            for m in metrics:
                pv = pretty.loc[row_idx, m]
                if pd.notna(pv):
                    pvf = float(pv)
                    pretty.loc[row_idx, m] = f"{pvf:.3g}{stars(pvf)}"
        if isinstance(label_val, str) and label_val.startswith("effect_"):
            for m in metrics:
                v = pretty.loc[row_idx, m]
                if pd.notna(v):
                    pretty.loc[row_idx, m] = f"{float(v):.3f}"

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Cluster vs liquidity proxies\n\n")
        f.write(f"- Labels: `{args.labels}`\n")
        f.write(f"- Proxies: `{proxies_path}`\n")
        f.write(f"- N windows utilisés: {n}\n\n")
        f.write(pretty.to_markdown(index=False))
        f.write("\n\n*Signif.: ★ p<0.05, ★★ p<0.01, ★★★ p<0.001*\n")

    print(f"✅ Alignement OK → {out_csv} & {out_md}")

    # 6) Figure boxplots par cluster
    try:
        plt.figure(figsize=(10, 4.8))
        n_m = len(metrics)
        for j, m in enumerate(metrics, 1):
            plt.subplot(1, n_m, j)
            data = [P[P["label"] == g][m].dropna() for g in groups]
            plt.boxplot(data, tick_labels=[str(g) for g in groups], showfliers=False)
            plt.title(m)
        plt.tight_layout()
        fig_path = ART / "cluster_proxies_boxplots.png"
        plt.savefig(fig_path, dpi=160)
        print(f"🖼️  Figure -> {fig_path}")
    except Exception as e:
        print(f"⚠️  Impossible de générer la figure boxplots: {e}")

if __name__ == "__main__":
    main()
