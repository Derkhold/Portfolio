# --- python -m scripts.05d_cluster_vs_features --labels artifacts/labels_softdtw_K2.csv
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# -------------------- helpers --------------------

def load_labels(path: Path) -> np.ndarray:
    """
    Charge un CSV de labels:
      - 1 colonne (label)
      - ou 2 colonnes (index,label)
    Retourne un np.ndarray shape (N,)
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        lab = df.iloc[:, 0].to_numpy()
    else:
        # Détecte une colonne nommée label sinon prend la dernière
        if "label" in [c.lower() for c in df.columns]:
            col = [c for c in df.columns if c.lower() == "label"][0]
            lab = df[col].to_numpy()
        else:
            lab = df.iloc[:, -1].to_numpy()
    # force int si possible
    try:
        lab = lab.astype(int)
    except Exception:
        pass
    return lab

def autodetect_numeric_columns(df: pd.DataFrame, forbid=("i","label")) -> list[str]:
    cols = []
    for c in df.columns:
        if c in forbid:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        n_valid = np.isfinite(s).sum()
        if n_valid >= max(5, int(0.4 * len(s))):
            cols.append(c)
    return cols

def cohen_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    # pooled sd
    sp2 = ((nx-1)*vx + (ny-1)*vy) / max(nx+ny-2, 1)
    if sp2 <= 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / np.sqrt(sp2)
    return float(d)

def cliffs_delta(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # rapide: échantillonnage si très grand
    n1, n2 = len(x), len(y)
    if n1*n2 > 2e6:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=min(n1, 5000), replace=False)
        y = rng.choice(y, size=min(n2, 5000), replace=False)
        n1, n2 = len(x), len(y)
    greater = 0
    less = 0
    for a in x:
        greater += np.sum(a > y)
        less    += np.sum(a < y)
    delta = (greater - less) / (n1 * n2)
    return float(delta)

def safe_ttest(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        return float(ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        return np.nan

def safe_mwu(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    try:
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def add_stars(p):
    if not np.isfinite(p): return ""
    if p < 1e-3: return "★★★"
    if p < 1e-2: return "★★"
    if p < 5e-2: return "★"
    return ""

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=str, required=True, help="CSV des labels (1 ou 2 colonnes)")
    ap.add_argument("--roughness", type=str, default=str(ART / "roughness.csv"))
    ap.add_argument("--out-prefix", type=str, default="cluster_features")
    args = ap.parse_args()

    # Labels
    lab_path = Path(args.labels)
    if not lab_path.exists():
        raise SystemExit(f"Labels introuvables: {lab_path}")
    labels = load_labels(lab_path)
    N = len(labels)

    # Features (roughness)
    rf_path = Path(args.roughness)
    if not rf_path.exists():
        raise SystemExit(f"Fichier introuvable: {rf_path}")
    rf = pd.read_csv(rf_path)
    # Assure un index 'i' 0..N-1 si présent, sinon crée
    if "i" in rf.columns:
        rf = rf.sort_values("i").reset_index(drop=True)
    else:
        rf.insert(0, "i", np.arange(len(rf)))
    if len(rf) != N:
        # tente de tronquer/aligner
        m = min(len(rf), N)
        rf = rf.iloc[:m, :].copy()
        labels = labels[:m]
        N = m

    # Détection automatique des colonnes numériques
    numeric_cols = autodetect_numeric_columns(rf.drop(columns=["i"], errors="ignore"))
    if not numeric_cols:
        raise SystemExit(f"Aucune colonne numérique détectée dans {rf_path}. Colonnes: {list(rf.columns)}")

    # DataFrame avec labels
    df = rf.copy()
    df["label"] = labels

    # Export "toutes les fenêtres" (utile pour annexes)
    all_out = ART / "all_window_features.csv"
    df.to_csv(all_out, index=False)

    # Calculs par cluster
    uniq = np.unique(labels)
    res_rows = []
    for lab in uniq:
        sub = df[df["label"] == lab]
        row = {"label": lab, "n": int(len(sub))}
        for c in numeric_cols:
            s = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(np.nanmean(s))
            row[f"{c}_std"]  = float(np.nanstd(s, ddof=1))
        res_rows.append(row)

    # p-values & tailles d’effet (K=2)
    p_row      = {"label": "p_value"}
    eff_row_d  = {"label": "effect_cohen_d"}
    eff_row_cd = {"label": "effect_cliffs_delta"}

    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]
        A = df[df["label"] == a]
        B = df[df["label"] == b]
        for c in numeric_cols:
            x = pd.to_numeric(A[c], errors="coerce")
            y = pd.to_numeric(B[c], errors="coerce")
            # tests
            p_t   = safe_ttest(x, y)
            p_mwu = safe_mwu(x, y)
            p_row[f"{c}_t"]   = p_t
            p_row[f"{c}_mwu"] = p_mwu
            # effets
            eff_row_d[f"{c}_cohen_d"]       = cohen_d(x, y)
            eff_row_cd[f"{c}_cliffs_delta"] = cliffs_delta(x, y)

    # Assemble CSV (numérique uniquement)
    out_csv_rows = []
    out_csv_rows.extend(res_rows)
    if len(uniq) == 2:
        out_csv_rows.append(p_row)
        out_csv_rows.append(eff_row_d)
        out_csv_rows.append(eff_row_cd)

    res_df = pd.DataFrame(out_csv_rows)
    res_csv_path = ART / f"{args.out_prefix}.csv".replace("-", "_")
    res_df.to_csv(res_csv_path, index=False)

    # ---------------- Sanity checks (parity / early-late) ----------------
    idx = np.arange(N)
    parity = (idx % 2).astype(int)
    early_late = (idx >= (N // 2)).astype(int)

    def chi2_and_ari(binary_vec):
        tab = pd.crosstab(labels, binary_vec)
        chi2_p = chi2_contingency(tab)[1]
        ari = adjusted_rand_score(labels, binary_vec)
        return chi2_p, ari, tab

    p_parity, ari_parity, tab_parity = chi2_and_ari(parity)
    p_early,  ari_early,  tab_early  = chi2_and_ari(early_late)

    # ---------------- Markdown (table lisible + sanity) ----------------
    res_md = pd.DataFrame(index=[str(r["label"]) for r in res_rows] +
                                 ["p_value", "effect_cohen_d", "effect_cliffs_delta"],
                          columns=["n"] +
                                  [f"{c}_mean" for c in numeric_cols] +
                                  [f"{c}_std"  for c in numeric_cols] +
                                  [f"{c}_t"    for c in numeric_cols] +
                                  [f"{c}_mwu"  for c in numeric_cols] +
                                  [f"{c}_cohen_d" for c in numeric_cols] +
                                  [f"{c}_cliffs_delta" for c in numeric_cols],
                          dtype=object)

    for r in res_rows:
        ridx = str(r["label"])
        res_md.loc[ridx, "n"] = r["n"]
        for c in numeric_cols:
            res_md.loc[ridx, f"{c}_mean"] = f"{r[f'{c}_mean']:.6g}"
            res_md.loc[ridx, f"{c}_std"]  = f"{r[f'{c}_std']:.6g}"

    if len(uniq) == 2:
        for c in numeric_cols:
            pt = p_row.get(f"{c}_t", np.nan)
            pm = p_row.get(f"{c}_mwu", np.nan)
            d  = eff_row_d.get(f"{c}_cohen_d", np.nan)
            cd = eff_row_cd.get(f"{c}_cliffs_delta", np.nan)

            res_md.loc["p_value", f"{c}_t"]   = f"{pt:.3g}{add_stars(pt)}" if np.isfinite(pt) else ""
            res_md.loc["p_value", f"{c}_mwu"] = f"{pm:.3g}{add_stars(pm)}" if np.isfinite(pm) else ""
            res_md.loc["effect_cohen_d", f"{c}_cohen_d"]       = f"{d:.3f}"  if np.isfinite(d)  else ""
            res_md.loc["effect_cliffs_delta", f"{c}_cliffs_delta"] = f"{cd:.3f}" if np.isfinite(cd) else ""

    md_lines = []
    md_lines.append("# Cluster vs features (auto-detected numeric columns)\n")
    md_lines.append(f"- Labels: `{args.labels}`")
    md_lines.append(f"- N windows: {N}\n")
    md_lines.append(res_md.fillna("").to_markdown(index=True))
    md_lines.append("\n*Signif.: ★ p<0.05, ★★ p<0.01, ★★★ p<0.001*")

    md_lines.append("\n\n## Sanity checks (proxy variables)")
    md_lines.append("- **Parity proxy (even/odd windows)** : teste si les clusters recouvrent juste la parité des fenêtres.")
    md_lines.append("- **Early vs Late (first half vs second half)** : teste une dépendance à la chronologie globale.\n")
    md_lines.append(f"- Parity: χ² p-value = **{p_parity:.3g}**, ARI(labels, parity) = **{ari_parity:.3f}**")
    md_lines.append(f"- Early/Late: χ² p-value = **{p_early:.3g}**, ARI(labels, early/late) = **{ari_early:.3f}**\n")
    md_lines.append("**Contingence (labels × parity)**")
    md_lines.append(pd.crosstab(labels, parity).to_markdown())
    md_lines.append("\n**Contingence (labels × early/late)**")
    md_lines.append(pd.crosstab(labels, early_late).to_markdown())

    out_md = ART / f"{args.out_prefix}.md".replace("-", "_")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"✅ Features -> {res_csv_path} & {out_md}  (toutes -> {all_out})")
    print("Colonnes utilisées :", ", ".join(numeric_cols))

if __name__ == "__main__":
    main()
