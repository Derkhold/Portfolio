# --- python -m scripts.06c_leadlag_tau_vs_proxies --lags 12
# Alignement robuste tau_c vs proxies + xcorr lead-lag + figures

import sys, argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# -------------------------
# Utils temporels / IO
# -------------------------

def _to_utc(s: pd.Series) -> pd.Series:
    """Force en tz-aware UTC (tolère naive/aware/strings)."""
    x = pd.to_datetime(s, errors="coerce", utc=True)
    return x

def _find_tau_file(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Fichier tau_c introuvable: {p}")
        return p
    cands = sorted(glob.glob(str(ART / "tau_c_rolling*.csv")))
    if not cands:
        raise FileNotFoundError("Aucun tau_c_rolling*.csv dans artifacts/. Passe --tau-c.")
    return Path(cands[-1])

def _load_proxies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start","end"])
    if "start" not in df.columns:
        raise ValueError("proxies_window.csv doit contenir la colonne 'start'.")
    keep = ["start","end","roll","corwin_schultz","garman_klass","parkinson"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].sort_values("start").reset_index(drop=True)
    # normalise en UTC
    df["start"] = _to_utc(df["start"])
    if "end" in df.columns:
        df["end"] = _to_utc(df["end"])
    df["i"] = np.arange(len(df))
    return df

def _guess_tau_column(df: pd.DataFrame) -> str:
    cands = ["tau_c","tau","tau_star","tau_crit","tau_c_hat","tau_c_est"]
    for c in cands:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError("Aucune colonne numérique candidate pour tau_c trouvée.")
    # heuristique: médiane dans [0,1]
    for c in num_cols:
        med = pd.Series(df[c]).dropna().median()
        if np.isfinite(med) and 0 <= med <= 1:
            return c
    return num_cols[0]

def _guess_time_columns_tau(df: pd.DataFrame) -> list[str]:
    cands = ["start","window_start","t","time","timestamp","date"]
    return [c for c in cands if c in df.columns]

def _load_tau_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_cols = _guess_time_columns_tau(df)
    for c in time_cols:
        df[c] = _to_utc(df[c])
    tau_col = _guess_tau_column(df)
    cols = time_cols + [tau_col]
    df = df[cols].copy()
    # crée 'start' si absent mais une colonne temps existe
    if "start" not in df.columns and time_cols:
        df = df.rename(columns={time_cols[0]: "start"})
    if "start" in df.columns:
        df = df.sort_values("start").reset_index(drop=True)
    else:
        df["i"] = np.arange(len(df))
    df = df.rename(columns={tau_col: "tau_c"})
    return df

def _zscore(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype="float64")
    mu = s.mean(skipna=True)
    sd = s.std(ddof=1, skipna=True)
    return (s - mu) / sd if np.isfinite(sd) and sd != 0 else s * np.nan

def _xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> pd.DataFrame:
    a = pd.Series(a, dtype="float64")
    b = pd.Series(b, dtype="float64")
    out = []
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            x = a[-lag:]
            y = b[:len(b)+lag]
        elif lag > 0:
            x = a[:len(a)-lag]
            y = b[lag:]
        else:
            x, y = a, b
        r = pd.Series(x).corr(pd.Series(y)) if len(x) >= 3 and len(y) >= 3 else np.nan
        out.append({"lag": lag, "xcorr": r})
    return pd.DataFrame(out)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxies", type=str, default=str(ART / "proxies_window.csv"),
                    help="Chemin vers proxies_window.csv")
    ap.add_argument("--tau-c", dest="tau_c_path", type=str, default="",
                    help="Chemin vers tau_c_rolling*.csv (sinon auto)")
    ap.add_argument("--lags", type=int, default=12,
                    help="lags ± pour xcorr (fenêtres)")
    ap.add_argument("--align", type=str, choices=["auto","time","index"], default="auto",
                    help="Stratégie d’alignement (auto=try time→asof→index)")
    ap.add_argument("--time-tolerance", type=str, default="6min",
                    help="Tolérance pour merge_asof (ex: 5min, 300s)")
    ap.add_argument("--out-prefix", type=str, default="leadlag_tau_vs_proxies",
                    help="Préfixe des sorties")
    args = ap.parse_args()

    # 1) charge
    P = _load_proxies(Path(args.proxies))
    metrics = [c for c in ["roll","corwin_schultz","garman_klass","parkinson"] if c in P.columns]
    if not metrics:
        raise SystemExit("Aucun proxy trouvé dans proxies_window.csv.")
    tau_path = _find_tau_file(args.tau_c_path if args.tau_c_path else None)
    T = _load_tau_series(tau_path)

    # 2) alignement
    M = None
    log = []

    def _align_time_exact():
        if "start" in T.columns:
            tmp = pd.merge(P[["start","i"] + metrics], T[["start","tau_c"]], on="start", how="inner")
            return tmp.sort_values("start").reset_index(drop=True)
        return None

    def _align_time_asof():
        if "start" in T.columns:
            # asof nécessite clés triées
            p = P[["start","i"] + metrics].sort_values("start").reset_index(drop=True)
            t = T[["start","tau_c"]].sort_values("start").reset_index(drop=True)
            m = pd.merge_asof(p, t, on="start", direction="nearest",
                              tolerance=pd.Timedelta(args.time_tolerance))
            m = m.dropna(subset=["tau_c"]).reset_index(drop=True)
            return m
        return None

    def _align_index():
        n = min(len(P), len(T))
        m = P.iloc[:n][["start","i"] + metrics].copy()
        m["tau_c"] = T.iloc[:n]["tau_c"].to_numpy()
        return m.reset_index(drop=True)

    if args.align in ("auto","time"):
        M = _align_time_exact()
        if M is not None:
            log.append(f"Alignement exact sur 'start' → {len(M)} matchs")
    if (M is None or len(M) < 10) and args.align in ("auto","time"):
        M = _align_time_asof()
        if M is not None:
            log.append(f"Alignement asof (tol={args.time_tolerance}) → {len(M)} matchs")
    if (M is None or len(M) < 10) and args.align in ("auto","index"):
        M = _align_index()
        log.append(f"Alignement par index → {len(M)} lignes")

    if M is None or len(M) < 10:
        print("\n".join(log))
        raise SystemExit("Trop peu d'observations après alignement (<10). Vérifie les timestamps / tolérance / fichiers.")

    # logs utiles
    print("\n".join(log))
    print(f"Aperçu aligné:\n{M[['start','tau_c'] + metrics].head(3)}")

    # 3) z-score & xcorr
    Mz = M.copy()
    Mz["tau_c_z"] = _zscore(M["tau_c"])
    for m in metrics:
        Mz[m + "_z"] = _zscore(M[m])

    rows = []
    for m in metrics:
        xx = _xcorr(Mz["tau_c_z"].to_numpy(), Mz[m + "_z"].to_numpy(), args.lags)
        xx["metric"] = m
        rows.append(xx)
    X = pd.concat(rows, ignore_index=True)

    # 4) save
    out_csv = ART / f"{args.out_prefix}.csv".replace("-", "_")
    X[["metric","lag","xcorr"]].to_csv(out_csv, index=False)

    # heatmap
    try:
        piv = X.pivot(index="metric", columns="lag", values="xcorr").sort_index()
        plt.figure(figsize=(1.2*(2*args.lags+1), 1.0 + 0.7*len(metrics)))
        im = plt.imshow(piv.values, aspect="auto", origin="lower", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="xcorr")
        plt.yticks(ticks=np.arange(len(piv.index)), labels=list(piv.index))
        plt.xticks(ticks=np.arange(piv.shape[1]), labels=list(piv.columns))
        plt.xlabel("lag (fenêtres) — lag>0 : tau_c lead proxy")
        plt.title("Lead–lag: xcorr(tau_c, proxy)")
        plt.tight_layout()
        heat_path = ART / f"{args.out_prefix}_heatmap.png".replace("-", "_")
        plt.savefig(heat_path, dpi=170)
        plt.close()
    except Exception as e:
        print(f"⚠️  Heatmap non générée: {e}")

    # panels
    try:
        n_m = len(metrics)
        ncols = min(2, n_m)
        nrows = int(np.ceil(n_m / ncols))
        plt.figure(figsize=(7*ncols, 2.6*nrows))
        for j, m in enumerate(metrics, 1):
            ax = plt.subplot(nrows, ncols, j)
            ax.plot(M["start"], Mz["tau_c_z"], label="tau_c (z)", linewidth=1.2)
            ax.plot(M["start"], Mz[m + "_z"], label=f"{m} (z)", linewidth=1.0, alpha=0.9)
            ax.set_title(m); ax.grid(True, linewidth=0.3, alpha=0.5)
            if j == 1:
                ax.legend(loc="upper right")
        plt.tight_layout()
        pan_path = ART / f"{args.out_prefix}_panels.png".replace("-", "_")
        plt.savefig(pan_path, dpi=160)
        plt.close()
    except Exception as e:
        print(f"⚠️  Panels non générés: {e}")

    print(f"✅ Lead–lag terminé → {out_csv}")
    print("   (heatmap & panels dans artifacts/ si tout s’est bien passé)")

if __name__ == "__main__":
    main()
