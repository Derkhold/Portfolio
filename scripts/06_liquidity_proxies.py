# --- python -m scripts.06_liquidity_proxies --ohlc data/raw/ES1.xlsx --L 12 --S 1
import sys, argparse
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.io import load_ohlcv, sessionize, resample_ohlc

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# -------------------------
# Proxies de liquidité
# -------------------------

def roll_spread(returns: pd.Series) -> float:
    """
    Roll spread = 2*sqrt(-gamma1) avec gamma1 = autocovariance lag 1.
    Estime gamma1 = rho1 * var(returns) si Series n'a pas autocovariance().
    """
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return np.nan
    var = float(r.var(ddof=1))
    rho1 = float(r.autocorr(lag=1))
    gamma1 = rho1 * var
    if not np.isfinite(gamma1) or gamma1 >= 0:
        return np.nan
    return 2.0 * np.sqrt(-gamma1)

def corwin_schultz(df: pd.DataFrame) -> float:
    # version 2‐bar classique (approx) sur la fenêtre
    hl2 = np.log(df["high"] / df["low"]) ** 2
    beta = hl2.rolling(2).sum().dropna().mean()
    gamma = (np.log(df["high"].rolling(2).max() / df["low"].rolling(2).min()) ** 2).dropna().mean()
    if not np.isfinite(beta) or not np.isfinite(gamma):
        return np.nan
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
    if not np.isfinite(alpha):
        return np.nan
    return 2 * (np.exp(alpha) - 1)

def garman_klass(df: pd.DataFrame) -> float:
    rs = 0.5 * (np.log(df["high"] / df["low"]) ** 2) - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"]) ** 2)
    v = rs.mean()
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else np.nan

def parkinson(df: pd.DataFrame) -> float:
    rs = (1.0 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
    v = rs.mean()
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else np.nan

# -------------------------
# Fenêtrage
# -------------------------

def make_windows_by_day(df: pd.DataFrame, L: int, S: int) -> list[pd.DataFrame]:
    """
    Crée des fenêtres *intraday* uniquement (pas de chevauchement overnight).
    Suppose df déjà filtré par sessionize() et resample_ohlc(), trié par temps.
    """
    wins = []
    # groupby sur la date locale (le timestamp est tz-aware)
    # on utilise .dt.date pour éviter les soucis de fuseau
    df = df.sort_values("timestamp")
    days = df["timestamp"].dt.tz_convert(df["timestamp"].dt.tz).dt.date if hasattr(df["timestamp"].dt, "tz") else df["timestamp"].dt.date
    for _, dfg in df.groupby(days, sort=True):
        n = len(dfg)
        for s in range(0, n - L + 1, S):
            w = dfg.iloc[s:s + L]
            if len(w) == L:
                wins.append(w)
    return wins

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlc", type=str, required=True, help="Fichier CSV/Excel OHLCV")
    ap.add_argument("--L", type=int, default=12, help="Taille de fenêtre (en pas)")
    ap.add_argument("--S", type=int, default=1, help="Stride entre fenêtres")
    ap.add_argument("--limit-n", type=int, default=0, help="(Optionnel) Limiter à N fenêtres pour aligner avec les labels")
    ap.add_argument("--out", type=str, default="proxies_window.csv", help="Nom de fichier de sortie (dans artifacts/)")
    args = ap.parse_args()

    # 1) Chargement + session + resample
    df = load_ohlcv(args.ohlc)
    df = sessionize(df, tz="Europe/Paris", start="09:30", end="16:00")
    df = resample_ohlc(df, rule="5min")

    print(f"✅ Données OHLC chargées: {len(df)} lignes, de {df['timestamp'].min()} à {df['timestamp'].max()}")

    # 2) Fenêtrage intraday strict (pas d'overnight)
    wins = make_windows_by_day(df, L=args.L, S=args.S)
    if args.limit_n and args.limit_n > 0:
        wins = wins[:args.limit_n]

    print(f"🪟 Fenêtres construites (intraday only): {len(wins)} (L={args.L}, S={args.S})")

    # 3) Calcul des proxys par fenêtre
    rows = []
    for w in wins:
        ret = np.log(w["close"]).diff().dropna()
        rows.append({
            "start": w["timestamp"].iloc[0],
            "end":   w["timestamp"].iloc[-1],
            "roll":  roll_spread(ret),
            "corwin_schultz": corwin_schultz(w),
            "garman_klass":   garman_klass(w),
            "parkinson":      parkinson(w),
        })

    out = pd.DataFrame(rows)
    out_path = ART / args.out
    out.to_csv(out_path, index=False)
    print(f"📄 Proxies sauvegardés -> {out_path}")

if __name__ == "__main__":
    main()
