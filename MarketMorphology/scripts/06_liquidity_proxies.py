# scripts/06_liquidity_proxies.py
# Usage:
#   python -m scripts.06_liquidity_proxies --ohlc data/raw/ES1.xlsx --L 12 --S 1
#
# Output:
#   artifacts/proxies_window.csv  (columns: start, end, roll, corwin_schultz, garman_klass, parkinson)

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
# Liquidity proxies
# -------------------------

def roll_spread(returns: pd.Series) -> float:
    """
    Roll (1984) effective spread estimator on a window:
      spread ≈ 2 * sqrt(-γ1), with γ1 the lag-1 autocovariance of returns.
    Here γ1 is approximated by ρ1 * Var(returns), where ρ1 is lag-1 autocorrelation.
    Returns NaN if γ1 ≥ 0 or insufficient data.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return np.nan
    var = float(r.var(ddof=1))
    rho1 = float(r.autocorr(lag=1))
    gamma1 = rho1 * var
    if not np.isfinite(gamma1) or gamma1 >= 0:
        return np.nan
    return 2.0 * float(np.sqrt(-gamma1))

def corwin_schultz(df: pd.DataFrame) -> float:
    """
    Corwin–Schultz (2012) two-bar high–low based spread estimator (window-average).
    Uses the standard 2-bar approximation.
    """
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
    """
    Garman–Klass (1980) volatility estimator on the window, returned as sigma (not variance).
    """
    rs = 0.5 * (np.log(df["high"] / df["low"]) ** 2) - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"]) ** 2)
    v = rs.mean()
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else np.nan

def parkinson(df: pd.DataFrame) -> float:
    """
    Parkinson (1980) volatility estimator on the window, returned as sigma (not variance).
    """
    rs = (1.0 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
    v = rs.mean()
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else np.nan

# -------------------------
# Windowing
# -------------------------

def make_windows_by_day(df: pd.DataFrame, L: int, S: int) -> list[pd.DataFrame]:
    """
    Create *intraday* windows only (no overnight crossing).
    Assumes df is already sessionized and resampled, sorted by timestamp.
    """
    wins = []
    df = df.sort_values("timestamp")
    # Group by local date to avoid TZ issues
    if hasattr(df["timestamp"].dt, "tz_localize") or hasattr(df["timestamp"].dt, "tz"):
        local_dates = df["timestamp"].dt.tz_convert(df["timestamp"].dt.tz).dt.date
    else:
        local_dates = df["timestamp"].dt.date

    for _, dfg in df.groupby(local_dates, sort=True):
        n = len(dfg)
        for s in range(0, n - L + 1, S):
            w = dfg.iloc[s : s + L]
            if len(w) == L:
                wins.append(w)
    return wins

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlc", type=str, required=True, help="Path to CSV/Excel OHLCV file.")
    ap.add_argument("--L", type=int, default=12, help="Window length in bars.")
    ap.add_argument("--S", type=int, default=1, help="Window stride in bars.")
    ap.add_argument("--limit-n", type=int, default=0, help="Optional cap on number of windows (align with labels if needed).")
    ap.add_argument("--out", type=str, default="proxies_window.csv", help="Output filename (written under artifacts/).")
    # Session parameters (kept here for clarity; adjust to your market hours if needed)
    ap.add_argument("--tz", type=str, default="Europe/Paris", help="Session timezone.")
    ap.add_argument("--start", type=str, default="09:30", help="Session start (HH:MM).")
    ap.add_argument("--end", type=str, default="16:00", help="Session end (HH:MM).")
    ap.add_argument("--rule", type=str, default="5min", help="Resampling rule for OHLCV (e.g., '5min').")
    args = ap.parse_args()

    # 1) Load, sessionize, resample
    df = load_ohlcv(args.ohlc)
    df = sessionize(df, tz=args.tz, start=args.start, end=args.end)
    df = resample_ohlc(df, rule=args.rule)

    print(f"Loaded OHLC: {len(df)} rows, from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # 2) Strict intraday windowing (no overnight)
    wins = make_windows_by_day(df, L=args.L, S=args.S)
    if args.limit_n and args.limit_n > 0:
        wins = wins[: args.limit_n]

    print(f"Built intraday windows: {len(wins)}  (L={args.L}, S={args.S})")

    # 3) Compute proxies per window
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
    print(f"Saved liquidity proxies -> {out_path}")

if __name__ == "__main__":
    main()
