# scripts/07a_early_warning.py
# Early-warning metrics: link S (tau_c tail) to future proxy extremes.
#
# Features:
# - --signal-tail {high,low,two_sided}
# - --event-tail  {high,low}
# - Quantile scopes without look-ahead leak: global / expanding / expanding_bucket
# - Holm–Bonferroni multiple testing correction
# - Block-bootstrap confidence intervals for odds ratios
# - Optional dose–response sweep over q_tau
#
# Example:
# python -m scripts.07a_early_warning \
#   --tau-c artifacts/tau_c_per_window_dense.csv \
#   --proxies artifacts/proxies_window.csv \
#   --time-col-tau start --time-col-proxies start \
#   --lags 0 1 2 3 6 12 \
#   --q-tau 0.20 --q-proxy 0.80 \
#   --signal-tail low --event-tail high \
#   --quantile-scope expanding_bucket --bucket-size 60min \
#   --holm --block-bootstrap 60 --dose-response \
#   --out-prefix ew_q20_lowtail_eventHigh

import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------

def ensure_utc_series(s: pd.Series) -> pd.Series:
    """Coerce a series to timezone-aware UTC datetimes."""
    return pd.to_datetime(s, errors="coerce", utc=True)

# ---------------------------------------------------------------------
# Column selection / standardization
# ---------------------------------------------------------------------

def pick_time_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Pick a time column, preferring `preferred` if present."""
    cands = ([preferred] if preferred else []) + ["start","timestamp","Timestamp","time","datetime","date","ts"]
    for c in cands:
        if c in df.columns:
            return c
    raise SystemExit(f"No time column found. Available: {list(df.columns)}")

def pick_tau_col(df: pd.DataFrame) -> str:
    """Find the tau column (tau_c if available, otherwise any column starting with 'tau')."""
    if "tau_c" in df.columns:
        return "tau_c"
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("tau","tauc","tau_c") or cl.startswith("tau"):
            return c
    raise SystemExit("No 'tau_c' column and none starting with 'tau'.")

def standardize_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep and standardize expected proxy names:
      parkinson, garman_klass, corwin_schultz, roll.
    Requires a 'start' time column (added upstream).
    """
    std = ["parkinson","garman_klass","corwin_schultz","roll"]
    lower = {str(c).lower(): c for c in df.columns}
    keep = [(lower[name], name) for name in std if name in lower]
    if not keep:
        raise SystemExit("None of the expected proxies found (parkinson, garman_klass, corwin_schultz, roll).")
    out = df.rename(columns=dict(keep))
    if "start" not in out.columns:
        raise SystemExit("Missing 'start' in proxies after time standardization.")
    return out[["start"] + [n for _, n in keep]]

# ---------------------------------------------------------------------
# Quantile thresholds (no look-ahead leakage)
# ---------------------------------------------------------------------

def global_threshold(series: pd.Series, q: float) -> pd.Series:
    thr = series.quantile(q)
    return pd.Series([thr] * len(series), index=series.index)

def expanding_threshold(series: pd.Series, q: float, min_periods: int = 30) -> pd.Series:
    out = series.expanding().quantile(q).shift(1)
    g = series.quantile(q)
    return out.fillna(g)

def expanding_bucket_threshold(series: pd.Series, times: pd.Series, bucket: pd.Timedelta, q: float) -> pd.Series:
    times = ensure_utc_series(times)
    bkt = times.dt.floor(bucket)
    df = pd.DataFrame({"val": series, "bucket": bkt})
    agg = df.groupby("bucket", dropna=False)["val"].apply(list)
    buckets = agg.index.sort_values()
    thr_map, hist = {}, []
    g = series.quantile(q)
    for bk in buckets:
        if not hist:
            thr_map[bk] = g
        else:
            all_prev = np.concatenate(hist)
            all_prev = all_prev[~np.isnan(all_prev)]
            thr_map[bk] = np.nanquantile(all_prev, q) if len(all_prev) else g
        cur = np.array(agg.loc[bk])
        hist.append(cur[~np.isnan(cur)])
    return bkt.map(thr_map)

def make_threshold_scope(series, times, q, scope, bucket):
    if scope == "global":
        return global_threshold(series, q)
    if scope == "expanding":
        return expanding_threshold(series, q)
    return expanding_bucket_threshold(series, times, bucket, q)

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def contingency_and_or(y_true: np.ndarray, s_mask: np.ndarray):
    """
    Build 2x2 contingency table for S (signal) vs Y (event) and compute:
    odds ratio (with Haldane–Anscombe 0.5 correction if needed),
    Fisher p-value, conditional event rates, lift, accuracy, MCC.
    """
    a = int(((s_mask == 1) & (y_true == 1)).sum())
    b = int(((s_mask == 1) & (y_true == 0)).sum())
    c = int(((s_mask == 0) & (y_true == 1)).sum())
    d = int(((s_mask == 0) & (y_true == 0)).sum())
    a2, b2, c2, d2 = (a, b, c, d)
    if 0 in (a, b, c, d):
        a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    OR = (a2 * d2) / (b2 * c2)
    _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    rate_S1 = a / max(1, (a + b))
    rate_S0 = c / max(1, (c + d))
    lift = (rate_S1 / rate_S0) if rate_S0 > 0 else np.nan
    acc = (a + d) / max(1, (a + b + c + d))
    denom = np.sqrt((a + b) * (a + c) * (d + b) * (d + c))
    mcc = ((a * d - b * c) / denom) if denom > 0 else 0.0
    return dict(a=a, b=b, c=c, d=d, odds_ratio=OR, fisher_p=p,
                event_rate_S1=rate_S1, event_rate_S0=rate_S0,
                lift=lift, accuracy=acc, mcc=mcc)

def block_bootstrap_or(df, signal_col, y_col, block_len, B=200, seed=42):
    """Block bootstrap CI for the odds ratio using fixed-length blocks (non-overlapping)."""
    if len(df) < block_len or block_len <= 1 or B <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(df)
    nblocks = max(1, n // block_len)
    ors = []
    for _ in range(B):
        idx = []
        for _b in range(nblocks):
            s = rng.integers(0, max(1, n - block_len + 1))
            idx.extend(range(s, s + block_len))
        sub = df.iloc[idx]
        res = contingency_and_or(sub[y_col].to_numpy(), sub[signal_col].to_numpy())
        ors.append(res["odds_ratio"])
    lo, hi = np.nanpercentile(ors, [2.5, 97.5])
    return lo, hi

# ---------------------------------------------------------------------
# Dose–response plotting
# ---------------------------------------------------------------------

def plot_event_rates(df, proxies, q_grid, outpath: Path):
    """Plot event rates conditional on S for multiple q_tau levels."""
    plt.figure(figsize=(9, 4.8))
    for j, proxy in enumerate(proxies, 1):
        plt.subplot(1, len(proxies), j)
        rows = []
        for q in q_grid:
            sub = df[df["q_used"] == q]
            if len(sub) == 0:
                continue
            S1 = sub[sub["S"] == 1]["Y"].mean() if len(sub[sub["S"] == 1]) else np.nan
            S0 = sub[sub["S"] == 0]["Y"].mean() if len(sub[sub["S"] == 0]) else np.nan
            rows.append((q, S0, S1))
        if rows:
            qs, s0, s1 = zip(*rows)
            plt.plot(qs, s0, marker="o", label="event_rate | S=0")
            plt.plot(qs, s1, marker="o", label="event_rate | S=1")
        plt.title(proxy)
        plt.xlabel("q_tau")
        plt.ylabel("event rate")
        if j == len(proxies):
            plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Early-warning metrics: tau_c tails vs future proxy extremes.")
    ap.add_argument("--tau-c", type=str, default=str(ART / "tau_c_per_window_dense.csv"))
    ap.add_argument("--proxies", type=str, default=str(ART / "proxies_window.csv"))
    ap.add_argument("--time-col-tau", type=str, default="start")
    ap.add_argument("--time-col-proxies", type=str, default="start")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3], help="Future lags (in windows) for event definition.")
    ap.add_argument("--q-tau", type=float, default=0.8, help="Quantile for tau_c signal threshold.")
    ap.add_argument("--q-proxy", type=float, default=0.8, help="Quantile for proxy event threshold.")
    ap.add_argument("--signal-tail", choices=["high", "low", "two_sided"], default="high",
                    help="Which tail of tau_c defines S=1.")
    ap.add_argument("--event-tail", choices=["high", "low"], default="high",
                    help="Which tail of the proxy defines Y=1.")
    ap.add_argument("--bucket-size", type=str, default="60min", help="Bucket size for expanding_bucket scope.")
    ap.add_argument("--quantile-scope", choices=["global", "expanding", "expanding_bucket"], default="global",
                    help="How quantile thresholds are computed without look-ahead.")
    ap.add_argument("--holm", action="store_true", help="Apply Holm–Bonferroni correction.")
    ap.add_argument("--block-bootstrap", type=int, default=0, help="Block length for OR CI (0 disables).")
    ap.add_argument("--dose-response", action="store_true", help="Run dose–response sweep over q_tau.")
    ap.add_argument("--out-prefix", type=str, default="ew_metrics", help="Prefix for output files.")
    args = ap.parse_args()

    # 1) Load and standardize
    T = pd.read_csv(args.tau_c)
    P = pd.read_csv(args.proxies)
    tcol_T = pick_time_col(T, args.time_col_tau)
    tcol_P = pick_time_col(P, args.time_col_proxies)
    T[tcol_T] = ensure_utc_series(T[tcol_T])
    P[tcol_P] = ensure_utc_series(P[tcol_P])
    T = T.rename(columns={tcol_T: "start"})
    P = P.rename(columns={tcol_P: "start"})
    tau_col = pick_tau_col(T)
    if tau_col != "tau_c":
        T = T.rename(columns={tau_col: "tau_c"})
    P = standardize_proxy_columns(P)
    proxies_cols = [c for c in ["parkinson", "garman_klass", "corwin_schultz", "roll"] if c in P.columns]

    # 2) Merge on time
    df = T[["start", "tau_c"]].merge(P, on="start", how="inner").sort_values("start").reset_index(drop=True)
    print(f"Aligned: {len(df)} windows | proxies={proxies_cols}")
    if len(df) < 30:
        print("Warning: Few observations after merge; results may be fragile.")

    # 3) Build signal S from tau_c using no-leak thresholds
    bucket = pd.Timedelta(args.bucket_size)
    def make_thr(series, times, q): return make_threshold_scope(series, times, q, args.quantile_scope, bucket)
    thr_tau = make_thr(df["tau_c"], df["start"], args.q_tau)

    if args.signal_tail == "high":
        S = (df["tau_c"] >= thr_tau).astype(int)
    elif args.signal_tail == "low":
        S = (df["tau_c"] <= thr_tau).astype(int)
    else:
        thr_low  = make_thr(df["tau_c"], df["start"], min(args.q_tau, 1 - args.q_tau))
        thr_high = make_thr(df["tau_c"], df["start"], max(args.q_tau, 1 - args.q_tau))
        S = ((df["tau_c"] <= thr_low) | (df["tau_c"] >= thr_high)).astype(int)

    # 4) Evaluate contingency metrics for each proxy × lag
    results = []
    for proxy in proxies_cols:
        thr_proxy_full = make_thr(df[proxy], df["start"], args.q_proxy)
        for h in args.lags:
            if args.event_tail == "high":
                Y = (df[proxy].shift(-h) >= thr_proxy_full.shift(-h)).astype(float)
            else:
                Y = (df[proxy].shift(-h) <= thr_proxy_full.shift(-h)).astype(float)
            valid = Y.notna() & S.notna()
            sub = pd.DataFrame({"start": df["start"][valid],
                                "S": S[valid].astype(int),
                                "Y": Y[valid].astype(int)}).reset_index(drop=True)
            if len(sub) < 20:
                continue
            res = contingency_and_or(sub["Y"].to_numpy(), sub["S"].to_numpy())
            res.update({"proxy": proxy, "lag": int(h), "n": int(len(sub))})
            if args.block_bootstrap and args.block_bootstrap > 0:
                lo, hi = block_bootstrap_or(sub, "S", "Y", block_len=args.block_bootstrap, B=400)
                res["or_ci_lo"], res["or_ci_hi"] = lo, hi
            else:
                res["or_ci_lo"], res["or_ci_hi"] = np.nan, np.nan
            results.append(res)

    if not results:
        raise SystemExit("No computable result (too few valid rows after shifts/filters).")
    R = pd.DataFrame(results)

    # 5) Holm–Bonferroni (step-down)
    if args.holm:
        p = R["fisher_p"].to_numpy()
        m = len(p)
        order = np.argsort(p)
        adj = np.empty_like(p)
        prev = 0.0
        for rank, idx in enumerate(order, start=1):
            adj_p = (m - rank + 1) * p[idx]
            adj_p = max(prev, adj_p)     # enforce monotonicity
            adj[idx] = min(1.0, adj_p)
            prev = adj[idx]
        R["fisher_p_holm"] = adj
    else:
        R["fisher_p_holm"] = np.nan

    # 6) Save CSV and Markdown
    out_csv = ART / f"{args.out_prefix}.csv"
    out_md  = ART / f"{args.out_prefix}.md"
    R_sorted = R.sort_values(["fisher_p", "odds_ratio"], ascending=[True, False]).reset_index(drop=True)
    R_sorted.to_csv(out_csv, index=False)

    R_md = R_sorted.copy()
    for c in ["odds_ratio","fisher_p","fisher_p_holm","event_rate_S0","event_rate_S1",
              "lift","accuracy","mcc","or_ci_lo","or_ci_hi"]:
        if c in R_md.columns:
            R_md[c] = R_md[c].apply(lambda x: f"{x:.6g}" if pd.notna(x) else "")
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Early warning metrics (tau_c → proxies)\n\n")
        f.write(f"- tau_c file: `{args.tau_c}`\n")
        f.write(f"- proxies file: `{args.proxies}`\n")
        f.write(f"- time cols: tau={args.time_col_tau} | proxies={args.time_col_proxies}\n")
        f.write(f"- tested lags: {args.lags}\n")
        f.write(f"- quantile scope: **{args.quantile_scope}** (bucket={args.bucket_size})\n")
        f.write(f"- q_tau={args.q_tau}, q_proxy={args.q_proxy}\n")
        f.write(f"- signal tail: {args.signal_tail}\n")
        f.write(f"- event tail: {args.event_tail}\n")
        if args.holm: f.write("- Holm–Bonferroni: **ON**\n")
        if args.block_bootstrap: f.write(f"- Block bootstrap OR CI: **ON** (block_len={args.block_bootstrap})\n\n")
        f.write(R_md.to_markdown(index=False))
        f.write("\n")

    # 7) Bar plot: best lag per proxy
    fig_path = ART / "ew_event_rates.png"
    try:
        best = R_sorted.groupby("proxy", as_index=False).first()
        plt.figure(figsize=(8, 4.2))
        x = np.arange(len(best)); w = 0.35
        plt.bar(x - w/2, best["event_rate_S0"], width=w, label="event_rate (S=0)")
        plt.bar(x + w/2, best["event_rate_S1"], width=w, label="event_rate (S=1)")
        plt.xticks(x, best["proxy"])
        plt.ylabel("Event rate (proxy ≥/≤ quantile)")
        plt.title("Conditional event rates — best lag per proxy")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
    except Exception as e:
        print(f"Warning: Could not build event-rate figure: {e}")

    print(f"Wrote results to {out_csv} and {out_md}")
    print(f"Wrote figure to {fig_path}")

    # 8) Dose–response sweep (optional)
    if args.dose_response:
        q_grid = [0.6, 0.7, 0.8, 0.9] if args.signal_tail != "low" else [0.1, 0.2, 0.3, 0.4]
        rows = []
        for proxy in proxies_cols:
            subR = R_sorted[R_sorted["proxy"] == proxy]
            if subR.empty:
                continue
            lag_star = int(subR.iloc[0]["lag"])
            for q in q_grid:
                thr_tau_q = make_thr(df["tau_c"], df["start"], q)
                if args.signal_tail == "high":
                    S_q = (df["tau_c"] >= thr_tau_q).astype(int)
                elif args.signal_tail == "low":
                    S_q = (df["tau_c"] <= thr_tau_q).astype(int)
                else:
                    thr_low_q  = make_thr(df["tau_c"], df["start"], min(q, 1 - q))
                    thr_high_q = make_thr(df["tau_c"], df["start"], max(q, 1 - q))
                    S_q = ((df["tau_c"] <= thr_low_q) | (df["tau_c"] >= thr_high_q)).astype(int)

                thr_proxy_full_q = make_thr(df[proxy], df["start"], args.q_proxy)
                if args.event_tail == "high":
                    Y_q = (df[proxy].shift(-lag_star) >= thr_proxy_full_q.shift(-lag_star)).astype(float)
                else:
                    Y_q = (df[proxy].shift(-lag_star) <= thr_proxy_full_q.shift(-lag_star)).astype(float)

                valid = Y_q.notna() & S_q.notna()
                rows.append(pd.DataFrame({
                    "proxy": proxy, "q_used": q,
                    "S": S_q[valid].astype(int), "Y": Y_q[valid].astype(int)
                }))
        if rows:
            DR = pd.concat(rows, ignore_index=True)
            dr_path = ART / "ew_dose_response.png"
            plot_event_rates(DR, proxies_cols, q_grid, dr_path)
            print(f"Wrote dose–response figure to {dr_path}")

if __name__ == "__main__":
    main()
