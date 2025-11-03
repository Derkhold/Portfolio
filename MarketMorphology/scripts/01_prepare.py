# scripts/01_prepare.py
# Build 5-minute windows and a few descriptive figures (Chapter 5 §5.1)


from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

# make "cmorph" importable when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmorph.io import load_excel_ohlcv, sessionize  # type: ignore
from cmorph.windows import make_windows, normalize_window_returns_z  # type: ignore


# --- Config ------------------------------------------------------------------

ASSET = "ES1"
RAW_PATH = "data/raw/ES1I.xlsx"   # change per asset
TZ = "Europe/Paris"
START_HHMM = "09:30"
END_HHMM = "16:00"
RESAMPLE_TO_5MIN = True

L, S = 12, 1                      # window length/stride (12*5min ≈ 1h)
THEO_BARS_PER_DAY = 78            # 09:30→16:00 at 5-min

ARTIFACTS = Path("artifacts")
FIGDIR = ARTIFACTS / "figures"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


# --- Small helpers -----------------------------------------------------------

def _infer_step_minutes(ts: pd.Series) -> int:
    """Median spacing (minutes) between consecutive timestamps."""
    if len(ts) < 3:
        return 30
    diffs = np.diff(ts.values).astype("timedelta64[m]").astype(int)
    return int(np.median(diffs))


def _robust_vol_bps(ret: pd.Series) -> float:
    """MAD-based sigma (≈ 1.4826*MAD) in bps for 5-min returns."""
    if ret.isna().all():
        return 0.0
    mad = (ret - ret.median()).abs().median()
    return float(1.4826 * mad * 1e4)


def _time_index_from_groupby_clock(grouped_index) -> pd.DatetimeIndex:
    """Map time-of-day index to a fixed date for clean Matplotlib formatting."""
    times = pd.to_datetime(grouped_index.astype(str), format="%H:%M:%S")
    return pd.DatetimeIndex(times.map(lambda t: t.replace(year=2000, month=1, day=1)))


# --- Main --------------------------------------------------------------------

def main() -> None:
    # 1) Load and sessionize
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"File not found: {RAW_PATH}")

    print(f"Reading: {RAW_PATH}")
    df = load_excel_ohlcv(RAW_PATH)           # standardized cols, tz-aware UTC
    df = sessionize(df, tz=TZ, start=START_HHMM, end=END_HHMM)

    print("After sessionize:", df.head(3), sep="\n")
    print("Timestamp range:", df["timestamp"].min(), "->", df["timestamp"].max())
    print("Rows:", len(df))

    # 2) Homogenize frequency
    step_min = _infer_step_minutes(df["timestamp"])
    print(f"Detected time step ≈ {step_min} min")

    if RESAMPLE_TO_5MIN and step_min <= 10:
        print("Resampling to 5-minute bars")
        x = df.set_index("timestamp").sort_index()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in x.columns:
            agg["volume"] = "sum"
        df = (
            x.resample("5min").agg(agg)
             .dropna(subset=["open", "high", "low", "close"])
             .reset_index()
        )
    else:
        print("Keep native granularity")

    print("Rows after resample:", len(df))
    print(df.head(3))

    # 3) Build windows
    wins_curve, wins_price = [], []
    for _, g in df.groupby(df["timestamp"].dt.date):
        close = g["close"].astype(float).reset_index(drop=True)
        for w in make_windows(close, L=L, S=S):
            w = w.astype(float)
            wins_price.append(w)
            wins_curve.append(normalize_window_returns_z(w))

    # 4) Persist windows
    import pickle
    with open(ARTIFACTS / "windows.pkl", "wb") as f:
        pickle.dump(wins_curve, f)
    with open(ARTIFACTS / "windows_price.pkl", "wb") as f:
        pickle.dump(wins_price, f)

    print(
        f"Windowing done. Curves -> {ARTIFACTS/'windows.pkl'} (n={len(wins_curve)}), "
        f"Prices -> {ARTIFACTS/'windows_price.pkl'} (n={len(wins_price)})"
    )

    # 5) Coverage figure + CSV
    df5i = df.set_index("timestamp").sort_index()
    daily_counts = df5i["close"].groupby(pd.Grouper(freq="1D")).count()

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    df5i["close"].plot(ax=ax[0], lw=1.1)
    ax[0].set_title(f"Figure 5.1a – 5-minute coverage (close & bars/day) [{ASSET}]", loc="left")
    ax[0].set_ylabel("Close")
    ax[0].grid(alpha=0.25)

    daily_counts.plot(kind="line", ax=ax[1], lw=1.2)
    ax[1].axhline(THEO_BARS_PER_DAY, ls="--", lw=1.0)
    ax[1].set_ylabel("Bars per day")
    ax[1].set_xlabel("")
    ax[1].grid(alpha=0.25)

    locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)
    plt.setp(ax[1].get_xticklabels(), rotation=25, ha="right")

    plt.tight_layout()
    out_cov = FIGDIR / f"fig_5_1a_coverage_{ASSET}.png"
    plt.savefig(out_cov, dpi=200)
    plt.close()

    # Coverage CSV
    desc = {
        "asset": [ASSET],
        "tz": [TZ],
        "window_L": [L],
        "window_S": [S],
        "n_days": [int(daily_counts.shape[0])],
        "bars_5min_total": [int(daily_counts.sum())],
        "bars_per_day_mean": [float(daily_counts.mean())],
        "bars_per_day_std": [float(daily_counts.std(ddof=1)) if daily_counts.shape[0] > 1 else 0.0],
        "n_windows": [int(len(wins_curve))],
        "date_min": [str(df5i.index.min())],
        "date_max": [str(df5i.index.max())],
    }
    pd.DataFrame(desc).to_csv(ARTIFACTS / f"desc_{ASSET}.csv", index=False)

    # 6) Intraday seasonality (robust)
    intra = df5i.copy()
    intra["ret"] = np.log(intra["close"]).diff()
    intra["clock"] = intra.index.time

    g = intra.groupby("clock", sort=True)
    vol_bps = g["ret"].apply(_robust_vol_bps)

    if "volume" in intra.columns:
        vol_median = g["volume"].median().fillna(0)
        vol_q25 = g["volume"].quantile(0.25).fillna(0)
        vol_q75 = g["volume"].quantile(0.75).fillna(0)
    else:
        vol_median = pd.Series(0, index=vol_bps.index)
        vol_q25 = pd.Series(0, index=vol_bps.index)
        vol_q75 = pd.Series(0, index=vol_bps.index)

    times = _time_index_from_groupby_clock(vol_bps.index)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(times, vol_bps.values, lw=1.2)
    ax1.set_ylabel("Vol (bps, 5-min)")
    ax1.set_title(f"Figure 5.1b – Intraday patterns: robust volatility & volume IQR [{ASSET}]", loc="left")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.fill_between(times, vol_q25.values, vol_q75.values, alpha=0.15, step="mid")
    ax2.plot(times, vol_median.values, lw=1.0, ls="--")
    ax2.set_ylabel("Volume (median, IQR)")

    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    out_u = FIGDIR / f"fig_5_1b_intraday_u_shape_{ASSET}.png"
    plt.savefig(out_u, dpi=220)
    plt.close()

    # 7) Return distribution (core + tails inset)
    rets = intra["ret"].dropna()

    moments = pd.DataFrame([{
        "asset": ASSET,
        "mean": float(rets.mean()),
        "std": float(rets.std(ddof=1)),
        "skew": float(rets.skew()),
        "kurt": float(rets.kurtosis()),
        "n": int(rets.size),
    }])
    moments.to_csv(ARTIFACTS / f"moments_{ASSET}.csv", index=False)

    q1, q99 = rets.quantile([0.01, 0.99])
    core = rets.clip(q1, q99)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(core.values, bins=60, density=True)
    ax.set_title(
        f"Figure 5.1c – Distribution of 5-min log-returns (core) [{ASSET}] "
        f"(μ={rets.mean():.4f}, σ={rets.std(ddof=1):.4f}, skew={rets.skew():.2f}, kurt={rets.kurtosis():.2f})",
        loc="left",
    )
    ax.set_xlabel("Return (winsorised 1–99%)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(ax, width="35%", height="70%", loc="upper right", borderpad=1.0)
    ax_ins.hist(rets.values, bins=120)
    ax_ins.set_yscale("log")
    ax_ins.set_title("Tails (log y)", fontsize=9)
    ax_ins.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    out_hist = FIGDIR / f"fig_5_1c_return_dist_{ASSET}.png"
    plt.savefig(out_hist, dpi=220)
    plt.close()

    # 8) Final log
    print(f"Saved windows to: {ARTIFACTS/'windows.pkl'} / {ARTIFACTS/'windows_price.pkl'}")
    print(f"Saved figures to: {out_cov.name}, {out_u.name}, {out_hist.name} in {FIGDIR}")
    print(f"Saved CSVs to: desc_{ASSET}.csv, moments_{ASSET}.csv in {ARTIFACTS}")


if __name__ == "__main__":
    main()
