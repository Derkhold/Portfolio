# scripts/02_roughness.py
# Compute roughness metrics and figures (Chapter 5 §5.1)
# Outputs:
#   - artifacts/roughness.csv
#   - artifacts/figures/fig_5_1d_roughness_joint_<ASSET>.png
#   - artifacts/figures/fig_5_1e_alpha_by_clock_<ASSET>.png

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cmorph.io import load_excel_ohlcv, sessionize
from cmorph.windows import make_windows, normalize_window_returns_z  # kept for parity with 01
from cmorph.fractals import dfa_alpha, higuchi_fd

# --- Config ------------------------------------------------------------------

RAW_PATH = "data/raw/ES1I.xlsx"
ASSET = "ES1"
TZ = "Europe/Paris"
START = "09:30"
END = "16:00"
L, S = 12, 1
FIGDIR = Path("artifacts/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


# --- Helpers -----------------------------------------------------------------

def _detect_step_minutes(ts: pd.Series) -> int:
    """Median spacing (minutes) between consecutive timestamps."""
    if len(ts) < 3:
        return 30
    return int(np.median(np.diff(ts.values).astype("timedelta64[m]").astype(int)))


def _rebuild_window_end_times(df_5min: pd.DataFrame, L: int, S: int) -> list[pd.Timestamp]:
    """Rebuild window end timestamps day-by-day to match 01_prepare."""
    ends: list[pd.Timestamp] = []
    for _, g in df_5min.groupby(df_5min["timestamp"].dt.date):
        idx = g["timestamp"].to_list()
        n = max(0, (len(idx) - L) // S + 1)
        for j in range(n):
            end_pos = j * S + L - 1
            ends.append(idx[end_pos])
    return ends


def _window_realized_vol(paths: list[np.ndarray]) -> np.ndarray:
    """In-window realized vol proxy (std of log-returns)."""
    out = np.empty(len(paths))
    for i, p in enumerate(paths):
        r = np.diff(np.log(p.astype(float)))
        out[i] = np.nanstd(r)
    return out


# --- Load windows and roughness ---------------------------------------------

print("Loading windows...")
wins_curve = pickle.load(open("artifacts/windows.pkl", "rb"))
wins_price = pickle.load(open("artifacts/windows_price.pkl", "rb"))

rough_csv = Path("artifacts/roughness.csv")
if rough_csv.exists():
    print("Found artifacts/roughness.csv (reuse).")
    rough = pd.read_csv(rough_csv)
else:
    print("Computing roughness (DFA α, Higuchi D)...")
    rows = [{"i": i, "alpha_dfa": dfa_alpha(w), "D_higuchi": higuchi_fd(w)} for i, w in enumerate(wins_curve)]
    rough = pd.DataFrame(rows)
    rough.to_csv(rough_csv, index=False)
    print("Wrote artifacts/roughness.csv.")

# sanity checks
n_wins = len(wins_curve)
assert len(wins_price) == n_wins, "windows.pkl and windows_price.pkl sizes differ"
assert len(rough) == n_wins, "roughness.csv length does not match windows"


# --- Re-read OHLCV and rebuild window-end timestamps ------------------------

print("Re-reading OHLCV to rebuild window end timestamps...")
if not Path(RAW_PATH).exists():
    print(f"OHLCV file not found: {RAW_PATH}. Skipping Figure 5.1e.")
    have_clock = False
else:
    df = load_excel_ohlcv(RAW_PATH)
    df = sessionize(df, tz=TZ, start=START, end=END)
    step_min = _detect_step_minutes(df["timestamp"])
    if step_min <= 10:
        df = (
            df.set_index("timestamp")
              .resample("5min")
              .last()
              .dropna(subset=["close"])
              .reset_index()
        )
    ends = _rebuild_window_end_times(df, L=L, S=S)
    if len(ends) != n_wins:
        print(f"Mismatch #windows: rough={n_wins}, ends={len(ends)}. Skipping Figure 5.1e.")
        have_clock = False
    else:
        have_clock = True
        ends = pd.to_datetime(pd.Series(ends))
        rough["window_end"] = ends
        rough["clock"] = rough["window_end"].dt.strftime("%H:%M")


# --- Figures -----------------------------------------------------------------

# Figure 5.1d: joint density α vs D (hexbin)
print("Figure 5.1d – joint density (α_DFA vs D_Higuchi)...")
fig, ax = plt.subplots(figsize=(6.8, 5.0))
hb = ax.hexbin(
    rough["alpha_dfa"].values,
    rough["D_higuchi"].values,
    gridsize=40,
    mincnt=3,
    linewidths=0.2,
)
ax.set_xlabel(r"DFA slope $\alpha_{\mathrm{DFA}}$")
ax.set_ylabel(r"Higuchi fractal dimension $D_{\mathrm{H}}$")
ax.set_title(f"Figure 5.1d – Roughness joint density ({ASSET})", loc="left")
cb = fig.colorbar(hb, ax=ax)
cb.set_label("count")
ax.axvline(0.5, ls="--", lw=0.8)
ax.axhline(1.5, ls="--", lw=0.8)
plt.tight_layout()
f1 = FIGDIR / f"fig_5_1d_roughness_joint_{ASSET}.png"
plt.savefig(f1, dpi=200)
plt.close()
print(f" -> {f1}")

# Figure 5.1e: median α by clock (if available)
if have_clock:
    print("Figure 5.1e – DFA α by clock...")
    g = rough.groupby("clock")["alpha_dfa"]
    med = g.median()
    q25 = g.quantile(0.25)
    q75 = g.quantile(0.75)

    order = sorted(med.index, key=lambda s: pd.to_datetime(s).time())
    med = med.reindex(order)
    q25 = q25.reindex(order)
    q75 = q75.reindex(order)

    times = [pd.to_datetime(s).time() for s in med.index]
    xmins = np.array([t.hour * 60 + t.minute for t in times])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(xmins, med.values, lw=1.4)
    ax.fill_between(xmins, q25.values, q75.values, alpha=0.15)
    ax.set_title(f"Figure 5.1e – Intraday median α_DFA with IQR ({ASSET})", loc="left")
    ax.set_ylabel(r"Median $\alpha_{\mathrm{DFA}}$")
    ax.set_xlabel("Intraday time")

    if len(xmins) > 0:
        tick_step = 60
        xticks = np.arange(xmins.min() // tick_step * tick_step, xmins.max() + 1, tick_step, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{m // 60:02d}:{m % 60:02d}" for m in xticks], rotation=45)

    plt.tight_layout()
    f2 = FIGDIR / f"fig_5_1e_alpha_by_clock_{ASSET}.png"
    plt.savefig(f2, dpi=200)
    plt.close()
    print(f" -> {f2}")
else:
    print("Figure 5.1e skipped (window-end timestamps unavailable).")
