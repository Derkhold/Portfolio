# scripts/04e_plot_colored_session.py
# Plot a real intraday trajectory, colored by cluster labels (per 5-min window)

import sys
import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
FIGDIR = ART / "figures"; FIGDIR.mkdir(exist_ok=True)


def load_labels(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "label" in df.columns:
        return df["label"].astype(int).to_numpy()
    return df.iloc[:, -1].astype(int).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ART / "labels_softdtw_K2.csv"))
    ap.add_argument("--wins_price", default=str(ART / "windows_price.pkl"))
    ap.add_argument(
        "--starts_csv",
        default=str(ART / "proxies_window.csv"),
        help="Optional: CSV with a 'start' column to align windows by date.",
    )
    ap.add_argument("--day_index", type=int, default=0, help="Which day to plot (0-based).")
    ap.add_argument("--out", default=str(FIGDIR / "fig_5_2g_colored_session.png"))
    args = ap.parse_args()

    # windows (prices) and labels
    wins_price: list[np.ndarray] = pickle.load(open(args.wins_price, "rb"))
    L = len(wins_price[0]) if wins_price else 0
    labels = load_labels(args.labels)

    if len(labels) != len(wins_price):
        print("Labels and windows have different lengths; truncating to min length.")
    n = min(len(labels), len(wins_price))
    labels = labels[:n]
    wins_price = wins_price[:n]

    # Try to group windows by day using a 'start' timestamp (if provided)
    starts = None
    try:
        dfp = pd.read_csv(args.starts_csv, parse_dates=["start"])
        if "start" in dfp.columns and len(dfp) >= n:
            starts = pd.to_datetime(dfp["start"].iloc[:n])
    except Exception:
        pass

    if starts is not None:
        dates = starts.dt.date.to_numpy()
        uniq = np.unique(dates)
        if args.day_index >= len(uniq):
            args.day_index = 0
        sel_date = uniq[args.day_index]
        mask = dates == sel_date

        series = np.concatenate([np.asarray(w) for w, m in zip(wins_price, mask) if m])
        seg_labels = np.concatenate([np.full(L, lab) for lab, m in zip(labels, mask) if m])

        x = np.arange(series.size)
        xlab = f"time (5-min steps) — {pd.to_datetime(starts[mask]).min().strftime('%Y-%m-%d')}"
    else:
        # Fallback: contiguous block approximating one session
        WPD = 78  # ≈ number of 5-min bars per session; adjust if needed
        i0 = args.day_index * WPD
        i1 = min((args.day_index + 1) * WPD, n)

        series = np.concatenate([np.asarray(w) for w in wins_price[i0:i1]])
        seg_labels = np.concatenate([np.full(L, lab) for lab in labels[i0:i1]])

        x = np.arange(series.size)
        xlab = "window step (5-min)"

    # Normalize the concatenated series (robust visual scale)
    series = (series - np.nanmean(series)) / (np.nanstd(series) + 1e-12)

    # Plot segments with color by label
    plt.figure(figsize=(9.5, 3.2))
    palette = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}

    cur = 0
    while cur < len(series):
        lab = int(seg_labels[cur])
        j = cur + 1
        while j < len(series) and int(seg_labels[j]) == lab:
            j += 1
        plt.plot(x[cur:j], series[cur:j], linewidth=1.4, color=palette.get(lab, "0.5"))
        cur = j

    plt.title("Figure 5.2g – Real trajectory colored by cluster", loc="left")
    plt.xlabel(xlab)
    plt.ylabel("Normalized price (concatenated windows)")
    plt.grid(alpha=0.2)

    # Legend (only for labels present)
    present = np.unique(seg_labels)
    handles = [
        plt.Line2D([0], [0], color=palette.get(int(k), "0.5"), lw=2, label=f"Cluster {int(k)}")
        for k in present
    ]
    if handles:
        plt.legend(handles=handles, loc="upper left", frameon=False, ncol=min(3, len(handles)))

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
