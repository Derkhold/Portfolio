# --- python -m scripts.04e_plot_colored_session --labels artifacts/labels_softdtw_K2.csv
import sys, argparse, pickle
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
FIGDIR = ART / "figures"; FIGDIR.mkdir(exist_ok=True)

def load_labels(path):
    L = pd.read_csv(path)
    return (L["label"].to_numpy().astype(int)
            if "label" in L.columns else L.iloc[:, -1].to_numpy().astype(int))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ART / "labels_softdtw_K2.csv"))
    ap.add_argument("--wins_price", default=str(ART / "windows_price.pkl"))
    ap.add_argument("--starts_csv", default=str(ART / "proxies_window.csv"),
                    help="optionnel : pour les timestamps; doit contenir une colonne 'start'")
    ap.add_argument("--day_index", type=int, default=0,
                    help="jour à afficher (0 = premier jour avec windows)")
    ap.add_argument("--out", default=str(FIGDIR / "fig_5_2g_colored_session.png"))
    args = ap.parse_args()

    # fenêtres de prix (liste d'array de taille L)
    wins_price = pickle.load(open(args.wins_price, "rb"))
    L = len(wins_price[0])

    # labels
    labs = load_labels(args.labels)
    if len(labs) != len(wins_price):
        print("⚠️  labels et fenêtres de tailles différentes; on tronque au min.")
    n = min(len(labs), len(wins_price))
    labs = labs[:n]; wins_price = wins_price[:n]

    # indexe par journée : N_win_per_day ~= (session_length / stride)
    # On essaie de le déduire depuis 01_prepare (bars/session) : ~78 fenêtres par séance (L=12, stride=1, 5-min)
    # Approche robuste : détecter des ruptures par 'reset' d'horodatage si disponible.
    starts = None
    try:
        dfp = pd.read_csv(args.starts_csv, parse_dates=["start"])
        if "start" in dfp.columns and len(dfp) >= n:
            starts = pd.to_datetime(dfp["start"].iloc[:n])
    except Exception:
        pass

    if starts is not None:
        # regroupe par date
        dates = starts.dt.date.to_numpy()
        unique = np.unique(dates)
        if args.day_index >= len(unique):
            args.day_index = 0
        sel_date = unique[args.day_index]
        mask = dates == sel_date
        series = np.concatenate([np.asarray(w) for w, m in zip(wins_price, mask) if m])
        labels = np.concatenate([np.full(len(wins_price[0]), l)
                                 for l, m in zip(labs, mask) if m])
        x = np.arange(series.size)
        xtick_label = pd.to_datetime(starts[mask]).min().strftime("%Y-%m-%d")
        xlab = f"time (5-min steps) — {xtick_label}"
    else:
        # fallback : on prend un bloc contigu brut (heuristique 78 fenêtres ~ 1 séance)
        WPD = 78  # windows per day approx.
        i0 = args.day_index * WPD
        i1 = min((args.day_index + 1) * WPD, n)
        series = np.concatenate([np.asarray(w) for w in wins_price[i0:i1]])
        labels = np.concatenate([np.full(len(wins_price[0]), l) for l in labs[i0:i1]])
        x = np.arange(series.size)
        xlab = "window step (5-min)"

    # normalise une fois (retours cumulés centrés) pour homogénéiser l’échelle
    series = (series - np.nanmean(series)) / (np.nanstd(series) + 1e-12)

    # trace par segments colorés selon le label
    plt.figure(figsize=(9.5, 3.2))
    palette = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#9467bd"}
    cur = 0
    while cur < len(series):
        lab = int(labels[cur])
        j = cur + 1
        while j < len(series) and int(labels[j]) == lab:
            j += 1
        plt.plot(x[cur:j], series[cur:j], linewidth=1.4,
                 color=palette.get(lab, "0.5"))
        cur = j

    plt.title("Figure 5.2g – Real trajectory colored by cluster (illustrative session)", loc="left")
    plt.xlabel(xlab); plt.ylabel("Normalized price (window concat.)")
    plt.grid(alpha=0.2)
    # petite légende
    handles = [plt.Line2D([0],[0], color=c, lw=2, label=f"Cluster {k}")
               for k, c in palette.items() if k in np.unique(labels)]
    plt.legend(handles=handles, loc="upper left", frameon=False, ncol=3)
    plt.tight_layout(); plt.savefig(args.out, dpi=200); plt.close()
    print("📈 saved:", args.out)

if __name__ == "__main__":
    main()
