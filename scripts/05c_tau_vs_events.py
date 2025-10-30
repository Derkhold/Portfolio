# --- Exemples ---
# python -m scripts.05c_tau_vs_events --metric softdtw --gamma 0.4 --tz Europe/Paris
# python -m scripts.05c_tau_vs_events --tz Europe/Paris --tag k8_tau60_99 --aggregate daily --smooth 5 --show-raw
# python -m scripts.05c_tau_vs_events --tz Europe/Paris --roll-csv artifacts/tau_c_rolling_k8_tau60_99.csv --ylim 0.9,1.0

import sys, argparse, pickle, glob, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cmorph.io import load_excel_ohlcv, sessionize

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
DATA = Path("data")


def infer_window_timestamps(xlsx_path: Path, tz="Europe/Paris", start="09:00", end="16:00") -> pd.DatetimeIndex:
    """
    Reconstruit les timestamps 'centre' des fenêtres utilisées dans 01_prepare.
    - Sessionize l'Excel (start/end paramétrables)
    - Lit ART/windows.pkl (N, L)
    - Place N starts linéaires sur [0, n-L] et prend le milieu de chaque fenêtre
    """
    df = load_excel_ohlcv(str(xlsx_path))
    df = sessionize(df, tz=tz, start=start, end=end)
    ts = pd.DatetimeIndex(df["timestamp"])
    n = len(ts)

    win_pkl = ART / "windows.pkl"
    if not win_pkl.exists():
        raise FileNotFoundError(f"{win_pkl} introuvable (génère-le via 01_prepare).")
    wins = pickle.load(open(win_pkl, "rb"))
    N = len(wins)
    if N == 0:
        raise RuntimeError("windows.pkl est vide.")
    L = len(wins[0])

    last_start = max(0, n - L)
    starts = np.round(np.linspace(0, last_start, N)).astype(int)
    mids = [ts[s + L // 2] for s in starts]
    return pd.DatetimeIndex(mids)


def resolve_rolling_csv(tag: str | None, roll_csv: str | None) -> Path:
    """Trouve le CSV tau_c_rolling à utiliser."""
    if roll_csv:
        p = Path(roll_csv)
        if not p.exists():
            raise FileNotFoundError(f"roll-csv not found: {p}")
        return p
    if tag:
        p = ART / f"tau_c_rolling_{tag}.csv"
        if p.exists(): return p
        raise FileNotFoundError(f"CSV with tag not found: {p}")
    default_csv = ART / "tau_c_rolling.csv"
    if default_csv.exists():
        return default_csv
    candidates = sorted(glob.glob(str(ART / "tau_c_rolling*.csv")), key=os.path.getmtime)
    if candidates:
        print(f"ℹ️ no explicit tag/roll-csv provided — using latest: {candidates[-1]}")
        return Path(candidates[-1])
    raise FileNotFoundError("No tau_c_rolling CSV found in artifacts/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Entrées
    ap.add_argument("--events", type=str, default=str(DATA / "events.csv"),
                    help="CSV avec colonnes: timestamp,event (timestamps en UTC).")
    ap.add_argument("--xlsx", type=str, default=str(DATA / "raw/ES1.xlsx"),
                    help="Chemin vers l'Excel OHLCV utilisé en 01_prepare.")
    ap.add_argument("--tz", type=str, default="Europe/Paris", help="Timezone d'affichage.")

    # Paramètres informatifs (pour annoter le titre)
    ap.add_argument("--metric", type=str, default=None, help="Nom de la métrique (ex: softdtw).")
    ap.add_argument("--gamma", type=float, default=None, help="Paramètre gamma pour Soft-DTW.")

    # Sélection du CSV 'tau_c_rolling'
    ap.add_argument("--tag", type=str, default="", help="Suffixe du fichier tau_c_rolling (ex: k8_tau60_99).")
    ap.add_argument("--roll-csv", type=str, default="", help="Chemin direct vers un tau_c_rolling*.csv.")

    # Heures de session
    ap.add_argument("--start", type=str, default="09:00", help="Heure de début de session (HH:MM).")
    ap.add_argument("--end", type=str, default="16:00", help="Heure de fin de session (HH:MM).")

    # Aggregation / smoothing / plot
    ap.add_argument("--aggregate", choices=["none", "daily"], default="daily",
                    help="Agréger la série (médiane par jour) pour éviter l'effet 'peigne'.")
    ap.add_argument("--smooth", type=int, default=0,
                    help="Fenêtre (en jours) pour lisser la série agrégée (0=pas de lissage).")
    ap.add_argument("--show-raw", action="store_true",
                    help="Superposer les points bruts (scatter alpha=0.3).")
    ap.add_argument("--ylim", type=str, default="", help='Bornes y, ex: "0.90,1.00".')

    # Sortie & debug
    ap.add_argument("--out", type=str, default=str(ART / "tau_c_vs_events.png"),
                    help="Chemin de sortie de la figure.")
    ap.add_argument("--debug", action="store_true", help="Logs supplémentaires.")

    args = ap.parse_args()

    # Résolution du CSV rolling
    roll_path = resolve_rolling_csv(args.tag.strip() or None, args.roll_csv.strip() or None)

    # τ_c rolling
    roll = pd.read_csv(roll_path)
    if not {"center", "tau_c"}.issubset(roll.columns):
        raise RuntimeError(f"{roll_path} ne contient pas les colonnes requises: center, tau_c")
    centers = roll["center"].to_numpy(dtype=float)

    # timestamps 'milieu' interpolés
    mid_ts = infer_window_timestamps(Path(args.xlsx), tz=args.tz, start=args.start, end=args.end)
    N = len(mid_ts)
    if args.debug:
        print(f"N(windows) = {N} | min(mid_ts) = {mid_ts.min()} | max(mid_ts) = {mid_ts.max()}")
    mid_ns = mid_ts.view("int64")
    idx_base = np.arange(N, dtype=float)
    centers_ns = np.interp(centers, idx_base, mid_ns).astype("int64")
    roll["timestamp"] = pd.to_datetime(centers_ns, utc=True)
    roll = roll.sort_values("timestamp")

    # événements (tolérant à l'absence de fichier)
    try:
        ev = pd.read_csv(args.events)
        if not {"timestamp", "event"}.issubset(ev.columns):
            print(f"⚠️ events file {args.events} missing required columns; ignoring.")
            ev = pd.DataFrame(columns=["timestamp", "event"])
        else:
            ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True).dt.tz_convert(args.tz)
    except FileNotFoundError:
        print(f"⚠️ events file not found: {args.events} — plotting without event markers.")
        ev = pd.DataFrame(columns=["timestamp", "event"])

    # Agrégation et lissage
    agg_df = None
    if args.aggregate == "daily":
        roll["date"] = roll["timestamp"].dt.tz_convert(args.tz).dt.normalize()
        agg_df = roll.groupby("date", as_index=False)["tau_c"].median()
        if args.smooth and args.smooth > 1:
            agg_df["tau_c_smooth"] = (
                agg_df["tau_c"]
                .rolling(args.smooth, min_periods=max(1, args.smooth // 2), center=True)
                .median()
            )

    # plot
    plt.figure(figsize=(8.8, 4.2))

    if args.show_raw:
        # nuage de points bruts
        plt.scatter(roll["timestamp"], roll["tau_c"], s=10, alpha=0.3, label="raw")

    if agg_df is not None:
        plt.plot(agg_df["date"], agg_df["tau_c"], linewidth=1.8, marker="o", label="daily median")
        if "tau_c_smooth" in agg_df:
            plt.plot(agg_df["date"], agg_df["tau_c_smooth"], linewidth=2.2, linestyle="--", label=f"smoothed ({args.smooth}d)")
    else:
        # pas d'agrégation
        plt.plot(roll["timestamp"], roll["tau_c"], linewidth=1.6, marker="o", label=r"$\tau_c$")

    for _, r in ev.iterrows():
        plt.axvline(r["timestamp"], alpha=0.35, linewidth=1.2)
        try:
            ymax = float(roll["tau_c"].max())
            plt.text(r["timestamp"], ymax * 1.02, str(r["event"]),
                     rotation=90, va="bottom", ha="center", fontsize=8, alpha=0.7)
        except Exception:
            pass

    # Titre
    subtitle = []
    if args.metric: subtitle.append(f"metric={args.metric}")
    if args.gamma is not None: subtitle.append(r"$\gamma$=" + str(args.gamma))
    title = r"$\tau_c$ (rolling) and macro events"
    if subtitle: title += " — " + ", ".join(subtitle)
    plt.title(title)

    plt.ylabel(r"$\tau_c$")
    if args.ylim:
        try:
            lo, hi = [float(x) for x in args.ylim.split(",")]
            plt.ylim(lo, hi)
        except Exception:
            pass
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print("📈 saved:", out_path, "| source:", roll_path)
