# scripts/02_roughness.py
# Génère :
#   - artifacts/roughness.csv (si absent)
#   - artifacts/figures/fig_5_1d_roughness_joint_ES1.png
#   - artifacts/figures/fig_5_1e_alpha_by_clock_ES1.png
#   - artifacts/table_5_1b_roughness_ES1.csv

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
from cmorph.windows import make_windows, normalize_window_returns_z
from cmorph.roughness import dfa_alpha, higuchi_fd

# ----------------------- PARAMÈTRES À AJUSTER -----------------------
RAW_PATH = "data/raw/ES1I.xlsx"     # fichier OHLCV utilisé pour 01_prepare
ASSET    = "ES1"                     # étiquette à insérer dans les sorties
TZ       = "Europe/Paris"
START    = "09:30"
END      = "16:00"
L, S     = 12, 1                     # fenêtres de 12x5min (≈1h), stride=1
FIGDIR   = Path("artifacts/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------------

# ===== (0) Utilitaires =====
def _detect_step_minutes(ts: pd.Series) -> int:
    if len(ts) < 3:
        return 30
    return int(np.median(np.diff(ts.values).astype("timedelta64[m]").astype(int)))

def _rebuild_window_end_times(df_5min: pd.DataFrame, L: int, S: int) -> list[pd.Timestamp]:
    """Reconstruit la liste des timestamps de fin de fenêtre, jour par jour, comme dans 01_prepare."""
    ends = []
    for _, g in df_5min.groupby(df_5min["timestamp"].dt.date):
        idx = g["timestamp"].to_list()
        # nombre de fenêtres pour ce jour
        n = max(0, (len(idx) - L) // S + 1)
        for j in range(n):
            end_pos = j*S + L - 1
            ends.append(idx[end_pos])
    return ends

def _window_realized_vol(paths: list[np.ndarray]) -> np.ndarray:
    """Volatilité intra-fenêtre (écart-type des log-returns) pour chaque fenêtre de prix."""
    out = np.empty(len(paths))
    for i, p in enumerate(paths):
        r = np.diff(np.log(p.astype(float)))
        out[i] = np.nanstd(r)
    return out

# ===== (1) Charger les fenêtres et roughness (ou les calculer) =====
print("🔧 Chargement des fenêtres…")
wins_curve = pickle.load(open("artifacts/windows.pkl","rb"))          # profils normalisés (shape)
wins_price = pickle.load(open("artifacts/windows_price.pkl","rb"))    # prix (pour vol intra)

rough_csv = Path("artifacts/roughness.csv")
if rough_csv.exists():
    print("✅ artifacts/roughness.csv trouvé (on réutilise).")
    rough = pd.read_csv(rough_csv)
else:
    print("🧮 Calcul des roughness (α_DFA, D_Higuchi)…")
    rows = []
    for i, w in enumerate(wins_curve):
        rows.append({"i": i, "alpha_dfa": dfa_alpha(w), "D_higuchi": higuchi_fd(w)})
    rough = pd.DataFrame(rows)
    rough.to_csv(rough_csv, index=False)
    print("✅ artifacts/roughness.csv écrit.")

# Sanity check tailles
n_wins = len(wins_curve)
assert len(wins_price) == n_wins, "windows.pkl et windows_price.pkl ont des tailles différentes"
assert len(rough) == n_wins, "roughness.csv n'a pas le même nombre de fenêtres"

# ===== (2) Relecture OHLCV et reconstruction timestamps de fin =====
print("📂 Relecture OHLCV pour reconstruire les horodatages de fin de fenêtre…")
if not Path(RAW_PATH).exists():
    print(f"⚠️ Fichier OHLCV introuvable : {RAW_PATH}. "
          f"Figure 5.1e (par heure) sera sautée.")
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
    # reconstruire fins de fenêtre
    ends = _rebuild_window_end_times(df, L=L, S=S)
    if len(ends) != n_wins:
        print(f"⚠️ Mismatch #windows: rough={n_wins}, ends={len(ends)}. "
              f"Figure 5.1e sera sautée.")
        have_clock = False
    else:
        have_clock = True
        ends = pd.to_datetime(pd.Series(ends))
        rough["window_end"] = ends
        rough["clock"] = rough["window_end"].dt.strftime("%H:%M")

# ===== (3) Figures =====

# --- Figure 5.1d : Joint density α vs D (hexbin) ---
print("📈 Figure 5.1d – Joint density (α_DFA vs D_Higuchi)…")
fig, ax = plt.subplots(figsize=(6.8, 5.0))
hb = ax.hexbin(
    rough["alpha_dfa"].values,
    rough["D_higuchi"].values,
    gridsize=40, mincnt=3, linewidths=0.2
)
ax.set_xlabel(r"DFA slope $\alpha_{\mathrm{DFA}}$")
ax.set_ylabel(r"Higuchi fractal dimension $D_{\mathrm{H}}$")
ax.set_title(f"Figure 5.1d – Roughness joint density ({ASSET})", loc="left")
cb = fig.colorbar(hb, ax=ax); cb.set_label("count")
# Références théoriques (benchmark RW)
ax.axvline(0.5, ls="--", lw=0.8)
ax.axhline(1.5, ls="--", lw=0.8)
plt.tight_layout()
f1 = FIGDIR / f"fig_5_1d_roughness_joint_{ASSET}.png"
plt.savefig(f1, dpi=200)
plt.close()
print(f"   ↳ {f1}")

# --- Figure 5.1e : Médiane α par heure (si horodatages dispo) ---
# --- Figure 5.1e : Médiane α par heure (si horodatages dispo) ---
if have_clock:
    print("📈 Figure 5.1e – α_DFA par heure de la journée…")
    g = rough.groupby("clock")["alpha_dfa"]
    med = g.median()
    q25 = g.quantile(0.25)
    q75 = g.quantile(0.75)

    # ordonner par heure ("HH:MM")
    order = sorted(med.index, key=lambda s: pd.to_datetime(s).time())
    med = med.reindex(order)
    q25 = q25.reindex(order)
    q75 = q75.reindex(order)

    # minutes since midnight -> numeric x-axis
    times = [pd.to_datetime(s).time() for s in med.index]
    xmins = np.array([t.hour * 60 + t.minute for t in times])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(xmins, med.values, lw=1.4)
    ax.fill_between(xmins, q25.values, q75.values, alpha=0.15)
    ax.set_title(f"Figure 5.1e – Intraday median α_DFA with IQR ({ASSET})", loc="left")
    ax.set_ylabel(r"Median $\alpha_{\mathrm{DFA}}$ (per 5-min slot)")
    ax.set_xlabel("Intraday clock time")

    # nice hour ticks every ~60 minutes, with HH:MM labels
    if len(xmins) > 0:
        tick_step = 60  # every hour
        xticks = np.arange(xmins.min() // tick_step * tick_step,
                           xmins.max() + 1, tick_step, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{m // 60:02d}:{m % 60:02d}" for m in xticks], rotation=45)

    plt.tight_layout()
    f2 = FIGDIR / f"fig_5_1e_alpha_by_clock_{ASSET}.png"
    plt.savefig(f2, dpi=200)
    plt.close()
    print(f"   ↳ {f2}")
else:
    print("⏭️ Figure 5.1e sautée (horodatages de fin de fenêtre indisponibles).")
