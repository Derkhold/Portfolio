from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
from datetime import timedelta
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
)

# ------------------------------------------------------------
# Helpers internes
# ------------------------------------------------------------

# Variantes de noms de colonnes possibles (FR/EN/Bbg)
_COL_ALIASES: Dict[str, Iterable[str]] = {
    "timestamp": ["timestamp", "time", "date", "datetime", "Date", "DATE", "DATETIME", "Date/Heure"],
    "open":      ["open", "Open", "OPEN", "Ouv", "Ouvt", "Ouverture", "O", "PX_OPEN"],
    "high":      ["high", "High", "HIGH", "'+Haut", "+Haut", "Plus Haut", "Haut", "H", "PX_HIGH"],
    "low":       ["low", "Low", "LOW", "'+Bas", "+Bas", "Plus Bas", "Bas", "L", "PX_LOW"],
    "close":     ["close", "Close", "CLOSE", "Clôture", "Cloture", "Dernier", "C", "PX_LAST", "PX_CLOSE"],
    "volume":    ["volume", "Volume", "VOL", "Qty", "Quantité", "Size", "SIZE", "PX_VOLUME"],
}

def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def _read_any_ohlc(path: Path) -> pd.DataFrame:
    """
    Lit CSV (UTF-8 puis latin-1) ou Excel (.xlsx/.xls).
    Ne parse PAS encore les dates; la normalisation s'en charge.
    """
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def _normalize_numeric(series: pd.Series) -> pd.Series:
    """
    Convertit une série texte/nombre en float :
    - remplace virgule décimale par point,
    - enlève espaces (y compris insécables),
    - remplace '--' / vide par NaN.
    """
    s = (
        series.astype(str)
        .str.replace("\u00A0", "", regex=False)  # espace insécable
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("--", "", regex=False)
        .str.strip()
    )
    s = s.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

def _standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme colonnes vers ['timestamp','open','high','low','close','volume'],
    nettoie nombres et parse les dates (robuste tz).
    """
    df = raw.copy()

    # 1) Map des colonnes
    mapping = {}
    for std_name, aliases in _COL_ALIASES.items():
        found = _find_col(df, aliases)
        if found is not None:
            mapping[found] = std_name
    df = df.rename(columns=mapping)

    missing = [k for k in ["timestamp", "open", "high", "low", "close"] if k not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes OHLC manquantes après normalisation: {missing}. "
            f"Colonnes lues: {list(raw.columns)}"
        )

    # 2) Nettoyage numérique
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = _normalize_numeric(df[c])

    # 3) Parsing des timestamps (robuste tz)
    ts = df["timestamp"]
    if not is_datetime64_any_dtype(ts):
        # Excel/CSV FR → dayfirst probable ; force UTC pour éviter ambigüité,
        # on convertira le fuseau plus tard dans sessionize().
        df["timestamp"] = pd.to_datetime(ts, dayfirst=True, errors="coerce", utc=True)
    else:
        # Déjà datetime ; si tz-naïf → localise en UTC
        if not is_datetime64tz_dtype(ts):
            df["timestamp"] = pd.to_datetime(ts, errors="coerce", utc=True)

    # 4) Tri & drop des timestamps invalides
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 5) Volume facultatif
    if "volume" not in df.columns:
        df["volume"] = np.nan

    return df[["timestamp", "open", "high", "low", "close", "volume"]]

# ------------------------------------------------------------
# API publique
# ------------------------------------------------------------

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """
    Charge un fichier OHLCV (CSV ou Excel) et renvoie un DataFrame standardisé :
    colonnes = ['timestamp','open','high','low','close','volume'], timestamp tz-aware (UTC).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier OHLC introuvable: {p}")
    raw = _read_any_ohlc(p)
    df = _standardize_columns(raw)
    # dédoublonnage éventuel
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df

def sessionize(
    df: pd.DataFrame,
    tz: str = "Europe/Paris",
    start: str = "09:30",
    end: str = "16:00",
) -> pd.DataFrame:
    """
    Convertit en fuseau 'tz', filtre jours ouvrés et plage horaire [start, end] (inclus).
    Accepte timestamp naïf ou tz-aware.
    """
    if "timestamp" not in df.columns:
        raise ValueError("sessionize: colonne 'timestamp' absente")

    out = df.copy()

    # S'assurer d'un dtype datetime
    if not is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    # Localiser / convertir fuseau
    if is_datetime64tz_dtype(out["timestamp"]):
        out["timestamp"] = out["timestamp"].dt.tz_convert(tz)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_localize("UTC").dt.tz_convert(tz)

    # Index temporel & filtres
    out = out.set_index("timestamp").sort_index()

    # Lundi(0)–Vendredi(4)
    out = out[out.index.dayofweek < 5]

    # Plage horaire (pandas >=1.4 : paramètres nommés)
    out = out.between_time(start_time=start, end_time=end)

    return out.reset_index()

def resample_ohlc(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """
    Resample en OHLCV sur l'index temporel.
      - open: first, high: max, low: min, close: last, volume: sum (si présent)
    """
    need = {"timestamp", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"resample_ohlc: colonnes requises manquantes: {sorted(need - set(df.columns))}")

    x = df.set_index("timestamp").sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in x.columns:
        agg["volume"] = "sum"

    out = x.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out.reset_index()

# ------------------------------------------------------------
# Compat (ancienne fonction)
# ------------------------------------------------------------

def load_excel_ohlcv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Compat : lit Excel/CSV via load_ohlcv(), puis convertit vers 'tz' si fourni.
    """
    df = load_ohlcv(path)
    if tz:
        if not is_datetime64tz_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    return df
