from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype


# --- Internal helpers --------------------------------------------------------

# Flexible column aliases (FR/EN/Bloomberg)
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
    Read CSV (UTF-8 then latin-1 fallback) or Excel (.xlsx/.xls).
    Date parsing is deferred to normalization.
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
    Convert text/number series to float:
    - replace comma decimal with dot
    - strip spaces (incl. non-breaking)
    - map empty / '--' to NaN
    """
    s = (
        series.astype(str)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("--", "", regex=False)
        .str.strip()
    )
    s = s.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rename to ['timestamp','open','high','low','close','volume'],
    clean numerics, and parse timestamps (UTC tz-aware).
    """
    df = raw.copy()

    # Map columns
    mapping: Dict[str, str] = {}
    for std_name, aliases in _COL_ALIASES.items():
        found = _find_col(df, aliases)
        if found is not None:
            mapping[found] = std_name
    df = df.rename(columns=mapping)

    missing = [k for k in ["timestamp", "open", "high", "low", "close"] if k not in df.columns]
    if missing:
        raise ValueError(
            f"Missing OHLC columns after normalization: {missing}. "
            f"Read columns: {list(raw.columns)}"
        )

    # Numeric cleanup
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = _normalize_numeric(df[c])

    # Timestamp parsing (robust tz)
    ts = df["timestamp"]
    if not is_datetime64_any_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, dayfirst=True, errors="coerce", utc=True)
    else:
        if not is_datetime64tz_dtype(ts):
            df["timestamp"] = pd.to_datetime(ts, errors="coerce", utc=True)

    # Sort and drop invalid timestamps
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Optional volume
    if "volume" not in df.columns:
        df["volume"] = np.nan

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# --- Public API --------------------------------------------------------------

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """
    Load OHLCV (CSV/Excel) and return standardized DataFrame with
    ['timestamp','open','high','low','close','volume'], timestamp tz-aware (UTC).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OHLC file not found: {p}")
    raw = _read_any_ohlc(p)
    df = _standardize_columns(raw)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df


def sessionize(
    df: pd.DataFrame,
    tz: str = "Europe/Paris",
    start: str = "09:30",
    end: str = "16:00",
) -> pd.DataFrame:
    """
    Convert to timezone 'tz', keep weekdays, and filter intraday range [start, end] (inclusive).
    Accepts naive or tz-aware timestamps.
    """
    if "timestamp" not in df.columns:
        raise ValueError("sessionize: 'timestamp' column missing")

    out = df.copy()

    if not is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    if is_datetime64tz_dtype(out["timestamp"]):
        out["timestamp"] = out["timestamp"].dt.tz_convert(tz)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_localize("UTC").dt.tz_convert(tz)

    out = out.set_index("timestamp").sort_index()
    out = out[out.index.dayofweek < 5]
    out = out.between_time(start_time=start, end_time=end)

    return out.reset_index()


def resample_ohlc(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """
    Resample on the time index with OHLCV aggregation:
      open: first, high: max, low: min, close: last, volume: sum (if present)
    """
    need = {"timestamp", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        missing = sorted(need - set(df.columns))
        raise ValueError(f"resample_ohlc: missing required columns: {missing}")

    x = df.set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in x.columns:
        agg["volume"] = "sum"

    out = x.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out.reset_index()


# --- Backward-compat ---------------------------------------------------------

def load_excel_ohlcv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Compatibility: read via load_ohlcv(), then convert to 'tz' if provided.
    """
    df = load_ohlcv(path)
    if tz:
        if not is_datetime64tz_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    return df
