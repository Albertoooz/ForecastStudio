"""Polars-first helpers; pandas only for third-party model stacks (lazy import)."""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd


def read_df_from_bytes(data: bytes, *, path_hint: str = "") -> pl.DataFrame:
    """Load Parquet or CSV from raw bytes."""
    buf = io.BytesIO(data)
    if path_hint.endswith(".parquet"):
        return pl.read_parquet(buf)
    if path_hint.endswith(".csv"):
        return pl.read_csv(buf)
    try:
        buf.seek(0)
        return pl.read_parquet(buf)
    except Exception:
        buf.seek(0)
        return pl.read_csv(buf)


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert for Prophet / legacy workflow code (requires pandas installed)."""
    import pandas as pd  # noqa: F401

    return df.to_pandas()


def schema_dtype_map(df: pl.DataFrame) -> dict[str, str]:
    """Column name -> dtype string for JSON schema storage."""
    return {c: str(df[c].dtype) for c in df.columns}


def infer_frequency(series: pl.Series) -> str | None:
    """
    Infer time series frequency from a sorted datetime/date series.

    Returns a pandas-compatible frequency string (e.g. 'D', 'H', 'W', 'M')
    or None if it cannot be determined.
    """
    if series.len() < 3:
        return None
    try:
        if series.dtype == pl.Date:
            deltas = series.cast(pl.Datetime).diff().dt.total_seconds().drop_nulls()
        elif series.dtype.is_temporal():
            deltas = series.cast(pl.Datetime).diff().dt.total_seconds().drop_nulls()
        else:
            deltas = series.cast(pl.Datetime, strict=False).diff().dt.total_seconds().drop_nulls()
    except Exception:
        return None

    if deltas.len() == 0:
        return None

    # Use median to be robust against outliers
    med = float(deltas.median() or 0)
    if med <= 0:
        return None

    day = 86400.0
    hour = 3600.0
    minute = 60.0

    if abs(med - minute) < 5:
        return "T"
    if abs(med - hour) < 120:
        return "H"
    if abs(med - day) < 1800:
        return "D"
    if abs(med - 7 * day) < 12 * hour:
        return "W"
    if day * 25 <= med <= day * 35:
        return "M"
    # Return generic seconds-based string
    return f"{int(med)}s"


def polars_date_range(
    start: datetime,
    *,
    periods: int,
    freq: str,
) -> pl.Series:
    """
    Generate a date range similar to pd.date_range(start, periods=periods, freq=freq).

    Returns a Polars Series of Datetime dtype.
    Supports pandas-style freq strings: D, H, T/min, W, M/MS, s.
    """
    freq = (freq or "D").strip()
    u = freq.upper()

    _freq_map: dict[str, timedelta] = {
        "D": timedelta(days=1),
        "1D": timedelta(days=1),
        "H": timedelta(hours=1),
        "1H": timedelta(hours=1),
        "T": timedelta(minutes=1),
        "MIN": timedelta(minutes=1),
        "W": timedelta(weeks=1),
        "1W": timedelta(weeks=1),
    }

    if u in _freq_map:
        delta = _freq_map[u]
    elif u in ("M", "MS"):
        # Monthly: approximate by generating start + i months
        dates = []
        dt = start
        for _ in range(periods):
            dates.append(dt)
            month = dt.month + 1
            year = dt.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            dt = dt.replace(year=year, month=month, day=1)
        return pl.Series(dates, dtype=pl.Datetime)
    else:
        # Try to parse e.g. "30s", "15T", "2H"
        try:
            n = int("".join(c for c in freq if c.isdigit()) or "1")
            unit = "".join(c for c in freq if c.isalpha()).upper()
            unit_map = {
                "S": timedelta(seconds=1),
                "T": timedelta(minutes=1),
                "MIN": timedelta(minutes=1),
                "H": timedelta(hours=1),
                "D": timedelta(days=1),
                "W": timedelta(weeks=1),
            }
            delta = unit_map.get(unit, timedelta(days=1)) * n
        except Exception:
            delta = timedelta(days=1)

    dates = [start + i * delta for i in range(periods)]
    return pl.Series(dates, dtype=pl.Datetime)
