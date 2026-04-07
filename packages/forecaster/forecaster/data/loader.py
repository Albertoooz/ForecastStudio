"""Time series data loading and validation (Polars)."""

from pathlib import Path

import polars as pl


def load_time_series(
    path: str,
    date_column: str = "date",
    value_column: str = "value",
    freq: str | None = None,
) -> pl.DataFrame:
    """
    Load time series data from CSV.

    Returns a Polars DataFrame with a datetime column and a 'value' column,
    sorted by date.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pl.read_csv(path)

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in {path}")
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found in {path}")

    df = df.with_columns(pl.col(date_column).cast(pl.Datetime, strict=False).alias(date_column))
    df = df.select([date_column, pl.col(value_column).alias("value")])
    df = df.sort(date_column)

    return df


def load_full_dataframe(
    path: str,
    datetime_column: str | None = None,
) -> pl.DataFrame:
    """
    Load full DataFrame without filtering columns.

    Useful for dashboard display and multivariate forecasting.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pl.read_csv(path)

    if datetime_column and datetime_column in df.columns:
        df = df.with_columns(
            pl.col(datetime_column).cast(pl.Datetime, strict=False).alias(datetime_column)
        ).sort(datetime_column)

    return df


def validate_time_series(df: pl.DataFrame, min_points: int = 10) -> dict:
    """
    Validate time series data quality.

    Expects a Polars DataFrame with a datetime-typed column (index-equivalent)
    and a 'value' column.
    """
    errors: list[str] = []

    # Find datetime column
    dt_col = next((c for c in df.columns if df[c].dtype.is_temporal()), None)
    if dt_col is None:
        errors.append("No temporal (date/datetime) column found — index must be datetime")

    if "value" not in df.columns:
        errors.append("DataFrame must have 'value' column")
        return {
            "valid": False,
            "n_points": 0,
            "has_missing": False,
            "n_missing": 0,
            "has_duplicates": False,
            "errors": errors,
        }

    n_points = df.height
    n_missing = int(df["value"].null_count())
    has_duplicates = False
    if dt_col:
        has_duplicates = df.height > df.select(pl.col(dt_col)).n_unique()
        if has_duplicates:
            errors.append("Duplicate timestamps found")

    if n_points < min_points:
        errors.append(f"Too few data points: {n_points} < {min_points}")

    return {
        "valid": len(errors) == 0,
        "n_points": n_points,
        "has_missing": n_missing > 0,
        "n_missing": n_missing,
        "has_duplicates": has_duplicates,
        "errors": errors,
    }
