"""Time series data loading and validation."""

from pathlib import Path

import pandas as pd


def load_time_series(
    path: str,
    date_column: str = "date",
    value_column: str = "value",
    freq: str | None = None,
) -> pd.DataFrame:
    """
    Load time series data from CSV.

    Args:
        path: Path to CSV file
        date_column: Name of date column
        value_column: Name of value column
        freq: Optional frequency string (e.g., 'D', 'W', 'M')

    Returns:
        DataFrame with datetime index and value column

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in {path}")
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found in {path}")

    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Set as index
    df = df.set_index(date_column)
    df = df[[value_column]].copy()
    df.columns = ["value"]

    # Sort by date
    df = df.sort_index()

    # Set frequency if provided
    if freq:
        df = df.asfreq(freq)

    return df


def load_full_dataframe(
    path: str,
    datetime_column: str | None = None,
) -> pd.DataFrame:
    """
    Load full DataFrame without filtering columns.

    Useful for dashboard display and multivariate forecasting.

    Args:
        path: Path to CSV file
        datetime_column: Optional datetime column to set as index

    Returns:
        Full DataFrame with all columns
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # Set datetime index if specified
    if datetime_column and datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.set_index(datetime_column)
        df = df.sort_index()

    return df


def validate_time_series(df: pd.DataFrame, min_points: int = 10) -> dict:
    """
    Validate time series data quality.

    Args:
        df: DataFrame with datetime index and 'value' column
        min_points: Minimum number of data points required

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "n_points": int,
            "has_missing": bool,
            "n_missing": int,
            "has_duplicates": bool,
            "errors": list[str]
        }
    """
    errors = []

    # Check index type
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be DatetimeIndex")

    # Check required column
    if "value" not in df.columns:
        errors.append("DataFrame must have 'value' column")
        return {
            "valid": bool(len(errors) == 0),
            "n_points": 0,
            "has_missing": False,
            "n_missing": 0,
            "has_duplicates": False,
            "errors": errors,
        }

    n_points = len(df)
    n_missing = df["value"].isna().sum()
    has_duplicates = df.index.duplicated().any()

    if n_points < min_points:
        errors.append(f"Too few data points: {n_points} < {min_points}")

    if has_duplicates:
        errors.append("Duplicate timestamps found")

    return {
        "valid": bool(len(errors) == 0),
        "n_points": int(n_points),
        "has_missing": bool(n_missing > 0),
        "n_missing": int(n_missing),
        "has_duplicates": bool(has_duplicates),
        "errors": errors,
    }
