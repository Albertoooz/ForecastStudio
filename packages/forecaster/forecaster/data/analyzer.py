"""Data file analyzer - detects columns, types, and issues."""

from pathlib import Path

import pandas as pd

from forecaster.core.session import ColumnInfo, DataInfo


def analyze_file(filepath: Path, filename: str) -> DataInfo:
    """
    Analyze uploaded file and detect structure.

    Args:
        filepath: Path to the file
        filename: Original filename

    Returns:
        DataInfo with detected columns, types, and potential issues
    """
    df = _load_file(filepath)
    return analyze_dataframe(df, filename, filepath)


def analyze_dataframe(df: pd.DataFrame, filename: str, filepath: Path | None = None) -> DataInfo:
    """
    Analyze DataFrame and detect structure.

    Args:
        df: DataFrame to analyze
        filename: Original filename for reference
        filepath: Optional path to file (for compatibility)

    Returns:
        DataInfo with detected columns, types, and potential issues
    """
    # Use temp path if not provided
    if filepath is None:
        import tempfile
        from pathlib import Path

        filepath = Path(tempfile.gettempdir()) / filename

    # Analyze columns
    columns = []
    datetime_candidates = []
    numeric_candidates = []

    for col in df.columns:
        col_info = _analyze_column(df, col)
        columns.append(col_info)

        if col_info.is_datetime:
            datetime_candidates.append(col)
        if col_info.is_numeric:
            numeric_candidates.append(col)

    # Auto-detect datetime and target columns
    datetime_column = _detect_datetime_column(datetime_candidates, df)
    target_column = _detect_target_column(numeric_candidates, datetime_column)

    # Detect frequency
    frequency = None
    date_range = None
    if datetime_column:
        frequency = _detect_frequency(df, datetime_column)
        date_range = _get_date_range(df, datetime_column)

    # Identify issues
    issues = _identify_issues(df, columns, datetime_column)

    # Generate questions for user
    questions = _generate_questions(
        columns, datetime_column, target_column, datetime_candidates, numeric_candidates
    )

    return DataInfo(
        filepath=filepath,
        filename=filename,
        columns=columns,
        datetime_column=datetime_column,
        target_column=target_column,
        frequency=frequency,
        n_rows=len(df),
        date_range=date_range,
        issues=issues,
        questions=questions,
    )


def _load_file(filepath: Path) -> pd.DataFrame:
    """Load file into DataFrame."""
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(filepath)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)
    elif suffix == ".parquet":
        return pd.read_parquet(filepath)
    else:
        # Try CSV as default
        return pd.read_csv(filepath)


def _analyze_column(df: pd.DataFrame, col: str) -> ColumnInfo:
    """Analyze a single column."""
    series = df[col]

    # Basic info
    dtype = str(series.dtype)
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique())

    # Sample values (non-null)
    non_null = series.dropna()
    sample_values = [str(v) for v in non_null.head(3).tolist()]

    # Type detection
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = _is_datetime_column(series, col)

    return ColumnInfo(
        name=col,
        dtype=dtype,
        n_missing=n_missing,
        n_unique=n_unique,
        sample_values=sample_values,
        is_datetime=is_datetime,
        is_numeric=is_numeric,
    )


def _is_datetime_column(series: pd.Series, col_name: str) -> bool:
    """Check if column is datetime."""
    # Already datetime type
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Check by column name
    datetime_keywords = ["date", "time", "timestamp", "dt", "datetime", "data", "dzien", "miesiac"]
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in datetime_keywords):
        # Try to parse
        try:
            pd.to_datetime(series.dropna().head(10))
            return True
        except Exception:
            pass

    # Try to parse first few values
    try:
        non_null = series.dropna().head(10)
        if len(non_null) > 0:
            pd.to_datetime(non_null)
            return True
    except Exception:
        pass

    return False


def _detect_datetime_column(candidates: list[str], df: pd.DataFrame) -> str | None:
    """Auto-detect the datetime column."""
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        # Prefer columns with "date" in name
        for col in candidates:
            if "date" in col.lower() or "time" in col.lower():
                return col
        return candidates[0]

    return None


def _detect_target_column(numeric_candidates: list[str], datetime_column: str | None) -> str | None:
    """Auto-detect the target column."""
    # Filter out datetime column
    candidates = [c for c in numeric_candidates if c != datetime_column]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        # Prefer columns with common target names
        target_keywords = ["value", "sales", "revenue", "price", "amount", "count", "target", "y"]
        for col in candidates:
            if any(kw in col.lower() for kw in target_keywords):
                return col

    return None


def _detect_frequency(df: pd.DataFrame, datetime_column: str) -> str | None:
    """Detect time series frequency, including sub-daily (15min, hourly, etc.)."""
    try:
        dates = pd.to_datetime(df[datetime_column])
        dates = dates.sort_values()
        diffs = dates.diff().dropna()

        if len(diffs) == 0:
            return None

        # Get most common difference
        mode_diff = diffs.mode()[0]

        # Total seconds for sub-daily detection
        total_seconds = mode_diff.total_seconds()

        if total_seconds <= 0:
            return None

        # Sub-daily frequencies
        if total_seconds < 60:
            return f"{int(total_seconds)}s"  # seconds
        elif total_seconds < 3600:
            minutes = int(total_seconds // 60)
            return f"{minutes}min"
        elif total_seconds < 86400:
            hours = total_seconds / 3600
            if hours == int(hours):
                return f"{int(hours)}h"
            else:
                # Fractional hours, express in minutes
                return f"{int(total_seconds // 60)}min"

        # Daily and above
        days = mode_diff.days

        if days == 1:
            return "D"
        elif days == 7:
            return "W"
        elif 28 <= days <= 31:
            return "M"
        elif 365 <= days <= 366:
            return "Y"
        else:
            return f"{days}D"

    except Exception:
        return None


def _get_date_range(df: pd.DataFrame, datetime_column: str) -> tuple[str, str] | None:
    """Get date range of the data."""
    try:
        dates = pd.to_datetime(df[datetime_column])
        min_date = dates.min().strftime("%Y-%m-%d")
        max_date = dates.max().strftime("%Y-%m-%d")
        return (min_date, max_date)
    except Exception:
        return None


def _identify_issues(
    df: pd.DataFrame, columns: list[ColumnInfo], datetime_column: str | None
) -> list[str]:
    """Identify potential data issues."""
    issues = []

    # Check for missing values
    for col in columns:
        if col.n_missing > 0:
            pct = (col.n_missing / len(df)) * 100
            if pct > 5:
                issues.append(
                    f"Column '{col.name}' has {col.n_missing} missing values ({pct:.1f}%)"
                )

    # Check for too few rows
    if len(df) < 10:
        issues.append(f"Very little data ({len(df)} rows) - forecast may be inaccurate")

    # Check for duplicates in datetime column
    if datetime_column:
        n_duplicates = df[datetime_column].duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"Found {n_duplicates} duplicate dates")

    return issues


def _generate_questions(
    columns: list[ColumnInfo],
    datetime_column: str | None,
    target_column: str | None,
    datetime_candidates: list[str],
    numeric_candidates: list[str],
) -> list[str]:
    """Generate questions for user clarification."""
    questions = []

    # Question about datetime column
    if datetime_column is None and len(datetime_candidates) == 0:
        col_names = [c.name for c in columns]
        questions.append(
            f"Date column not detected. Which column contains dates? "
            f"Available: {', '.join(col_names)}"
        )
    elif datetime_column is None and len(datetime_candidates) > 1:
        questions.append(
            f"Detected several possible date columns: {', '.join(datetime_candidates)}. "
            f"Which one to choose?"
        )

    # Question about target column
    if target_column is None and len(numeric_candidates) == 0:
        questions.append("No numeric column detected for forecasting. Which column is the target?")
    elif target_column is None and len(numeric_candidates) > 1:
        # Filter out datetime
        candidates = [c for c in numeric_candidates if c != datetime_column]
        if len(candidates) > 1:
            questions.append(
                f"Detected several numeric columns: {', '.join(candidates)}. Which one to forecast?"
            )

    return questions
