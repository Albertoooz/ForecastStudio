"""Data file analyzer — detects columns, types, and issues (Polars)."""

from pathlib import Path

import polars as pl

from forecaster.core.session import ColumnInfo, DataInfo
from forecaster.utils.tabular import infer_frequency


def analyze_file(filepath: Path, filename: str) -> DataInfo:
    """Analyze uploaded file and detect structure."""
    df = _load_file(filepath)
    return analyze_dataframe(df, filename, filepath)


def analyze_dataframe(df: pl.DataFrame, filename: str, filepath: Path | None = None) -> DataInfo:
    """Analyze Polars DataFrame and detect structure."""
    if filepath is None:
        import tempfile

        filepath = Path(tempfile.gettempdir()) / filename

    columns = []
    datetime_candidates: list[str] = []
    numeric_candidates: list[str] = []

    for col in df.columns:
        col_info = _analyze_column(df, col)
        columns.append(col_info)
        if col_info.is_datetime:
            datetime_candidates.append(col)
        if col_info.is_numeric:
            numeric_candidates.append(col)

    datetime_column = _detect_datetime_column(datetime_candidates, df)
    target_column = _detect_target_column(numeric_candidates, datetime_column)

    frequency = None
    date_range = None
    if datetime_column:
        frequency = _detect_frequency(df, datetime_column)
        date_range = _get_date_range(df, datetime_column)

    issues = _identify_issues(df, columns, datetime_column)
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
        n_rows=df.height,
        date_range=date_range,
        issues=issues,
        questions=questions,
    )


def _load_file(filepath: Path) -> pl.DataFrame:
    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(filepath)
    if suffix in (".xlsx", ".xls"):
        return pl.read_excel(filepath)
    if suffix == ".parquet":
        return pl.read_parquet(filepath)
    return pl.read_csv(filepath)


def _analyze_column(df: pl.DataFrame, col: str) -> ColumnInfo:
    series = df[col]
    dtype = str(series.dtype)
    n_missing = int(series.null_count())
    n_unique = int(series.n_unique())

    non_null = series.drop_nulls()
    sample_values = [str(v) for v in non_null.head(3).to_list()]

    is_numeric = series.dtype.is_numeric()
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


def _is_datetime_column(series: pl.Series, col_name: str) -> bool:
    if series.dtype.is_temporal():
        return True

    datetime_keywords = ["date", "time", "timestamp", "dt", "datetime", "data", "dzien", "miesiac"]
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in datetime_keywords):
        try:
            series.drop_nulls().head(10).cast(pl.Datetime, strict=False)
            return True
        except Exception:
            pass

    try:
        non_null = series.drop_nulls().head(10)
        if non_null.len() > 0:
            non_null.cast(pl.Utf8).str.to_datetime(strict=False)
            return True
    except Exception:
        pass

    return False


def _detect_datetime_column(candidates: list[str], df: pl.DataFrame) -> str | None:
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for col in candidates:
            if "date" in col.lower() or "time" in col.lower():
                return col
        return candidates[0]
    return None


def _detect_target_column(numeric_candidates: list[str], datetime_column: str | None) -> str | None:
    candidates = [c for c in numeric_candidates if c != datetime_column]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        target_keywords = ["value", "sales", "revenue", "price", "amount", "count", "target", "y"]
        for col in candidates:
            if any(kw in col.lower() for kw in target_keywords):
                return col
    return None


def _detect_frequency(df: pl.DataFrame, datetime_column: str) -> str | None:
    try:
        series = df.sort(datetime_column)[datetime_column]
        return infer_frequency(series)
    except Exception:
        return None


def _get_date_range(df: pl.DataFrame, datetime_column: str) -> tuple[str, str] | None:
    try:
        s = df[datetime_column].cast(pl.Datetime, strict=False)
        mn = s.min()
        mx = s.max()
        if mn is None or mx is None:
            return None
        return (str(mn)[:10], str(mx)[:10])
    except Exception:
        return None


def _identify_issues(
    df: pl.DataFrame, columns: list[ColumnInfo], datetime_column: str | None
) -> list[str]:
    issues = []
    for col in columns:
        if col.n_missing > 0:
            pct = (col.n_missing / df.height) * 100
            if pct > 5:
                issues.append(
                    f"Column '{col.name}' has {col.n_missing} missing values ({pct:.1f}%)"
                )
    if df.height < 10:
        issues.append(f"Very little data ({df.height} rows) - forecast may be inaccurate")
    if datetime_column and datetime_column in df.columns:
        n_dup = df.height - df.select(pl.col(datetime_column)).n_unique()
        if n_dup > 0:
            issues.append(f"Found {n_dup} duplicate dates")
    return issues


def _generate_questions(
    columns: list[ColumnInfo],
    datetime_column: str | None,
    target_column: str | None,
    datetime_candidates: list[str],
    numeric_candidates: list[str],
) -> list[str]:
    questions = []
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
    if target_column is None and len(numeric_candidates) == 0:
        questions.append("No numeric column detected for forecasting. Which column is the target?")
    elif target_column is None and len(numeric_candidates) > 1:
        candidates = [c for c in numeric_candidates if c != datetime_column]
        if len(candidates) > 1:
            questions.append(
                f"Detected several numeric columns: {', '.join(candidates)}. Which one to forecast?"
            )
    return questions
