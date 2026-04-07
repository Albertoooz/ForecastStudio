"""
Data Service — business logic for dataset operations.

Upload, validation, schema detection, transformation (Polars).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast
from uuid import UUID

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.csv_upload import read_upload_bytes
from app.db.models import Dataset


def _infer_frequency_from_sorted_datetimes(series: pl.Series) -> str | None:
    """Best-effort frequency label from median delta (Polars-only)."""
    if series.len() < 3:
        return None
    deltas = series.diff().dt.total_seconds().drop_nulls()
    if deltas.len() == 0:
        return None
    med_raw = deltas.median()
    med = float(cast(Any, med_raw)) if med_raw is not None else 0.0
    if med <= 0:
        return None
    day = 86400.0
    hour = 3600.0
    if abs(med - day) < 1800:
        return "D"
    if abs(med - hour) < 120:
        return "H"
    if abs(med - 7 * day) < 12 * hour:
        return "W"
    if day * 25 <= med <= day * 35:
        return "M"
    return None


class DataService:
    """Stateless service — each method receives a DB session."""

    @staticmethod
    def read_file(content: bytes, filename: str) -> pl.DataFrame:
        """Parse raw bytes → DataFrame."""
        return read_upload_bytes(filename, content)

    @staticmethod
    def detect_schema(df: pl.DataFrame) -> dict:
        """Auto-detect column types and time-series structure."""
        schema: dict = {
            "columns": {},
            "datetime_candidates": [],
            "numeric_columns": [],
            "categorical_columns": [],
        }
        for col in df.columns:
            dtype = df[col].dtype
            schema["columns"][col] = str(dtype)
            if dtype.is_temporal():
                schema["datetime_candidates"].append(col)
            elif dtype.is_numeric():
                schema["numeric_columns"].append(col)
            else:
                cl = col.lower()
                if "date" in cl or "time" in cl or cl == "ds":
                    schema["datetime_candidates"].append(col)
                else:
                    schema["categorical_columns"].append(col)

        return schema

    @staticmethod
    def detect_frequency(df: pl.DataFrame, datetime_col: str) -> str | None:
        """Infer time series frequency from sorted timestamp deltas."""
        try:
            s = df.sort(datetime_col)[datetime_col]
            return _infer_frequency_from_sorted_datetimes(s)
        except Exception:
            return None

    @staticmethod
    def compute_summary(df: pl.DataFrame) -> dict:
        """Compute dataset summary statistics."""
        n = df.height or 1
        missing = {c: int(df[c].null_count()) for c in df.columns}
        missing_pct = {c: round(100.0 * missing[c] / n, 2) for c in df.columns}
        describe_rows: list[dict] = []
        try:
            describe_rows = df.describe().to_dicts()
        except Exception:
            pass
        return {
            "shape": [df.height, df.width],
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing": missing,
            "missing_pct": missing_pct,
            "describe": describe_rows,
        }

    @staticmethod
    def apply_transform(df: pl.DataFrame, operation: str, params: dict) -> pl.DataFrame:
        """Apply a data transformation (Polars)."""
        if operation == "fill_missing":
            method = params.get("method", "ffill")
            columns = params.get("columns")
            if columns:
                for col in columns:
                    if col in df.columns:
                        if method == "ffill":
                            df = df.with_columns(pl.col(col).forward_fill())
                        elif method == "bfill":
                            df = df.with_columns(pl.col(col).backward_fill())
            else:
                if method == "ffill":
                    df = df.fill_null(strategy="forward")
                elif method == "bfill":
                    df = df.fill_null(strategy="backward")

        elif operation == "drop_missing":
            columns = params.get("columns")
            threshold = params.get("threshold")
            if threshold:
                thresh = int(len(df.columns) * float(threshold))
                nn = pl.sum_horizontal([pl.col(c).is_not_null() for c in df.columns])
                df = df.filter(nn >= thresh)
            elif columns:
                df = df.drop_nulls(subset=columns)
            else:
                df = df.drop_nulls()

        elif operation == "filter_date_range":
            col = params["column"]
            tsc = pl.col(col).cast(pl.Datetime, strict=False)
            cond = pl.lit(True)
            if "start" in params:
                s = params["start"]
                if isinstance(s, datetime):
                    cond = cond & (tsc >= pl.lit(s))
                else:
                    cond = cond & (tsc >= pl.lit(str(s)).str.to_datetime(strict=False))
            if "end" in params:
                e = params["end"]
                if isinstance(e, datetime):
                    cond = cond & (tsc <= pl.lit(e))
                else:
                    cond = cond & (tsc <= pl.lit(str(e)).str.to_datetime(strict=False))
            df = df.filter(cond)

        elif operation == "combine_datetime":
            date_col = params["date_column"]
            time_col = params["time_column"]
            target = params.get("target_column", "datetime")
            combined = (
                pl.col(date_col).cast(pl.Utf8) + pl.lit(" ") + pl.col(time_col).cast(pl.Utf8)
            ).str.to_datetime(strict=False)
            df = df.with_columns(combined.alias(target))
            if params.get("drop_original", True):
                df = df.drop(date_col, time_col)

        elif operation == "resample":
            datetime_col = params["datetime_column"]
            freq = params["frequency"]
            agg = params.get("aggregation", "mean")
            vc = params.get("value_column") or params.get("agg_column")
            every = _pd_freq_to_polars_service(freq)
            sorted_df = df.sort(datetime_col)
            if vc:
                sorted_df = sorted_df.group_by_dynamic(datetime_col, every=every).agg(
                    _agg_expr_service(vc, agg)
                )
            else:
                num_cols = [c for c in df.columns if c != datetime_col and df[c].dtype.is_numeric()]
                if not num_cols:
                    raise ValueError(
                        "resample requires value_column/agg_column or numeric columns besides datetime"
                    )
                exprs = [_agg_expr_service(c, agg) for c in num_cols]
                sorted_df = sorted_df.group_by_dynamic(datetime_col, every=every).agg(*exprs)
            df = sorted_df

        elif operation == "rename_columns":
            mapping = params["mapping"]
            df = df.rename(mapping)

        elif operation == "drop_columns":
            cols = params["columns"]
            df = df.drop(*[c for c in cols if c in df.columns])

        elif operation == "cast_types":
            for col, dtype in params["types"].items():
                if dtype == "datetime":
                    df = df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))
                elif dtype == "numeric":
                    df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                else:
                    df = df.with_columns(pl.col(col).cast(dtype, strict=False))

        else:
            raise ValueError(f"Unknown operation: {operation}")

        return df

    @staticmethod
    async def get_dataset(dataset_id: UUID, tenant_id: UUID, db: AsyncSession) -> Dataset:
        result = await db.execute(
            select(Dataset).where(
                Dataset.id == dataset_id,
                Dataset.tenant_id == tenant_id,
            )
        )
        dataset = result.scalar_one_or_none()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        return dataset


def _pd_freq_to_polars_service(frequency: str) -> str:
    f = (frequency or "D").strip()
    u = f.upper()
    mapping = {
        "D": "1d",
        "1D": "1d",
        "H": "1h",
        "1H": "1h",
        "W": "1w",
        "M": "1mo",
        "T": "1m",
        "MIN": "1m",
    }
    return mapping.get(u, f if any(c.isdigit() for c in f) else "1d")


def _agg_expr_service(value_column: str, agg: str) -> pl.Expr:
    if not value_column:
        raise ValueError("resample requires value_column or agg_column in params")
    c = pl.col(value_column)
    a = (agg or "mean").lower()
    if a == "mean":
        return c.mean()
    if a == "sum":
        return c.sum()
    if a == "count":
        return c.count()
    if a == "min":
        return c.min()
    if a == "max":
        return c.max()
    raise ValueError(f"Unknown aggregation: {agg}")
