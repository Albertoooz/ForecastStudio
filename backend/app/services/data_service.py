"""
Data Service — business logic for dataset operations.

Wraps upload, validation, schema detection, transformation.
Re-uses forecaster.data.loader and forecaster.data.analyzer.
"""

import io
from typing import cast
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Dataset


class DataService:
    """Stateless service — each method receives a DB session."""

    # ── Upload / Load ────────────────────────────────────────────────────

    @staticmethod
    def read_file(content: bytes, filename: str) -> pd.DataFrame:
        """Parse raw bytes → DataFrame."""
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
        elif filename.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(content))
        else:
            return pd.read_csv(io.BytesIO(content))

    @staticmethod
    def detect_schema(df: pd.DataFrame) -> dict:
        """Auto-detect column types and time-series structure."""
        schema: dict = {
            "columns": {},
            "datetime_candidates": [],
            "numeric_columns": [],
            "categorical_columns": [],
        }
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema["columns"][col] = dtype
            if "datetime" in dtype or "date" in col.lower() or "time" in col.lower():
                schema["datetime_candidates"].append(col)
            elif df[col].dtype.kind in ("i", "f"):
                schema["numeric_columns"].append(col)
            else:
                schema["categorical_columns"].append(col)

        # Try to parse potential datetime columns
        for col in list(schema["categorical_columns"]):
            try:
                pd.to_datetime(df[col].head(5))
                schema["datetime_candidates"].append(col)
            except Exception:
                pass

        return schema

    @staticmethod
    def detect_frequency(df: pd.DataFrame, datetime_col: str) -> str | None:
        """Infer time series frequency."""
        try:
            ts = pd.to_datetime(df[datetime_col]).sort_values()
            freq = pd.infer_freq(ts)
            return cast(str | None, freq)
        except Exception:
            return None

    @staticmethod
    def compute_summary(df: pd.DataFrame) -> dict:
        """Compute dataset summary statistics."""
        return {
            "shape": list(df.shape),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "describe": df.describe(include="all").to_dict(),
        }

    # ── Transformations ──────────────────────────────────────────────────

    @staticmethod
    def apply_transform(df: pd.DataFrame, operation: str, params: dict) -> pd.DataFrame:
        """
        Apply a data transformation.
        Operations: fill_missing, drop_missing, filter_date_range,
                    combine_datetime, resample, rename_columns, etc.
        """
        if operation == "fill_missing":
            method = params.get("method", "ffill")
            columns = params.get("columns")
            if columns:
                df[columns] = df[columns].fillna(method=method)
            else:
                df = df.fillna(method=method)

        elif operation == "drop_missing":
            columns = params.get("columns")
            threshold = params.get("threshold")
            if threshold:
                df = df.dropna(thresh=int(len(df) * threshold))
            elif columns:
                df = df.dropna(subset=columns)
            else:
                df = df.dropna()

        elif operation == "filter_date_range":
            col = params["column"]
            df[col] = pd.to_datetime(df[col])
            if "start" in params:
                df = df[df[col] >= params["start"]]
            if "end" in params:
                df = df[df[col] <= params["end"]]

        elif operation == "combine_datetime":
            date_col = params["date_column"]
            time_col = params["time_column"]
            target = params.get("target_column", "datetime")
            df[target] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))
            if params.get("drop_original", True):
                df = df.drop(columns=[date_col, time_col])

        elif operation == "resample":
            datetime_col = params["datetime_column"]
            freq = params["frequency"]
            agg = params.get("aggregation", "mean")
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col).resample(freq).agg(agg).reset_index()

        elif operation == "rename_columns":
            mapping = params["mapping"]  # {"old_name": "new_name", ...}
            df = df.rename(columns=mapping)

        elif operation == "drop_columns":
            cols = params["columns"]
            df = df.drop(columns=cols, errors="ignore")

        elif operation == "cast_types":
            for col, dtype in params["types"].items():
                if dtype == "datetime":
                    df[col] = pd.to_datetime(df[col])
                elif dtype == "numeric":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dtype)

        else:
            raise ValueError(f"Unknown operation: {operation}")

        return df

    # ── DB helpers ───────────────────────────────────────────────────────

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
