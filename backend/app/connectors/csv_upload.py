"""Helpers for file-based datasets (CSV / Excel / Parquet) — used by upload endpoint."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from app.connectors.base import BaseConnector


def read_upload_bytes(filename: str, content: bytes) -> pd.DataFrame:
    """Parse uploaded file bytes into a DataFrame (same rules as data.py)."""
    name = filename or ""
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


class CsvUploadConnector(BaseConnector):
    """Placeholder for registry; actual upload uses read_upload_bytes + blob in API."""

    source_type = "file"

    def test_connection(self, config: dict[str, Any]) -> bool:
        return True

    def fetch_data(self, config: dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError("Use POST /data/upload for file imports")
