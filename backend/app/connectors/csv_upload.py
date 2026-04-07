"""Helpers for file-based datasets (CSV / Excel / Parquet) — used by upload endpoint."""

from __future__ import annotations

import io
from typing import Any

import polars as pl

from app.connectors.base import BaseConnector


def read_upload_bytes(filename: str, content: bytes) -> pl.DataFrame:
    """Parse uploaded file bytes into a Polars DataFrame (same rules as data.py)."""
    name = filename or ""
    buf = io.BytesIO(content)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pl.read_excel(buf)
    if name.endswith(".parquet"):
        return pl.read_parquet(buf)
    return pl.read_csv(buf)


class CsvUploadConnector(BaseConnector):
    """Placeholder for registry; actual upload uses read_upload_bytes + blob in API."""

    source_type = "file"

    def test_connection(self, config: dict[str, Any]) -> bool:
        return True

    def fetch_data(self, config: dict[str, Any]) -> pl.DataFrame:
        raise NotImplementedError("Use POST /data/upload for file imports")
