"""Connector registry."""

from __future__ import annotations

from app.connectors.base import BaseConnector
from app.connectors.csv_upload import CsvUploadConnector
from app.connectors.postgres import PostgresConnector

_REGISTRY: dict[str, BaseConnector] = {
    "postgres": PostgresConnector(),
    "sql": PostgresConnector(),  # alias: stored as sql in legacy model
    "file": CsvUploadConnector(),
}


def get_connector(source_type: str) -> BaseConnector:
    key = (source_type or "").lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown data source type: {source_type}")
    return _REGISTRY[key]
