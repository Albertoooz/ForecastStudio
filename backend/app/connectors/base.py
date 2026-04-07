"""Abstract base for data connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class BaseConnector(ABC):
    """Load tabular data from an external source into a Polars DataFrame."""

    source_type: str = ""

    @abstractmethod
    def test_connection(self, config: dict[str, Any]) -> bool:
        """Return True if credentials / network allow a connection."""

    @abstractmethod
    def fetch_data(self, config: dict[str, Any]) -> pl.DataFrame:
        """Load full snapshot (table or custom SQL)."""

    def list_tables(self, config: dict[str, Any]) -> list[str]:
        """Optional: list relation names (default: not supported)."""
        raise NotImplementedError

    def preview_table(self, config: dict[str, Any], table: str, limit: int = 50) -> pl.DataFrame:
        """Optional: first N rows of a table."""
        raise NotImplementedError
