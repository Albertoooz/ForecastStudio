"""Data source connectors — Postgres, file, etc."""

from app.connectors.registry import get_connector

__all__ = ["get_connector"]
