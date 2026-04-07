"""PostgreSQL connector — sync SQLAlchemy + pandas."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.connectors.base import BaseConnector

_SAFE_PART = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _build_engine(config: dict[str, Any]) -> Engine:
    host = config.get("host") or "localhost"
    port = int(config.get("port") or 5432)
    database = config.get("database") or "postgres"
    user = config.get("username") or config.get("user") or "postgres"
    password = quote_plus(str(config.get("password") or ""))
    user_q = quote_plus(str(user))
    url = f"postgresql+psycopg2://{user_q}:{password}@{host}:{port}/{database}"
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 15})


def _quote_table(table: str) -> str:
    """Quote schema.table or single identifier for use in SQL."""
    raw = table.strip()
    if "." in raw:
        parts = raw.split(".", 1)
        if len(parts) != 2 or not _SAFE_PART.match(parts[0]) or not _SAFE_PART.match(parts[1]):
            raise ValueError("Invalid table name")
        return f'"{parts[0]}"."{parts[1]}"'
    if not _SAFE_PART.match(raw):
        raise ValueError("Invalid table name")
    return f'"{raw}"'


class PostgresConnector(BaseConnector):
    source_type = "postgres"

    def test_connection(self, config: dict[str, Any]) -> bool:
        eng = _build_engine(config)
        try:
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
        finally:
            eng.dispose()

    def list_tables(self, config: dict[str, Any]) -> list[str]:
        eng = _build_engine(config)
        try:
            q = """
            SELECT table_schema || '.' || table_name AS fq
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
            """
            with eng.connect() as conn:
                rows = conn.execute(text(q)).fetchall()
            return [r[0] for r in rows]
        finally:
            eng.dispose()

    def preview_table(self, config: dict[str, Any], table: str, limit: int = 50) -> pd.DataFrame:
        lim = max(1, min(int(limit), 500))
        eng = _build_engine(config)
        try:
            qtext = config.get("query")
            if qtext and str(qtext).strip():
                inner = str(qtext).strip()
                sql = f"SELECT * FROM ({inner}) AS _probe LIMIT {lim}"
                return pd.read_sql(text(sql), eng)
            quoted = _quote_table(table)
            sql = f"SELECT * FROM {quoted} LIMIT {lim}"
            return pd.read_sql(text(sql), eng)
        finally:
            eng.dispose()

    def fetch_data(self, config: dict[str, Any]) -> pd.DataFrame:
        eng = _build_engine(config)
        try:
            qtext = config.get("query")
            if qtext and str(qtext).strip():
                return pd.read_sql(text(str(qtext).strip()), eng)
            table = config.get("table")
            if not table:
                raise ValueError("postgres config requires 'query' or 'table'")
            quoted = _quote_table(str(table))
            sql = f"SELECT * FROM {quoted}"
            return pd.read_sql(text(sql), eng)
        finally:
            eng.dispose()
