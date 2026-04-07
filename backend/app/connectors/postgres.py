"""PostgreSQL connector — SQLAlchemy metadata + Polars for row loads."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.connectors.base import BaseConnector

_SAFE_PART = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_FREQ_TO_DATE_TRUNC: dict[str, str] = {
    "H": "hour",
    "1H": "hour",
    "D": "day",
    "1D": "day",
    "W": "week",
    "1W": "week",
    "M": "month",
    "1M": "month",
    "Q": "quarter",
    "Y": "year",
}

_AGG_TO_SQL: dict[str, str] = {
    "sum": "SUM",
    "mean": "AVG",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}

# Default safety cap — never load more rows than this unless explicitly overridden.
_DEFAULT_ROW_LIMIT = 1_000_000


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

    def preview_table(self, config: dict[str, Any], table: str, limit: int = 50) -> pl.DataFrame:
        lim = max(1, min(int(limit), 500))
        eng = _build_engine(config)
        try:
            qtext = config.get("query")
            if qtext and str(qtext).strip():
                inner = str(qtext).strip()
                sql = f"SELECT * FROM ({inner}) AS _probe LIMIT {lim}"
            else:
                quoted = _quote_table(table)
                sql = f"SELECT * FROM {quoted} LIMIT {lim}"
            return pl.read_database(sql, eng)
        finally:
            eng.dispose()

    def fetch_data(self, config: dict[str, Any]) -> pl.DataFrame:
        eng = _build_engine(config)
        try:
            qtext = config.get("query")
            pre_agg: dict | None = config.get("pre_aggregate")
            row_limit: int | None = config.get("row_limit", _DEFAULT_ROW_LIMIT)

            if qtext and str(qtext).strip():
                base_sql = str(qtext).strip()
            else:
                table = config.get("table")
                if not table:
                    raise ValueError("postgres config requires 'query' or 'table'")
                quoted = _quote_table(str(table))
                base_sql = f"SELECT * FROM {quoted}"

            if pre_agg:
                sql = _build_pre_agg_sql(base_sql, pre_agg)
            else:
                sql = base_sql

            if row_limit and row_limit > 0:
                # Wrap in a subquery so LIMIT works on arbitrary base queries too.
                sql = f"SELECT * FROM ({sql}) AS _lim LIMIT {int(row_limit)}"

            return pl.read_database(sql, eng)
        finally:
            eng.dispose()


def _build_pre_agg_sql(base_sql: str, pre_agg: dict) -> str:
    """
    Wrap *base_sql* in a server-side GROUP BY so Postgres returns aggregated
    rows instead of raw transactional data.

    pre_agg keys:
      datetime_column  str        – timestamp/date column to truncate
      frequency        str        – "D" | "W" | "M" | "H" | "Q" | "Y"  (default "D")
      target_columns   list[str]  – columns to aggregate
      group_columns    list[str]  – extra GROUP BY columns (e.g. store_id)
      aggregation      str        – "sum" | "mean" | "count" | "min" | "max" (default "sum")
    """
    dt_col = pre_agg.get("datetime_column")
    if not dt_col or not _SAFE_PART.match(str(dt_col)):
        raise ValueError(
            f"pre_aggregate.datetime_column is required and must be a plain identifier, got: {dt_col!r}"
        )

    freq = (pre_agg.get("frequency") or "D").upper()
    trunc_unit = _FREQ_TO_DATE_TRUNC.get(freq)
    if not trunc_unit:
        raise ValueError(
            f"pre_aggregate.frequency {freq!r} not supported. "
            f"Use one of: {', '.join(_FREQ_TO_DATE_TRUNC)}"
        )

    agg_fn_key = (pre_agg.get("aggregation") or "sum").lower()
    agg_fn = _AGG_TO_SQL.get(agg_fn_key)
    if not agg_fn:
        raise ValueError(f"pre_aggregate.aggregation {agg_fn_key!r} not supported")

    target_cols: list[str] = pre_agg.get("target_columns") or []
    group_cols: list[str] = pre_agg.get("group_columns") or []

    for c in [dt_col, *target_cols, *group_cols]:
        if not _SAFE_PART.match(str(c)):
            raise ValueError(f"pre_aggregate column name must be a plain identifier, got: {c!r}")

    if not target_cols:
        raise ValueError("pre_aggregate.target_columns must list at least one column to aggregate")

    dt_q = f'"{dt_col}"'
    trunc_expr = f"date_trunc('{trunc_unit}', {dt_q})"

    select_parts: list[str] = [f"{trunc_expr} AS {dt_q}"]
    group_by_parts: list[str] = [trunc_expr]

    for gc in group_cols:
        gc_q = f'"{gc}"'
        select_parts.append(gc_q)
        group_by_parts.append(gc_q)

    for tc in target_cols:
        tc_q = f'"{tc}"'
        select_parts.append(f"{agg_fn}({tc_q}) AS {tc_q}")

    select_clause = ", ".join(select_parts)
    group_by_clause = ", ".join(group_by_parts)
    order_by_cols = [trunc_expr] + [f'"{gc}"' for gc in group_cols]
    order_by_clause = ", ".join(order_by_cols)

    return (
        f"SELECT {select_clause} "
        f"FROM ({base_sql}) AS _raw "
        f"GROUP BY {group_by_clause} "
        f"ORDER BY {order_by_clause}"
    )
