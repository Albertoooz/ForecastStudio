"""
Data connections API — PostgreSQL and other connectors; materialize snapshots to blob.
"""

from datetime import datetime
from uuid import UUID, uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.auth import get_current_user
from app.connectors.registry import get_connector
from app.db.models import DataSource, Dataset, User
from app.db.session import get_db
from app.storage.blob import get_blob_storage

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class PostgresConnectionBody(BaseModel):
    host: str
    port: int = 5432
    database: str
    username: str
    password: str
    query: str | None = None
    table: str | None = None


class ConnectionTestRequest(BaseModel):
    source_type: str = Field(..., description="postgres")
    postgres: PostgresConnectionBody


class ProbeTablesRequest(ConnectionTestRequest):
    pass


class ProbePreviewRequest(ConnectionTestRequest):
    table: str = Field("", description="schema.table (ignored when postgres.query is set)")
    rows: int = Field(20, ge=1, le=500)


class ConnectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    source_type: str = "postgres"
    postgres: PostgresConnectionBody
    dataset_type: str | None = "training"


def _pg_to_config(pg: PostgresConnectionBody) -> dict:
    return {
        "host": pg.host,
        "port": pg.port,
        "database": pg.database,
        "username": pg.username,
        "password": pg.password,
        "query": pg.query,
        "table": pg.table,
    }


def _query_or_table_label(pg: PostgresConnectionBody) -> str:
    if pg.query and str(pg.query).strip():
        return (pg.query.strip()[:500] + ("…" if len(pg.query) > 500 else "")) if pg.query else ""
    return pg.table or ""


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/test")
async def test_connection(
    body: ConnectionTestRequest,
    user: User = Depends(get_current_user),
):
    """Verify connector credentials (nothing persisted)."""
    _ = user
    if body.source_type.lower() != "postgres":
        raise HTTPException(400, "Only postgres is supported in this release")
    conn = get_connector("postgres")
    cfg = _pg_to_config(body.postgres)
    if not conn.test_connection(cfg):
        raise HTTPException(400, "Connection failed — check host, port, database, user, password")
    return {"ok": True}


@router.post("/probe/tables")
async def probe_tables(
    body: ProbeTablesRequest,
    user: User = Depends(get_current_user),
):
    _ = user
    if body.source_type.lower() != "postgres":
        raise HTTPException(400, "Only postgres is supported")
    conn = get_connector("postgres")
    cfg = _pg_to_config(body.postgres)
    try:
        tables = conn.list_tables(cfg)
    except Exception as e:
        raise HTTPException(400, f"Could not list tables: {e}") from e
    return {"tables": tables}


@router.post("/probe/preview")
async def probe_preview(
    body: ProbePreviewRequest,
    user: User = Depends(get_current_user),
):
    _ = user
    if body.source_type.lower() != "postgres":
        raise HTTPException(400, "Only postgres is supported")
    conn = get_connector("postgres")
    cfg = _pg_to_config(body.postgres)
    try:
        df = conn.preview_table(cfg, body.table, limit=body.rows)
    except Exception as e:
        raise HTTPException(400, f"Preview failed: {e}") from e
    head = df.where(pd.notnull(df), None).to_dict(orient="records")
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": head,
        "shape": [len(df), len(df.columns)],
    }


@router.post("/", status_code=201)
async def create_connection(
    body: ConnectionCreateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Create a Dataset + DataSource, pull data from Postgres into blob (parquet snapshot).
    """
    if body.source_type.lower() != "postgres":
        raise HTTPException(400, "Only postgres is supported")
    conn = get_connector("postgres")
    cfg = _pg_to_config(body.postgres)
    if not cfg.get("query") and not cfg.get("table"):
        raise HTTPException(400, "Provide postgres.query or postgres.table")
    if not conn.test_connection(cfg):
        raise HTTPException(400, "Connection test failed")

    try:
        df = conn.fetch_data(cfg)
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch data: {e}") from e

    blob = get_blob_storage()
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in body.name)[:80]
    blob_path = f"{user.tenant_id}/conn_{safe_name}_{uuid4().hex[:8]}.parquet"

    blob.upload_df("datasets", blob_path, df, fmt="parquet")

    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    datetime_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or col.lower() == "ds":
            try:
                pd.to_datetime(df[col].head(10))
                datetime_col = col
                break
            except Exception:
                pass

    if body.dataset_type and body.dataset_type == "future_variables":
        schema["__type__"] = "future_variables"
    else:
        schema["__type__"] = "training"

    dataset = Dataset(
        tenant_id=user.tenant_id,
        name=body.name,
        blob_path=blob_path,
        file_type="parquet",
        schema_json=schema,
        row_count=len(df),
        column_count=len(df.columns),
        datetime_column=datetime_col,
    )
    db.add(dataset)
    await db.flush()

    ds = DataSource(
        dataset_id=dataset.id,
        source_type="postgres",
        config_json=cfg,
        query_or_table=_query_or_table_label(body.postgres),
        status="connected",
        last_error=None,
        last_sync_at=datetime.utcnow(),
    )
    db.add(ds)
    await db.commit()
    await db.refresh(dataset)
    await db.refresh(ds)

    from app.api.data import _dataset_meta

    return _dataset_meta(dataset, ds)


async def perform_data_source_sync(
    db: AsyncSession,
    ds_row: DataSource,
) -> Dataset:
    """
    Re-fetch data from connector into blob; mutates ds_row and dataset, flushes.
    Caller commits and refreshes.
    """
    dataset = ds_row.dataset
    conn = get_connector(ds_row.source_type)
    cfg = dict(ds_row.config_json or {})

    ds_row.status = "syncing"
    ds_row.last_error = None
    await db.flush()

    try:
        df = conn.fetch_data(cfg)
        blob = get_blob_storage()
        blob.upload_df("datasets", dataset.blob_path, df, fmt="parquet")
        dataset.row_count = len(df)
        dataset.column_count = len(df.columns)
        prev_type = (dataset.schema_json or {}).get("__type__", "training")
        dataset.schema_json = {c: str(t) for c, t in df.dtypes.items()}
        dataset.schema_json["__type__"] = prev_type
        dataset.file_type = "parquet"
        ds_row.last_sync_at = datetime.utcnow()
        ds_row.status = "connected"
        ds_row.last_error = None
    except Exception as e:
        ds_row.status = "error"
        ds_row.last_error = str(e)[:4000]

    return dataset


@router.post("/{data_source_id}/sync")
async def sync_connection(
    data_source_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Re-fetch data from the external source and overwrite the blob snapshot."""
    result = await db.execute(
        select(DataSource)
        .options(selectinload(DataSource.dataset))
        .where(DataSource.id == data_source_id)
    )
    ds_row = result.scalar_one_or_none()
    if not ds_row or ds_row.dataset.tenant_id != user.tenant_id:
        raise HTTPException(404, "Data source not found")

    await perform_data_source_sync(db, ds_row)
    dataset = ds_row.dataset

    await db.commit()
    await db.refresh(dataset)
    await db.refresh(ds_row)

    from app.api.data import _dataset_meta

    return _dataset_meta(dataset, ds_row)


async def sync_dataset_snapshot_for_chat(
    db: AsyncSession,
    *,
    tenant_id: UUID,
    dataset_id: UUID,
):
    """
    Re-sync the first SQL-backed DataSource for a dataset (chat agent tool).
    Returns _dataset_meta dict or None if not found / not syncable.
    """
    from app.api.data import _dataset_meta

    result = await db.execute(
        select(DataSource)
        .options(selectinload(DataSource.dataset))
        .where(DataSource.dataset_id == dataset_id)
    )
    rows = result.scalars().all()
    ds_row = None
    for r in rows:
        if r.source_type in ("postgres", "sql"):
            ds_row = r
            break
    if not ds_row or ds_row.dataset.tenant_id != tenant_id:
        return None

    await perform_data_source_sync(db, ds_row)
    dataset = ds_row.dataset
    await db.commit()
    await db.refresh(dataset)
    await db.refresh(ds_row)
    return _dataset_meta(dataset, ds_row)


@router.get("/{data_source_id}/tables")
async def list_tables_saved(
    data_source_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List tables using stored connection config."""
    result = await db.execute(
        select(DataSource)
        .options(selectinload(DataSource.dataset))
        .where(DataSource.id == data_source_id)
    )
    ds_row = result.scalar_one_or_none()
    if not ds_row or ds_row.dataset.tenant_id != user.tenant_id:
        raise HTTPException(404, "Data source not found")
    if ds_row.source_type not in ("postgres", "sql"):
        raise HTTPException(400, "Not a SQL connection")
    conn = get_connector("postgres")
    cfg = dict(ds_row.config_json or {})
    try:
        tables = conn.list_tables(cfg)
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    return {"tables": tables}


@router.get("/{data_source_id}/preview")
async def preview_saved(
    data_source_id: UUID,
    table: str,
    rows: int = 50,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Preview a table using stored credentials."""
    result = await db.execute(
        select(DataSource)
        .options(selectinload(DataSource.dataset))
        .where(DataSource.id == data_source_id)
    )
    ds_row = result.scalar_one_or_none()
    if not ds_row or ds_row.dataset.tenant_id != user.tenant_id:
        raise HTTPException(404, "Data source not found")
    conn = get_connector("postgres")
    cfg = dict(ds_row.config_json or {})
    lim = max(1, min(rows, 500))
    try:
        df = conn.preview_table(cfg, table, limit=lim)
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    head = df.where(pd.notnull(df), None).to_dict(orient="records")
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": head,
        "shape": [len(df), len(df.columns)],
    }
