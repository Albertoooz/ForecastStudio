"""
Data API — Upload, preview, transform, list, delete datasets.
"""

import io
from datetime import datetime
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.auth import get_current_user
from app.db.models import DataSource, Dataset, User
from app.db.session import get_db
from app.storage.blob import get_blob_storage

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class DatasetMeta(BaseModel):
    id: UUID
    name: str
    rows: int | None
    columns: int | None
    datetime_column: str | None
    target_column: str | None
    group_columns: list[str] | None
    frequency: str | None
    schema_columns: list[str] | None
    dataset_type: str = "training"  # "training" | "future_variables"
    linked_dataset_id: str | None = None
    created_at: str
    # Optional — set when dataset is backed by a DataSource (Postgres, etc.)
    data_source_id: UUID | None = None
    source_type: str | None = None
    sync_status: str | None = None
    last_sync_at: str | None = None
    last_error: str | None = None
    query_or_table: str | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetMeta]
    total: int


class PreviewResponse(BaseModel):
    columns: list[str]
    dtypes: dict[str, str]
    head: list[dict]
    shape: list[int]
    stats: dict


class ColumnConfigRequest(BaseModel):
    datetime_column: str | None = None
    target_column: str | None = None
    group_columns: list[str] | None = None
    frequency: str | None = None
    dataset_type: str | None = None  # "training" | "future_variables"
    linked_dataset_id: str | None = None  # ID of training dataset this FV belongs to


class TransformRequest(BaseModel):
    operation: str
    params: dict = {}


class TransformResponse(BaseModel):
    success: bool
    message: str
    rows_before: int | None = None
    rows_after: int | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_upload(file: UploadFile) -> tuple[pd.DataFrame, bytes]:
    content = file.file.read()
    name = file.filename or ""
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    elif name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content))
    return df, content


def _dataset_meta(d: Dataset, data_source: DataSource | None = None) -> DatasetMeta:
    schema = d.schema_json or {}
    schema_cols = [k for k in schema.keys() if not k.startswith("__")] if schema else None
    group_cols = (
        d.group_columns
        if isinstance(d.group_columns, list)
        else ([d.group_columns] if isinstance(d.group_columns, str) and d.group_columns else None)
    )
    ds = data_source
    if ds is None and getattr(d, "data_sources", None):
        srcs = d.data_sources
        ds = srcs[0] if srcs else None
    return DatasetMeta(
        id=d.id,
        name=d.name,
        rows=d.row_count,
        columns=d.column_count,
        datetime_column=d.datetime_column,
        target_column=d.target_column,
        group_columns=group_cols,
        frequency=d.frequency,
        schema_columns=schema_cols,
        dataset_type=schema.get("__type__", "training"),
        linked_dataset_id=schema.get("__linked_dataset_id__"),
        created_at=str(d.created_at),
        data_source_id=ds.id if ds else None,
        source_type=ds.source_type if ds else None,
        sync_status=ds.status if ds else None,
        last_sync_at=str(ds.last_sync_at) if ds and ds.last_sync_at else None,
        last_error=ds.last_error if ds else None,
        query_or_table=ds.query_or_table if ds else None,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/upload", response_model=DatasetMeta, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload CSV / Excel / Parquet → save to blob storage, store metadata in DB."""
    df, content = _read_upload(file)

    blob = get_blob_storage()
    blob_path = f"{user.tenant_id}/{file.filename}"
    blob.upload_bytes("datasets", blob_path, content)

    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Auto-detect datetime column
    datetime_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "ds" == col.lower():
            try:
                pd.to_datetime(df[col].head(10))
                datetime_col = col
                break
            except Exception:
                pass

    dataset = Dataset(
        tenant_id=user.tenant_id,
        name=file.filename or "untitled",
        blob_path=blob_path,
        schema_json=schema,
        row_count=len(df),
        column_count=len(df.columns),
        datetime_column=datetime_col,
    )
    db.add(dataset)
    await db.flush()

    file_ds = DataSource(
        dataset_id=dataset.id,
        source_type="file",
        config_json={"original_filename": file.filename},
        query_or_table=file.filename,
        status="connected",
        last_sync_at=datetime.utcnow(),
    )
    db.add(file_ds)
    await db.flush()

    return _dataset_meta(dataset, file_ds)


@router.get("/list", response_model=DatasetListResponse)
async def list_datasets(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    q = (
        select(Dataset)
        .where(Dataset.tenant_id == user.tenant_id)
        .options(selectinload(Dataset.data_sources))
        .order_by(Dataset.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(q)
    datasets = result.scalars().all()

    count_q = select(func.count(Dataset.id)).where(Dataset.tenant_id == user.tenant_id)
    total = (await db.execute(count_q)).scalar() or 0

    return DatasetListResponse(
        datasets=[_dataset_meta(d) for d in datasets],
        total=total,
    )


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
async def preview_dataset(
    dataset_id: UUID,
    rows: int = Query(20, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Load dataset from blob storage and return a preview."""
    dataset = await _get_dataset(dataset_id, user.tenant_id, db)

    blob = get_blob_storage()
    try:
        df = blob.download_df("datasets", dataset.blob_path)
    except FileNotFoundError:
        raise HTTPException(404, "Dataset file not found in storage")

    head = df.head(rows).where(pd.notnull(df.head(rows)), None).to_dict(orient="records")

    # Basic stats for numeric columns
    stats: dict = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].describe()
        stats[col] = {k: round(float(v), 4) for k, v in s.items()}

    return PreviewResponse(
        columns=list(df.columns),
        dtypes={col: str(dt) for col, dt in df.dtypes.items()},
        head=head,
        shape=[len(df), len(df.columns)],
        stats=stats,
    )


@router.patch("/{dataset_id}/columns", response_model=DatasetMeta)
async def configure_columns(
    dataset_id: UUID,
    body: ColumnConfigRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Configure datetime / target / group columns and frequency."""
    dataset = await _get_dataset(dataset_id, user.tenant_id, db)

    if body.datetime_column is not None:
        dataset.datetime_column = body.datetime_column
    if body.target_column is not None:
        dataset.target_column = body.target_column
    if body.group_columns is not None:
        dataset.group_columns = body.group_columns
    if body.frequency is not None:
        dataset.frequency = body.frequency

    # Store type metadata inside schema_json (no schema migration needed)
    if body.dataset_type is not None or body.linked_dataset_id is not None:
        schema = dict(dataset.schema_json or {})
        if body.dataset_type is not None:
            schema["__type__"] = body.dataset_type
        if body.linked_dataset_id is not None:
            schema["__linked_dataset_id__"] = body.linked_dataset_id
        dataset.schema_json = schema

    await db.flush()
    return _dataset_meta(dataset)


@router.post("/{dataset_id}/transform", response_model=TransformResponse)
async def transform_dataset(
    dataset_id: UUID,
    body: TransformRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Apply a transformation to the dataset and save it back to blob storage."""
    dataset = await _get_dataset(dataset_id, user.tenant_id, db)

    blob = get_blob_storage()
    try:
        df = blob.download_df("datasets", dataset.blob_path)
    except FileNotFoundError:
        raise HTTPException(404, "Dataset file not found in storage")

    rows_before = len(df)

    try:
        op = body.operation
        p = body.params

        if op == "fill_missing":
            col = p.get("column")
            method = p.get("method", "ffill")
            if col and col in df.columns:
                df[col] = (
                    df[col].fillna(method=method)
                    if method in ("ffill", "bfill")
                    else df[col].fillna(float(method))
                )
            else:
                df = df.fillna(method=method)

        elif op == "drop_missing":
            col = p.get("column")
            df = df.dropna(subset=[col]) if col else df.dropna()

        elif op == "clip_negative":
            col = p.get("column", dataset.target_column)
            if col and col in df.columns:
                df[col] = df[col].clip(lower=0)

        elif op == "filter_date_range":
            col = p.get("column", dataset.datetime_column)
            start = p.get("start")
            end = p.get("end")
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col])
                if start:
                    df = df[df[col] >= pd.Timestamp(start)]
                if end:
                    df = df[df[col] <= pd.Timestamp(end)]

        elif op == "rename_column":
            old = p.get("old_name")
            new = p.get("new_name")
            if old and new and old in df.columns:
                df = df.rename(columns={old: new})
                if dataset.datetime_column == old:
                    dataset.datetime_column = new
                if dataset.target_column == old:
                    dataset.target_column = new

        elif op == "drop_column":
            col = p.get("column")
            if col and col in df.columns:
                df = df.drop(columns=[col])

        else:
            raise HTTPException(400, f"Unknown operation: {op}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Transform failed: {e}")

    blob.upload_df("datasets", dataset.blob_path, df)
    dataset.row_count = len(df)
    dataset.column_count = len(df.columns)
    dataset.schema_json = {col: str(dt) for col, dt in df.dtypes.items()}
    await db.flush()

    return TransformResponse(
        success=True,
        message=f"Operation '{body.operation}' applied",
        rows_before=rows_before,
        rows_after=len(df),
    )


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a dataset (DB record + blob)."""
    dataset = await _get_dataset(dataset_id, user.tenant_id, db)

    blob = get_blob_storage()
    try:
        blob.delete("datasets", dataset.blob_path)
    except Exception:
        pass

    await db.delete(dataset)


# ── Internal ─────────────────────────────────────────────────────────────────


async def _get_dataset(dataset_id: UUID, tenant_id: UUID, db: AsyncSession) -> Dataset:
    result = await db.execute(
        select(Dataset)
        .options(selectinload(Dataset.data_sources))
        .where(Dataset.id == dataset_id, Dataset.tenant_id == tenant_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset
