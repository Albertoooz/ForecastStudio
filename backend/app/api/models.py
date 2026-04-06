"""
Models API — Train, status, registry, promote, predict.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db.models import ModelRun, ModelVersion, Forecast, User, PipelineStep
from app.db.session import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────────────


class TrainRequest(BaseModel):
    dataset_id: UUID
    model_type: str = "auto"  # auto | prophet | lightgbm | naive | linear
    horizon: int = 12
    gap: int = 0
    config: dict = {}


class ModelRunResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    status: str
    model_type: str
    horizon: int
    gap: int
    best_model_name: str | None
    metrics: dict | None
    duration_seconds: float | None
    created_at: str


class ModelRunListResponse(BaseModel):
    runs: list[ModelRunResponse]
    total: int


class ModelVersionResponse(BaseModel):
    id: UUID
    model_run_id: UUID
    version: int
    is_active: bool
    promoted_at: str | None
    created_at: str


class PipelineStepResponse(BaseModel):
    step_name: str
    agent_name: str | None
    status: str
    message: str | None
    duration_seconds: float | None


class PredictRequest(BaseModel):
    model_run_id: UUID
    horizon: int | None = None


class PredictResponse(BaseModel):
    forecast_id: UUID
    model_run_id: UUID
    horizon: int
    predictions: list[dict] | None


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/train", response_model=ModelRunResponse, status_code=202)
async def start_training(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Kick off a model training run (async via Celery)."""
    model_run = ModelRun(
        tenant_id=user.tenant_id,
        dataset_id=body.dataset_id,
        status="queued",
        model_type=body.model_type,
        config_json=body.config,
        horizon=body.horizon,
        gap=body.gap,
    )
    db.add(model_run)
    await db.flush()

    # Dispatch Celery training task
    try:
        from app.tasks.training import run_agent_pipeline

        run_agent_pipeline.delay(str(model_run.id))
    except Exception as e:
        logger.exception("Celery dispatch failed for model_run %s", model_run.id)
        model_run.status = "failed"
        model_run.error_message = f"Queue dispatch failed: {e}"[:2000]
        await db.flush()

    return _model_run_response(model_run)


@router.get("/runs", response_model=ModelRunListResponse)
async def list_runs(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    dataset_id: UUID | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List model training runs, optionally filtered."""
    q = select(ModelRun).where(ModelRun.tenant_id == user.tenant_id)
    count_q = select(func.count(ModelRun.id)).where(ModelRun.tenant_id == user.tenant_id)

    if dataset_id:
        q = q.where(ModelRun.dataset_id == dataset_id)
        count_q = count_q.where(ModelRun.dataset_id == dataset_id)
    if status:
        q = q.where(ModelRun.status == status)
        count_q = count_q.where(ModelRun.status == status)

    q = q.order_by(ModelRun.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(q)
    runs = result.scalars().all()
    total = (await db.execute(count_q)).scalar() or 0

    return ModelRunListResponse(
        runs=[_model_run_response(r) for r in runs],
        total=total,
    )


@router.get("/runs/{run_id}", response_model=ModelRunResponse)
async def get_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get details of a specific model run."""
    run = await _get_run(run_id, user.tenant_id, db)
    return _model_run_response(run)


@router.get("/runs/{run_id}/pipeline", response_model=list[PipelineStepResponse])
async def get_pipeline_steps(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get pipeline steps for a model run."""
    await _get_run(run_id, user.tenant_id, db)  # authz check
    result = await db.execute(
        select(PipelineStep)
        .where(PipelineStep.model_run_id == run_id)
        .order_by(PipelineStep.created_at)
    )
    steps = result.scalars().all()
    return [
        PipelineStepResponse(
            step_name=s.step_name,
            agent_name=s.agent_name,
            status=s.status,
            message=s.message,
            duration_seconds=s.duration_seconds,
        )
        for s in steps
    ]


@router.post("/runs/{run_id}/promote", response_model=ModelVersionResponse, status_code=201)
async def promote_model(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Promote a completed model run to an active model version."""
    run = await _get_run(run_id, user.tenant_id, db)
    if run.status != "completed":
        raise HTTPException(400, "Only completed runs can be promoted")

    # Deactivate previous active versions for same tenant
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.tenant_id == user.tenant_id,
            ModelVersion.is_active,
        )
    )
    for v in result.scalars().all():
        v.is_active = False

    # Get next version number
    max_ver = await db.execute(
        select(func.max(ModelVersion.version)).where(ModelVersion.tenant_id == user.tenant_id)
    )
    next_version = (max_ver.scalar() or 0) + 1

    version = ModelVersion(
        model_run_id=run_id,
        tenant_id=user.tenant_id,
        version=next_version,
        is_active=True,
    )
    db.add(version)
    await db.flush()

    return ModelVersionResponse(
        id=version.id,
        model_run_id=version.model_run_id,
        version=version.version,
        is_active=version.is_active,
        promoted_at=str(version.promoted_at) if version.promoted_at else None,
        created_at=str(version.created_at),
    )


@router.get("/registry", response_model=list[ModelVersionResponse])
async def list_versions(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    active_only: bool = Query(False),
):
    """List model versions (registry)."""
    q = select(ModelVersion).where(ModelVersion.tenant_id == user.tenant_id)
    if active_only:
        q = q.where(ModelVersion.is_active)
    q = q.order_by(ModelVersion.version.desc())

    result = await db.execute(q)
    versions = result.scalars().all()
    return [
        ModelVersionResponse(
            id=v.id,
            model_run_id=v.model_run_id,
            version=v.version,
            is_active=v.is_active,
            promoted_at=str(v.promoted_at) if v.promoted_at else None,
            created_at=str(v.created_at),
        )
        for v in versions
    ]


@router.post("/predict", response_model=PredictResponse, status_code=201)
async def predict(
    body: PredictRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Generate forecast using a trained model."""
    run = await _get_run(body.model_run_id, user.tenant_id, db)
    if run.status != "completed":
        raise HTTPException(400, "Model run must be completed before predicting")

    horizon = body.horizon or run.horizon

    # Return the most recent forecast if it already exists
    existing = await db.execute(
        select(Forecast)
        .where(Forecast.model_run_id == run.id)
        .order_by(Forecast.created_at.desc())
        .limit(1)
    )
    existing_fc = existing.scalar_one_or_none()
    if existing_fc and existing_fc.predictions_json:
        preds = existing_fc.predictions_json
        return PredictResponse(
            forecast_id=existing_fc.id,
            model_run_id=run.id,
            horizon=existing_fc.horizon,
            predictions=preds if isinstance(preds, list) else [preds],
        )

    forecast = Forecast(
        model_run_id=run.id,
        tenant_id=user.tenant_id,
        horizon=horizon,
    )
    db.add(forecast)
    await db.flush()

    return PredictResponse(
        forecast_id=forecast.id,
        model_run_id=run.id,
        horizon=horizon,
        predictions=None,
    )


# ── Internal ─────────────────────────────────────────────────────────────────


async def _get_run(run_id: UUID, tenant_id: UUID, db: AsyncSession) -> ModelRun:
    result = await db.execute(
        select(ModelRun).where(ModelRun.id == run_id, ModelRun.tenant_id == tenant_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(404, "Model run not found")
    return run


def _model_run_response(r: ModelRun) -> ModelRunResponse:
    return ModelRunResponse(
        id=r.id,
        dataset_id=r.dataset_id,
        status=r.status,
        model_type=r.model_type,
        horizon=r.horizon or 0,
        gap=r.gap or 0,
        best_model_name=r.best_model_name,
        metrics=r.metrics_json,
        duration_seconds=r.duration_seconds,
        created_at=str(r.created_at),
    )
