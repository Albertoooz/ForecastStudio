"""
Monitoring API — Model drift, data quality, alerts, metrics.
"""

from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db.models import MonitoringLog, ModelVersion, Schedule, User
from app.db.session import get_db

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class MetricPoint(BaseModel):
    timestamp: str
    metric_name: str
    metric_value: float
    threshold: float | None
    alert_triggered: bool


class DriftReport(BaseModel):
    model_version_id: UUID
    version: int
    metrics: list[MetricPoint]
    has_alerts: bool


class AlertResponse(BaseModel):
    id: UUID
    model_version_id: UUID | None
    metric_name: str
    metric_value: float
    threshold: float | None
    timestamp: str


class ScheduleRequest(BaseModel):
    model_version_id: UUID
    schedule_type: str = "retrain"  # retrain | forecast | monitor
    cron_expression: str = "0 2 * * 1"  # default: Mon 02:00
    config: dict = {}


class ScheduleResponse(BaseModel):
    id: UUID
    model_version_id: UUID | None
    schedule_type: str
    cron_expression: str
    is_active: bool
    next_run_at: str | None
    last_run_at: str | None


class MonitoringSummary(BaseModel):
    total_model_versions: int
    active_versions: int
    total_alerts_24h: int
    total_schedules: int
    active_schedules: int
    recent_metrics: list[MetricPoint]


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/summary", response_model=MonitoringSummary)
async def monitoring_summary(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """High-level monitoring dashboard data."""
    tenant_id = user.tenant_id

    # Model versions
    total_versions = (
        await db.execute(
            select(func.count(ModelVersion.id)).where(ModelVersion.tenant_id == tenant_id)
        )
    ).scalar() or 0

    active_versions = (
        await db.execute(
            select(func.count(ModelVersion.id)).where(
                ModelVersion.tenant_id == tenant_id,
                ModelVersion.is_active,
            )
        )
    ).scalar() or 0

    # Alerts in last 24h
    cutoff = datetime.utcnow() - timedelta(hours=24)
    alerts_24h = (
        await db.execute(
            select(func.count(MonitoringLog.id)).where(
                MonitoringLog.tenant_id == tenant_id,
                MonitoringLog.alert_triggered,
                MonitoringLog.created_at >= cutoff,
            )
        )
    ).scalar() or 0

    # Schedules
    total_schedules = (
        await db.execute(select(func.count(Schedule.id)).where(Schedule.tenant_id == tenant_id))
    ).scalar() or 0

    active_schedules = (
        await db.execute(
            select(func.count(Schedule.id)).where(
                Schedule.tenant_id == tenant_id,
                Schedule.is_active,
            )
        )
    ).scalar() or 0

    # Recent metrics (last 10)
    result = await db.execute(
        select(MonitoringLog)
        .where(MonitoringLog.tenant_id == tenant_id)
        .order_by(MonitoringLog.created_at.desc())
        .limit(10)
    )
    recent = result.scalars().all()

    return MonitoringSummary(
        total_model_versions=total_versions,
        active_versions=active_versions,
        total_alerts_24h=alerts_24h,
        total_schedules=total_schedules,
        active_schedules=active_schedules,
        recent_metrics=[
            MetricPoint(
                timestamp=str(m.created_at),
                metric_name=m.metric_name,
                metric_value=m.metric_value,
                threshold=m.threshold,
                alert_triggered=m.alert_triggered,
            )
            for m in recent
        ],
    )


@router.get("/drift/{model_version_id}", response_model=DriftReport)
async def get_drift_report(
    model_version_id: UUID,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get drift metrics for a model version."""
    version = await _get_version(model_version_id, user.tenant_id, db)
    cutoff = datetime.utcnow() - timedelta(days=days)

    result = await db.execute(
        select(MonitoringLog)
        .where(
            MonitoringLog.model_version_id == model_version_id,
            MonitoringLog.tenant_id == user.tenant_id,
            MonitoringLog.created_at >= cutoff,
        )
        .order_by(MonitoringLog.created_at)
    )
    logs = result.scalars().all()

    metrics = [
        MetricPoint(
            timestamp=str(m.created_at),
            metric_name=m.metric_name,
            metric_value=m.metric_value,
            threshold=m.threshold,
            alert_triggered=m.alert_triggered,
        )
        for m in logs
    ]

    return DriftReport(
        model_version_id=version.id,
        version=version.version,
        metrics=metrics,
        has_alerts=any(m.alert_triggered for m in metrics),
    )


@router.get("/alerts", response_model=list[AlertResponse])
async def list_alerts(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    hours: int = Query(24, ge=1, le=720),
    limit: int = Query(50, ge=1, le=200),
):
    """List recent alerts."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    result = await db.execute(
        select(MonitoringLog)
        .where(
            MonitoringLog.tenant_id == user.tenant_id,
            MonitoringLog.alert_triggered,
            MonitoringLog.created_at >= cutoff,
        )
        .order_by(MonitoringLog.created_at.desc())
        .limit(limit)
    )
    alerts = result.scalars().all()

    return [
        AlertResponse(
            id=a.id,
            model_version_id=a.model_version_id,
            metric_name=a.metric_name,
            metric_value=a.metric_value,
            threshold=a.threshold,
            timestamp=str(a.created_at),
        )
        for a in alerts
    ]


# ── Scheduling ──────────────────────────────────────────────────────────────


@router.post("/schedules", response_model=ScheduleResponse, status_code=201)
async def create_schedule(
    body: ScheduleRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Create a new schedule (retrain / forecast / monitor)."""
    await _get_version(body.model_version_id, user.tenant_id, db)

    schedule = Schedule(
        tenant_id=user.tenant_id,
        model_version_id=body.model_version_id,
        schedule_type=body.schedule_type,
        cron_expression=body.cron_expression,
        is_active=True,
        config_json=body.config,
    )
    db.add(schedule)
    await db.flush()

    return _schedule_response(schedule)


@router.get("/schedules", response_model=list[ScheduleResponse])
async def list_schedules(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    active_only: bool = Query(False),
):
    """List all schedules."""
    q = select(Schedule).where(Schedule.tenant_id == user.tenant_id)
    if active_only:
        q = q.where(Schedule.is_active)
    q = q.order_by(Schedule.created_at.desc())

    result = await db.execute(q)
    schedules = result.scalars().all()
    return [_schedule_response(s) for s in schedules]


@router.patch("/schedules/{schedule_id}/toggle", response_model=ScheduleResponse)
async def toggle_schedule(
    schedule_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Activate / deactivate a schedule."""
    result = await db.execute(
        select(Schedule).where(
            Schedule.id == schedule_id,
            Schedule.tenant_id == user.tenant_id,
        )
    )
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise HTTPException(404, "Schedule not found")

    schedule.is_active = not schedule.is_active
    await db.flush()

    return _schedule_response(schedule)


@router.delete("/schedules/{schedule_id}", status_code=204)
async def delete_schedule(
    schedule_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a schedule."""
    result = await db.execute(
        select(Schedule).where(
            Schedule.id == schedule_id,
            Schedule.tenant_id == user.tenant_id,
        )
    )
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise HTTPException(404, "Schedule not found")
    await db.delete(schedule)


# ── Internal ─────────────────────────────────────────────────────────────────


async def _get_version(version_id: UUID, tenant_id: UUID, db: AsyncSession) -> ModelVersion:
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.id == version_id,
            ModelVersion.tenant_id == tenant_id,
        )
    )
    version = result.scalar_one_or_none()
    if not version:
        raise HTTPException(404, "Model version not found")
    return version


def _schedule_response(s: Schedule) -> ScheduleResponse:
    return ScheduleResponse(
        id=s.id,
        model_version_id=s.model_version_id,
        schedule_type=s.schedule_type,
        cron_expression=s.cron_expression,
        is_active=s.is_active,
        next_run_at=str(s.next_run_at) if s.next_run_at else None,
        last_run_at=str(s.last_run_at) if s.last_run_at else None,
    )
