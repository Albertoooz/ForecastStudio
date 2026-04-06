"""
SQLAlchemy ORM models — multi-tenant forecasting platform.

All tenant-scoped tables include tenant_id for Row-Level Security.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


# ── Helpers ──────────────────────────────────────────────────────────────────


def _uuid():
    return uuid.uuid4()


def _now():
    return datetime.utcnow()


# ── Tenants & Users ──────────────────────────────────────────────────────────


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    plan: Mapped[str] = mapped_column(String(50), default="free")  # free / pro / enterprise
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    users: Mapped[list["User"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[str] = mapped_column(String(50), default="member")  # admin / member / viewer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    tenant: Mapped["Tenant"] = relationship(back_populates="users")


# ── Datasets & Data Sources ─────────────────────────────────────────────────


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    blob_path: Mapped[str] = mapped_column(String(1024), nullable=True)
    file_type: Mapped[str] = mapped_column(String(20), default="csv")  # csv / xlsx / parquet
    schema_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Forecast config
    datetime_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    group_columns: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    frequency: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    tenant: Mapped["Tenant"] = relationship(back_populates="datasets")
    data_sources: Mapped[list["DataSource"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DataSource(Base):
    """External data connector configuration (SQL / API / file upload)."""

    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)  # file / sql / api
    config_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    schedule_cron: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    dataset: Mapped["Dataset"] = relationship(back_populates="data_sources")


# ── Model Runs & Versions ───────────────────────────────────────────────────


class ModelRun(Base):
    """A single training run — stores config, metrics, and blob path to serialized model."""

    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending/running/done/failed
    model_type: Mapped[str] = mapped_column(String(50), default="auto")
    config_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    features_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_blob_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    best_model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    horizon: Mapped[int] = mapped_column(Integer, default=10)
    gap: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    dataset: Mapped["Dataset"] = relationship(back_populates="model_runs")
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan"
    )
    forecasts: Mapped[list["Forecast"]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan"
    )
    pipeline_steps: Mapped[list["PipelineStep"]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan"
    )


class ModelVersion(Base):
    """Versioned model in the registry — one run can produce one active version."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    model_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    model_run: Mapped["ModelRun"] = relationship(back_populates="model_versions")


# ── Forecasts ────────────────────────────────────────────────────────────────


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    model_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    predictions_blob_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    predictions_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    model_run: Mapped["ModelRun"] = relationship(back_populates="forecasts")


# ── Pipeline Steps (agent audit trail) ───────────────────────────────────────


class PipelineStep(Base):
    __tablename__ = "pipeline_steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    model_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    step_name: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    decision_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    model_run: Mapped["ModelRun"] = relationship(back_populates="pipeline_steps")


# ── Scheduling ───────────────────────────────────────────────────────────────


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    schedule_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # retraining / forecast / data_sync
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    config_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ── Monitoring ───────────────────────────────────────────────────────────────


class MonitoringLog(Base):
    __tablename__ = "monitoring_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    alert_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    messages_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
