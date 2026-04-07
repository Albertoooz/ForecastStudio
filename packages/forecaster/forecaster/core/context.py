"""
Context Window — explicit state object passed between agents.

This is the single source of truth for the entire forecast pipeline.
No hidden LLM memory, no global state. Every agent receives and returns this.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
#  Agent Decision (append-only log entry)
# ---------------------------------------------------------------------------


class AgentDecision(BaseModel):
    """
    Structured, auditable record of a single agent decision.

    Every action taken by any agent in the pipeline is captured here,
    including resource estimates so the MemoryManager can enforce budgets.
    """

    agent_name: str
    decision_type: Literal[
        "memory_reservation",
        "data_cleaning",
        "data_analysis",
        "external_data_join",
        "feature_engineering",
        "model_selection",
        "model_training",
        "forecast",
        "validation",
        "error",
    ]
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""
    requires_confirmation: bool = False
    estimated_compute_seconds: float = 0.0
    estimated_memory_mb: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
#  Resource Budget
# ---------------------------------------------------------------------------


class ResourceBudget(BaseModel):
    """Pre-allocated compute budget per forecast request."""

    memory_budget_mb: int = 512
    compute_budget_seconds: float = 300.0
    consumed_memory_mb: int = 0
    consumed_compute_seconds: float = 0.0

    @property
    def remaining_memory_mb(self) -> int:
        return self.memory_budget_mb - self.consumed_memory_mb

    @property
    def remaining_compute_seconds(self) -> float:
        return self.compute_budget_seconds - self.consumed_compute_seconds

    def can_reserve_memory(self, mb: int) -> bool:
        return self.remaining_memory_mb >= mb

    def can_reserve_compute(self, seconds: float) -> bool:
        return self.remaining_compute_seconds >= seconds

    def reserve_memory(self, mb: int) -> bool:
        """Reserve memory. Returns False if budget exceeded."""
        if not self.can_reserve_memory(mb):
            return False
        self.consumed_memory_mb += mb
        return True

    def reserve_compute(self, seconds: float) -> bool:
        """Reserve compute time. Returns False if budget exceeded."""
        if not self.can_reserve_compute(seconds):
            return False
        self.consumed_compute_seconds += seconds
        return True


# ---------------------------------------------------------------------------
#  Data Analysis Results (cached inside context)
# ---------------------------------------------------------------------------


class DataProfile(BaseModel):
    """Profile of a dataset produced by DataAnalyzerAgent."""

    n_rows: int = 0
    n_columns: int = 0
    frequency: str | None = None
    missing_pct: float = 0.0
    outlier_pct: float = 0.0
    has_trend: bool = False
    has_seasonality: bool = False
    seasonality_period: int | None = None
    date_range: tuple[str, str] | None = None
    column_types: dict[str, str] = Field(default_factory=dict)
    numeric_columns: list[str] = Field(default_factory=list)
    datetime_column: str | None = None
    target_column: str | None = None
    group_columns: list[str] = Field(default_factory=list)
    cleaning_recommendations: list[dict[str, Any]] = Field(default_factory=list)


class FeatureSpec(BaseModel):
    """Specification for a single engineered feature."""

    name: str
    feature_type: Literal[
        "lag", "rolling_mean", "rolling_std", "date_part", "holiday", "external", "custom"
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)
    estimated_impact: float | None = None  # 0-1 estimated SHAP importance


class ModelSpec(BaseModel):
    """Specification for a selected model."""

    model_type: str  # e.g. "lightgbm_default", "prophet", "simple_ewm", "mlforecast_global"
    reasoning: str = ""
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    tuning_ranges: dict[str, Any] = Field(default_factory=dict)
    estimated_compute_seconds: float = 0.0
    estimated_memory_mb: int = 0


# ---------------------------------------------------------------------------
#  Context Window
# ---------------------------------------------------------------------------


class ContextWindow(BaseModel):
    """
    Explicit state object passed between agents.

    This replaces hidden LLM memory with a deterministic,
    inspectable, serializable context.
    """

    # Identity
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Resource budget
    budget: ResourceBudget = Field(default_factory=ResourceBudget)

    # Data registry — keyed by semantic name ("sales", "holidays_pl")
    # NOTE: DataFrames stored here but excluded from serialization
    data_registry: dict[str, Any] = Field(default_factory=dict)

    # Source file info
    source_file: str | None = None
    target_column: str | None = None
    datetime_column: str | None = None
    group_columns: list[str] = Field(default_factory=list)
    horizon: int = 7
    gap: int = 0

    # Results of each phase (populated by agents)
    data_profile: DataProfile | None = None
    feature_specs: list[FeatureSpec] = Field(default_factory=list)
    model_spec: ModelSpec | None = None

    # Forecast output
    forecast_result: dict[str, Any] | None = None

    # Append-only decision log (audit trail)
    decision_log: list[AgentDecision] = Field(default_factory=list)

    # Pipeline state
    current_phase: str = "initialized"
    completed_phases: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    pending_decision: AgentDecision | None = None

    class Config:
        arbitrary_types_allowed = True

    # -- Convenience methods -----------------------------------------------

    def log_decision(self, decision: AgentDecision) -> None:
        """Append a decision to the audit trail (immutable once added)."""
        self.decision_log.append(decision)

    def reserve_memory(self, mb: int, agent_name: str = "unknown") -> bool:
        """Reserve memory, logging the decision."""
        ok = self.budget.reserve_memory(mb)
        self.log_decision(
            AgentDecision(
                agent_name=agent_name,
                decision_type="memory_reservation",
                action="reserve_memory",
                parameters={"mb": mb, "granted": ok},
                reasoning=f"Requested {mb}MB, remaining {self.budget.remaining_memory_mb}MB",
            )
        )
        return ok

    def reserve_compute(self, seconds: float, agent_name: str = "unknown") -> bool:
        """Reserve compute seconds, logging the decision."""
        ok = self.budget.reserve_compute(seconds)
        self.log_decision(
            AgentDecision(
                agent_name=agent_name,
                decision_type="memory_reservation",
                action="reserve_compute",
                parameters={"seconds": seconds, "granted": ok},
                reasoning=f"Requested {seconds}s, remaining {self.budget.remaining_compute_seconds}s",
            )
        )
        return ok

    def register_data(self, name: str, df: pl.DataFrame) -> None:
        """Register a DataFrame in the data registry."""
        self.data_registry[name] = df

    def get_data(self, name: str) -> pl.DataFrame | None:
        """Get DataFrame from registry."""
        return self.data_registry.get(name)

    def get_primary_data(self) -> pl.DataFrame | None:
        """Get the primary (first registered) DataFrame."""
        if not self.data_registry:
            return None
        first_key = next(iter(self.data_registry))
        return self.data_registry[first_key]

    def advance_phase(self, phase: str) -> None:
        """Advance the pipeline to a new phase."""
        if self.current_phase != "initialized":
            self.completed_phases.append(self.current_phase)
        self.current_phase = phase

    def get_cost_estimate(self) -> dict[str, float]:
        """Compute cost estimate based on consumed resources."""
        compute_cost = self.budget.consumed_compute_seconds * 0.00001156  # t3.medium
        return {
            "compute_cost_usd": round(compute_cost, 6),
            "llm_cost_usd": 0.0,  # No LLM in core pipeline
            "total_cost_usd": round(compute_cost, 6),
        }

    def get_audit_summary(self) -> list[dict[str, Any]]:
        """Get human-readable audit trail."""
        return [
            {
                "agent": d.agent_name,
                "action": d.action,
                "confidence": d.confidence,
                "reasoning": d.reasoning,
                "requires_confirmation": d.requires_confirmation,
                "timestamp": d.timestamp.isoformat(),
            }
            for d in self.decision_log
        ]
