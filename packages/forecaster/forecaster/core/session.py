"""Session state management for forecasting platform."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ColumnInfo(BaseModel):
    """Information about a single column in the dataset."""

    name: str
    dtype: str
    n_missing: int = 0
    n_unique: int = 0
    sample_values: list[str] = Field(default_factory=list)
    is_datetime: bool = False
    is_numeric: bool = False


class DataInfo(BaseModel):
    """Information about uploaded data."""

    filepath: Path
    filename: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    datetime_column: str | None = None
    target_column: str | None = None
    group_by_column: str | None = None  # For multivariate forecasting (legacy, single column)
    group_by_columns: list[str] = Field(
        default_factory=list
    )  # For multivariate forecasting (multiple columns)
    frequency: str | None = None  # D, W, M, etc.
    n_rows: int = 0
    date_range: tuple[str, str] | None = None
    issues: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)  # Questions for user


class ModelConfig(BaseModel):
    """Configuration for forecasting model."""

    model_type: str = "linear"  # naive, linear, arima, etc.
    horizon: int = 7
    gap: int = 0  # Periods between forecast execution and first forecast point
    frequency: str = "D"
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""  # Why this model was chosen


class ForecastResult(BaseModel):
    """Result of a forecast."""

    predictions: list[float]
    dates: list[str]
    model_name: str
    horizon: int = 7  # Actual forecast horizon (periods per group)
    metrics: dict[str, float] = Field(default_factory=dict)
    confidence_lower: list[float] | None = None
    confidence_upper: list[float] | None = None
    group_info: dict[str, Any] | None = None  # Info about groups if multivariate

    # Advanced analysis fields
    residuals: list[float] | None = None  # Training residuals for analysis
    residual_dates: list[str] | None = None  # Dates corresponding to residuals
    actual_values: list[float] | None = None  # Actual values (for residual calculation)
    baseline_metrics: dict[str, float] | None = None  # Naive baseline comparison
    warnings: list[str] = Field(default_factory=list)  # Automatic warnings
    trust_indicators: list[str] = Field(default_factory=list)  # Positive trust signals
    data_quality: dict[str, Any] | None = None  # Data quality summary
    health_score: float | None = None  # Overall model health (0-100)

    # Per-group analysis (when separate models per group)
    per_group_metrics: dict[str, dict] | None = None  # {group_id: {metric: value}}
    per_group_health_scores: dict[str, float] | None = None  # {group_id: score}
    per_group_warnings: dict[str, list] | None = None  # {group_id: [warnings]}
    per_group_residuals: dict[str, list] | None = None  # {group_id: [residuals]}
    per_group_residual_dates: dict[str, list] | None = None  # {group_id: [dates]}

    class Config:
        arbitrary_types_allowed = False


class ForecastSession(BaseModel):
    """Session state for a forecasting conversation."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    # Data state
    uploaded_file: Path | None = None
    data_info: DataInfo | None = None
    current_df: Any | None = None

    # Model state
    forecast_config: ModelConfig | None = None
    is_trained: bool = False
    trained_model: Any | None = None  # Store trained model for analysis

    # Forecast state
    forecast_result: ForecastResult | None = None

    # Conversation state
    messages: list[Message] = Field(default_factory=list)

    # Pending questions from agents
    pending_questions: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def add_message(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))

    def get_conversation_context(self, max_messages: int = 10) -> list[dict]:
        """Get recent messages as context for LLM."""
        recent = self.messages[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]

    def clear_forecast(self) -> None:
        """Clear forecast state (when data changes)."""
        self.forecast_result = None
        self.is_trained = False
        self.forecast_config = None


# ---------------------------------------------------------------------------
#  Session Manager — global memory cap and ContextWindow tracking
# ---------------------------------------------------------------------------


class SessionManager:
    """
    Manages active sessions with a global memory cap.

    Tracks:
      - active_sessions: dict[session_id, ContextWindow]
      - memory_usage_total_mb: global cap (default 4096 MB)
      - Rejects new sessions if cap exceeded
    """

    def __init__(self, global_memory_cap_mb: int = 4096):
        self.global_memory_cap_mb = global_memory_cap_mb
        self.active_sessions: dict[str, Any] = {}  # session_id → ContextWindow
        self._legacy_sessions: dict[str, ForecastSession] = {}

    @property
    def memory_usage_total_mb(self) -> int:
        """Total memory consumed across all active sessions."""
        total = 0
        for ctx in self.active_sessions.values():
            if hasattr(ctx, "budget"):
                total += ctx.budget.consumed_memory_mb
        return total

    @property
    def remaining_memory_mb(self) -> int:
        return self.global_memory_cap_mb - self.memory_usage_total_mb

    def can_create_session(self, estimated_memory_mb: int = 512) -> bool:
        """Check if a new session can be created within the global cap."""
        return self.remaining_memory_mb >= estimated_memory_mb

    def create_context_session(
        self,
        user_id: str = "default",
        memory_budget_mb: int = 512,
        compute_budget_seconds: float = 300.0,
    ) -> Any | None:
        """
        Create a new ContextWindow session.

        Returns None if global memory cap would be exceeded.
        """
        if not self.can_create_session(memory_budget_mb):
            return None

        from forecaster.core.context import ContextWindow, ResourceBudget

        ctx = ContextWindow(
            user_id=user_id,
            budget=ResourceBudget(
                memory_budget_mb=memory_budget_mb,
                compute_budget_seconds=compute_budget_seconds,
            ),
        )
        self.active_sessions[ctx.trace_id] = ctx
        return ctx

    def get_context(self, session_id: str) -> Any | None:
        """Get ContextWindow by session/trace ID."""
        return self.active_sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove a session and free resources."""
        self.active_sessions.pop(session_id, None)
        self._legacy_sessions.pop(session_id, None)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all active sessions."""
        return {
            "n_active": len(self.active_sessions),
            "memory_used_mb": self.memory_usage_total_mb,
            "memory_cap_mb": self.global_memory_cap_mb,
            "memory_remaining_mb": self.remaining_memory_mb,
            "sessions": [
                {
                    "trace_id": sid,
                    "user_id": getattr(ctx, "user_id", "?"),
                    "phase": getattr(ctx, "current_phase", "?"),
                    "memory_mb": getattr(getattr(ctx, "budget", None), "consumed_memory_mb", 0),
                }
                for sid, ctx in self.active_sessions.items()
            ],
        }
