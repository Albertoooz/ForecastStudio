"""Utilities for forecaster - observability, error recovery, monitoring."""

from forecaster.utils.error_recovery import (
    FatalError,
    RecoverableError,
    RetryConfig,
    TransientError,
    retry_with_backoff,
    suggest_recovery_action,
    with_error_recovery,
)
from forecaster.utils.monitoring import ForecastMonitor, compute_cost
from forecaster.utils.observability import (
    end_trace,
    flush_langfuse,
    get_current_trace_id,
    get_langfuse_client,
    get_tracer,
    langfuse_observation,
    log_step,
    start_trace,
)

__all__ = [
    # Observability
    "get_tracer",
    "start_trace",
    "log_step",
    "end_trace",
    "get_current_trace_id",
    "get_langfuse_client",
    "langfuse_observation",
    "flush_langfuse",
    # Error Recovery
    "retry_with_backoff",
    "with_error_recovery",
    "suggest_recovery_action",
    "RetryConfig",
    "RecoverableError",
    "TransientError",
    "FatalError",
    # Monitoring
    "ForecastMonitor",
    "compute_cost",
]
