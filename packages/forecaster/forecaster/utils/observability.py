"""
Observability and Tracing for Agent System.

Professional logging, tracing, and monitoring inspired by LangSmith/LangFuse.
"""

import json
import logging
import os
import time
import uuid
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Context variable for trace ID
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

try:
    from langfuse import get_client as _get_langfuse_client
except Exception:  # pragma: no cover - optional dependency
    _get_langfuse_client = None


@dataclass
class TraceStep:
    """Single step in agent execution trace."""

    step_type: str  # "tool_call", "llm_call", "action", "error"
    name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None = None
    error: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self, outputs: dict | None = None, error: str | None = None):
        """Mark step as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.outputs = outputs
        self.error = error

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "type": self.step_type,
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """Complete execution trace for an agent operation."""

    trace_id: str
    operation: str  # "user_message", "forecast", "data_operation"
    user_message: str | None = None
    steps: list[TraceStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TraceStep):
        """Add a step to the trace."""
        self.steps.append(step)

    def complete(self, success: bool = True, error: str | None = None):
        """Mark trace as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error

    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "user_message": self.user_message,
            "steps": [step.to_dict() for step in self.steps],
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat(),
            "metadata": self.metadata,
        }


class AgentTracer:
    """
    Professional agent tracer for observability.

    Features:
    - Hierarchical tracing (traces contain steps)
    - Context propagation via contextvars
    - JSON export for external observability platforms
    - Metrics aggregation
    """

    def __init__(self, max_traces: int = 1000):
        self.traces: dict[str, Trace] = {}
        self.max_traces = max_traces
        self.metrics = {
            "total_traces": 0,
            "successful_traces": 0,
            "failed_traces": 0,
            "total_steps": 0,
            "tool_calls": 0,
            "llm_calls": 0,
        }

    def start_trace(
        self, operation: str, user_message: str | None = None, metadata: dict | None = None
    ) -> str:
        """
        Start a new trace.

        Args:
            operation: Type of operation (e.g., "user_message", "forecast")
            user_message: Optional user message that triggered this
            metadata: Additional metadata

        Returns:
            trace_id: Unique trace identifier
        """
        trace_id = str(uuid.uuid4())
        trace_id_var.set(trace_id)

        trace = Trace(
            trace_id=trace_id,
            operation=operation,
            user_message=user_message,
            metadata=metadata or {},
        )

        self.traces[trace_id] = trace
        self.metrics["total_traces"] += 1

        # Cleanup old traces if needed
        if len(self.traces) > self.max_traces:
            oldest = sorted(self.traces.items(), key=lambda x: x[1].start_time)[0][0]
            del self.traces[oldest]

        logger.info(f"Started trace {trace_id[:8]}... for {operation}")
        return trace_id

    def log_step(
        self,
        step_type: str,
        name: str,
        inputs: dict,
        outputs: dict | None = None,
        error: str | None = None,
        metadata: dict | None = None,
    ):
        """Log a step in the current trace."""
        trace_id = trace_id_var.get()
        if not trace_id or trace_id not in self.traces:
            logger.warning("Attempted to log step without active trace")
            return

        step = TraceStep(
            step_type=step_type,
            name=name,
            inputs=inputs,
            outputs=outputs,
            error=error,
            metadata=metadata or {},
        )
        step.complete(outputs, error)

        self.traces[trace_id].add_step(step)
        self.metrics["total_steps"] += 1

        if step_type == "tool_call":
            self.metrics["tool_calls"] += 1
        elif step_type == "llm_call":
            self.metrics["llm_calls"] += 1

        logger.debug(f"Logged step {step_type}: {name} ({step.duration_ms:.0f}ms)")

    def end_trace(self, success: bool = True, error: str | None = None):
        """End the current trace."""
        trace_id = trace_id_var.get()
        if not trace_id or trace_id not in self.traces:
            logger.warning("Attempted to end trace without active trace")
            return

        trace = self.traces[trace_id]
        trace.complete(success, error)

        if success:
            self.metrics["successful_traces"] += 1
        else:
            self.metrics["failed_traces"] += 1

        logger.info(
            f"Ended trace {trace_id[:8]}... "
            f"({trace.duration_ms:.0f}ms, {len(trace.steps)} steps, "
            f"{'✅ success' if success else '❌ failed'})"
        )

        # Clear context
        trace_id_var.set(None)

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a specific trace."""
        return self.traces.get(trace_id)

    def get_recent_traces(self, n: int = 10) -> list[Trace]:
        """Get N most recent traces."""
        return sorted(
            self.traces.values(),
            key=lambda t: t.start_time,
            reverse=True,
        )[:n]

    def export_trace(self, trace_id: str) -> str | None:
        """Export trace as JSON."""
        trace = self.get_trace(trace_id)
        if not trace:
            return None
        return json.dumps(trace.to_dict(), indent=2)

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_traces"] / self.metrics["total_traces"]
                if self.metrics["total_traces"] > 0
                else 0.0
            ),
            "avg_steps_per_trace": (
                self.metrics["total_steps"] / self.metrics["total_traces"]
                if self.metrics["total_traces"] > 0
                else 0.0
            ),
        }

    def clear_traces(self):
        """Clear all traces (useful for testing)."""
        self.traces.clear()


# Global tracer instance
_tracer: AgentTracer | None = None
_langfuse_client = None


def get_tracer() -> AgentTracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = AgentTracer()
    return _tracer


def get_langfuse_client():
    """Return a Langfuse client if credentials are configured."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if _get_langfuse_client is None:
        return None

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None

    try:
        _langfuse_client = _get_langfuse_client()
        return _langfuse_client
    except Exception as exc:  # pragma: no cover - best effort observability
        logger.debug("Langfuse client unavailable: %s", exc)
        return None


def langfuse_observation(*, as_type: str = "span", name: str, **kwargs):
    """Create a Langfuse observation context or a no-op fallback."""
    client = get_langfuse_client()
    if client is None:
        return nullcontext()
    return client.start_as_current_observation(as_type=as_type, name=name, **kwargs)


# Convenience functions
def start_trace(operation: str, user_message: str | None = None, **metadata) -> str:
    """Start a new trace."""
    return get_tracer().start_trace(operation, user_message, metadata)


def log_step(step_type: str, name: str, inputs: dict, outputs: dict | None = None, **metadata):
    """Log a step."""
    get_tracer().log_step(step_type, name, inputs, outputs, metadata=metadata)


def end_trace(success: bool = True, error: str | None = None):
    """End current trace."""
    get_tracer().end_trace(success, error)


def get_current_trace_id() -> str | None:
    """Get current trace ID."""
    return trace_id_var.get()


def flush_langfuse():
    """Flush buffered Langfuse events when the app shuts down."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - best effort observability
        logger.debug("Langfuse flush failed: %s", exc)
