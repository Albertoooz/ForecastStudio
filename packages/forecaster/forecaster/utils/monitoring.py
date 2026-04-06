"""
ForecastMonitor — structured logging and cost tracking for agent decisions.

Phase 1: JSON file sink (always available)
Phase 2 (future): DuckDB table sink for Metabase/Grafana queries
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from forecaster.core.context import AgentDecision, ContextWindow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Cost calculator
# ---------------------------------------------------------------------------

# AWS t3.medium on-demand: ~$0.0416/hour = $0.00001156/second
_COMPUTE_COST_PER_SECOND = 0.00001156


def compute_cost(context: ContextWindow) -> dict[str, float]:
    """
    Calculate cost breakdown for a forecast request.

    Returns dict with compute_cost_usd, llm_cost_usd, total_cost_usd.
    """
    compute_usd = context.budget.consumed_compute_seconds * _COMPUTE_COST_PER_SECOND
    return {
        "compute_cost_usd": round(compute_usd, 6),
        "llm_cost_usd": 0.0,  # No LLM in core pipeline
        "external_data_cost_usd": 0.0,  # Local files, no API
        "total_cost_usd": round(compute_usd, 6),
    }


# ---------------------------------------------------------------------------
#  Decision record (flat dict for DuckDB / JSON)
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def decision_to_record(
    context: ContextWindow,
    decision: AgentDecision,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    """Convert a decision + context into a flat record for logging."""
    return {
        "trace_id": context.trace_id,
        "user_id": context.user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "agent_name": decision.agent_name,
        "decision_type": decision.decision_type,
        "action": decision.action,
        "parameters": json.dumps(decision.parameters, cls=_NumpyEncoder)
        if decision.parameters
        else "{}",
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "requires_confirmation": decision.requires_confirmation,
        "estimated_compute_seconds": decision.estimated_compute_seconds,
        "estimated_memory_mb": decision.estimated_memory_mb,
        "duration_ms": duration_ms or decision.duration_ms or 0.0,
        "memory_consumed_mb": context.budget.consumed_memory_mb,
        "memory_budget_mb": context.budget.memory_budget_mb,
        "compute_consumed_s": context.budget.consumed_compute_seconds,
        "compute_budget_s": context.budget.compute_budget_seconds,
        "cost_usd": context.budget.consumed_compute_seconds * _COMPUTE_COST_PER_SECOND,
        "phase": context.current_phase,
    }


# ---------------------------------------------------------------------------
#  ForecastMonitor
# ---------------------------------------------------------------------------


class ForecastMonitor:
    """
    Central monitor for all agent decisions.

    Sinks:
      1. JSON Lines file (always)
      2. DuckDB table `forecast_decisions` (when duckdb available)
      3. In-memory buffer (for UI queries)
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        enable_duckdb: bool = False,
        duckdb_path: str | None = None,
        max_buffer: int = 10_000,
    ):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / "forecast_decisions.jsonl"
        self._buffer: list[dict[str, Any]] = []
        self._max_buffer = max_buffer

        # DuckDB sink (optional)
        self._duckdb_conn = None
        if enable_duckdb:
            self._init_duckdb(duckdb_path or str(self.log_dir / "forecaster.duckdb"))

    def _init_duckdb(self, db_path: str) -> None:
        """Initialize DuckDB connection and create table if needed."""
        try:
            import duckdb  # type: ignore

            self._duckdb_conn = duckdb.connect(db_path)
            self._duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_decisions (
                    trace_id VARCHAR,
                    user_id VARCHAR,
                    timestamp VARCHAR,
                    agent_name VARCHAR,
                    decision_type VARCHAR,
                    action VARCHAR,
                    parameters VARCHAR,
                    confidence DOUBLE,
                    reasoning VARCHAR,
                    requires_confirmation BOOLEAN,
                    estimated_compute_seconds DOUBLE,
                    estimated_memory_mb INTEGER,
                    duration_ms DOUBLE,
                    memory_consumed_mb INTEGER,
                    memory_budget_mb INTEGER,
                    compute_consumed_s DOUBLE,
                    compute_budget_s DOUBLE,
                    cost_usd DOUBLE,
                    phase VARCHAR
                )
            """)
            logger.info(f"DuckDB sink initialized: {db_path}")
        except ImportError:
            logger.warning("duckdb not installed — DuckDB sink disabled")
        except Exception as e:
            logger.warning(f"DuckDB init failed: {e}")

    # -- Event handlers ----------------------------------------------------

    def on_decision(self, context: ContextWindow, decision: AgentDecision) -> None:
        """
        Log a single agent decision.

        Called by the orchestrator after each agent runs.
        """
        record = decision_to_record(context, decision)

        # 1. In-memory buffer
        self._buffer.append(record)
        if len(self._buffer) > self._max_buffer:
            self._buffer = self._buffer[-self._max_buffer :]

        # 2. JSON Lines file
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(record, cls=_NumpyEncoder) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write log: {e}")

        # 3. DuckDB (if available)
        if self._duckdb_conn is not None:
            try:
                self._duckdb_conn.execute(
                    "INSERT INTO forecast_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        record["trace_id"],
                        record["user_id"],
                        record["timestamp"],
                        record["agent_name"],
                        record["decision_type"],
                        record["action"],
                        record["parameters"],
                        record["confidence"],
                        record["reasoning"],
                        record["requires_confirmation"],
                        record["estimated_compute_seconds"],
                        record["estimated_memory_mb"],
                        record["duration_ms"],
                        record["memory_consumed_mb"],
                        record["memory_budget_mb"],
                        record["compute_consumed_s"],
                        record["compute_budget_s"],
                        record["cost_usd"],
                        record["phase"],
                    ],
                )
            except Exception as e:
                logger.warning(f"DuckDB insert failed: {e}")

    def on_pipeline_complete(self, context: ContextWindow) -> dict[str, Any]:
        """
        Called when the full pipeline finishes.

        Returns summary with cost breakdown and audit trail.
        """
        cost = compute_cost(context)
        summary = {
            "trace_id": context.trace_id,
            "n_decisions": len(context.decision_log),
            "phases_completed": context.completed_phases + [context.current_phase],
            "cost": cost,
            "audit_trail": context.get_audit_summary(),
        }

        # Log the completion as a meta-record
        try:
            meta_record = {
                "trace_id": context.trace_id,
                "event": "pipeline_complete",
                "timestamp": datetime.utcnow().isoformat(),
                "summary": summary,
            }
            with open(self._log_file, "a") as f:
                f.write(json.dumps(meta_record, cls=_NumpyEncoder, default=str) + "\n")
        except Exception:
            pass

        return summary

    # -- Query interface ---------------------------------------------------

    def get_recent_decisions(self, n: int = 50) -> list[dict[str, Any]]:
        """Get N most recent decisions from buffer."""
        return self._buffer[-n:]

    def get_trace_decisions(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all decisions for a specific trace."""
        return [r for r in self._buffer if r.get("trace_id") == trace_id]

    def get_cost_summary(self, trace_id: str | None = None) -> dict[str, float]:
        """Get cost summary, optionally filtered by trace."""
        records = self.get_trace_decisions(trace_id) if trace_id else self._buffer
        total_cost = sum(r.get("cost_usd", 0) for r in records)
        total_compute = sum(r.get("duration_ms", 0) for r in records) / 1000.0
        return {
            "total_cost_usd": round(total_cost, 6),
            "total_compute_seconds": round(total_compute, 2),
            "n_decisions": len(records),
        }

    def query_duckdb(self, sql: str) -> Any:
        """Run arbitrary SQL on the DuckDB sink (if available)."""
        if self._duckdb_conn is None:
            return None
        try:
            return self._duckdb_conn.execute(sql).fetchall()
        except Exception as e:
            logger.warning(f"DuckDB query failed: {e}")
            return None

    def close(self) -> None:
        """Close DuckDB connection."""
        if self._duckdb_conn is not None:
            try:
                self._duckdb_conn.close()
            except Exception:
                pass
