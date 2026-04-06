"""
MemoryManagerAgent — pre-flight resource check and reservation.

Runs FIRST in the pipeline to ensure we have enough resources
before any heavy computation begins.
"""

from __future__ import annotations

import pandas as pd

from forecaster.agents.base import BaseAgent
from forecaster.core.context import AgentDecision, ContextWindow


class MemoryManagerAgent(BaseAgent):
    """
    Pre-flight check: can we run the requested forecast within budget?

    Estimates memory and compute requirements based on:
      - Data size (rows × columns)
      - Planned model type
      - Number of groups (if multi-series)

    If budget exceeded → blocks execution and suggests downsampling.
    """

    name: str = "memory_manager"

    # Rough per-model estimates (MB per 10k rows)
    _MODEL_MEMORY_ESTIMATES: dict[str, float] = {
        "simple_ewm": 5,
        "naive": 5,
        "linear": 10,
        "prophet": 80,
        "lightgbm_default": 60,
        "mlforecast_global": 100,
    }

    # Rough per-model compute estimates (seconds per 10k rows)
    _MODEL_COMPUTE_ESTIMATES: dict[str, float] = {
        "simple_ewm": 0.1,
        "naive": 0.05,
        "linear": 0.2,
        "prophet": 5.0,
        "lightgbm_default": 2.0,
        "mlforecast_global": 3.0,
    }

    def execute(self, context: ContextWindow) -> tuple[ContextWindow, AgentDecision]:
        context.advance_phase("memory_check")

        df = context.get_primary_data()
        if df is None:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="memory_reservation",
                action="no_data",
                reasoning="No data loaded — cannot estimate resources.",
            )

        # Estimate data memory
        data_memory_mb = self._estimate_data_memory(df)

        # Estimate model resources (if model already selected)
        model_type = "lightgbm_default"  # Default assumption
        if context.model_spec:
            model_type = context.model_spec.model_type

        n_rows = len(df)
        n_groups = self._count_groups(df, context.group_columns)

        model_memory = self._estimate_model_memory(model_type, n_rows, n_groups)
        model_compute = self._estimate_model_compute(model_type, n_rows, n_groups)

        total_memory = data_memory_mb + model_memory
        total_compute = model_compute

        # Check budgets
        memory_ok = context.budget.can_reserve_memory(total_memory)
        compute_ok = context.budget.can_reserve_compute(total_compute)
        all_ok = memory_ok and compute_ok

        if all_ok:
            # Reserve resources
            context.budget.reserve_memory(total_memory)
            context.budget.reserve_compute(total_compute)

            decision = AgentDecision(
                agent_name=self.name,
                decision_type="memory_reservation",
                action="reserved",
                parameters={
                    "data_memory_mb": data_memory_mb,
                    "model_memory_mb": model_memory,
                    "total_memory_mb": total_memory,
                    "estimated_compute_s": total_compute,
                    "model_type": model_type,
                    "n_rows": n_rows,
                    "n_groups": n_groups,
                },
                confidence=0.9,
                reasoning=(
                    f"Reserved {total_memory}MB / {context.budget.memory_budget_mb}MB "
                    f"and {total_compute:.1f}s / {context.budget.compute_budget_seconds}s "
                    f"for {model_type} on {n_rows} rows"
                    + (f" × {n_groups} groups" if n_groups > 1 else "")
                ),
                requires_confirmation=False,
                estimated_memory_mb=total_memory,
                estimated_compute_seconds=total_compute,
            )
        else:
            # Budget exceeded — suggest remediation
            suggestions = self._suggest_remediation(
                total_memory, total_compute, context, n_rows, n_groups, model_type
            )

            decision = AgentDecision(
                agent_name=self.name,
                decision_type="memory_reservation",
                action="budget_exceeded",
                parameters={
                    "required_memory_mb": total_memory,
                    "available_memory_mb": context.budget.remaining_memory_mb,
                    "required_compute_s": total_compute,
                    "available_compute_s": context.budget.remaining_compute_seconds,
                    "suggestions": suggestions,
                },
                confidence=1.0,
                reasoning=(
                    f"Budget exceeded: need {total_memory}MB / {context.budget.remaining_memory_mb}MB avail, "
                    f"{total_compute:.1f}s / {context.budget.remaining_compute_seconds:.1f}s avail. "
                    + "; ".join(suggestions)
                ),
                requires_confirmation=True,
            )

        return context, decision

    # -- Estimation helpers ------------------------------------------------

    @staticmethod
    def _estimate_data_memory(df: pd.DataFrame) -> int:
        """Estimate memory used by the DataFrame in MB."""
        mem_bytes = df.memory_usage(deep=True).sum()
        return max(1, int(mem_bytes / (1024 * 1024)))

    def _estimate_model_memory(self, model_type: str, n_rows: int, n_groups: int) -> int:
        """Estimate memory needed for model training."""
        per_10k = self._MODEL_MEMORY_ESTIMATES.get(model_type, 60)
        base = max(1, int(per_10k * (n_rows / 10_000)))
        # Multiply by groups for non-global models
        if model_type not in ("mlforecast_global",) and n_groups > 1:
            base *= n_groups
        return base

    def _estimate_model_compute(self, model_type: str, n_rows: int, n_groups: int) -> float:
        """Estimate compute time for model training."""
        per_10k = self._MODEL_COMPUTE_ESTIMATES.get(model_type, 2.0)
        base = per_10k * (n_rows / 10_000)
        if model_type not in ("mlforecast_global",) and n_groups > 1:
            base *= n_groups
        return max(0.5, round(base, 1))

    @staticmethod
    def _count_groups(df: pd.DataFrame, group_columns: list[str]) -> int:
        """Count number of groups in data."""
        if not group_columns:
            return 1
        valid_cols = [c for c in group_columns if c in df.columns]
        if not valid_cols:
            return 1
        return df[valid_cols].drop_duplicates().shape[0]

    @staticmethod
    def _suggest_remediation(
        total_memory: int,
        total_compute: float,
        context: ContextWindow,
        n_rows: int,
        n_groups: int,
        model_type: str,
    ) -> list[str]:
        """Suggest ways to fit within budget."""
        suggestions = []

        if total_memory > context.budget.remaining_memory_mb:
            if n_rows > 10_000:
                target_rows = int(10_000 * (context.budget.remaining_memory_mb / total_memory))
                suggestions.append(f"Downsample to ~{target_rows} rows (currently {n_rows})")
            if model_type in ("prophet", "lightgbm_default"):
                suggestions.append("Switch to simple_ewm (uses ~5MB per 10k rows)")
            if n_groups > 1:
                suggestions.append(f"Reduce groups (currently {n_groups})")

        if total_compute > context.budget.remaining_compute_seconds:
            if model_type == "prophet":
                suggestions.append("Prophet is compute-heavy — try lightgbm_default")
            suggestions.append(
                f"Increase compute budget from {context.budget.compute_budget_seconds}s"
            )

        if not suggestions:
            suggestions.append("Increase memory or compute budget in ContextWindow")

        return suggestions
