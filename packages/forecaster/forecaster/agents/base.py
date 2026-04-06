"""
Base agent interface — pure function with Pydantic input/output.

Every agent:
  1. Receives a ContextWindow (explicit state)
  2. Returns a modified ContextWindow + AgentDecision
  3. Logs its decision to the audit trail
  4. Validates resource budget before executing
"""

import time
from abc import ABC
from typing import Any

from pydantic import BaseModel

from forecaster.core.context import AgentDecision, ContextWindow

# ---------------------------------------------------------------------------
#  Legacy response (kept for backward-compat with old agents)
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Structured response from an agent (legacy)."""

    success: bool
    message: str
    data: dict[str, Any] = {}
    errors: list[str] = []


# ---------------------------------------------------------------------------
#  New BaseAgent (abstract, pure-function style)
# ---------------------------------------------------------------------------


class BaseAgent(ABC):  # noqa: B024
    """
    Abstract base for all pipeline agents.

    Contract:
      execute(ctx) -> (ctx', decision)

    Every agent MUST:
      • Check resource budget via validate_resources()
      • Log exactly one AgentDecision
      • Never mutate global state — only the returned ContextWindow
    """

    name: str = "base_agent"

    def __init__(self, name: str | None = None):
        if name is not None:
            self.name = name

    # -- Public API --------------------------------------------------------

    def run(self, context: ContextWindow) -> tuple[ContextWindow, AgentDecision]:
        """
        Run the agent with resource validation and timing.

        Wraps execute() with pre-flight checks and duration tracking.
        """
        # Pre-flight resource check
        if not self.validate_resources(context):
            decision = AgentDecision(
                agent_name=self.name,
                decision_type="error",
                action="resource_budget_exceeded",
                parameters={
                    "remaining_memory_mb": context.budget.remaining_memory_mb,
                    "remaining_compute_seconds": context.budget.remaining_compute_seconds,
                },
                confidence=1.0,
                reasoning="Insufficient resources to run this agent.",
            )
            context.log_decision(decision)
            return context, decision

        start = time.time()
        context, decision = self.execute(context)
        elapsed_ms = (time.time() - start) * 1000
        decision.duration_ms = elapsed_ms

        # Auto-log decision to context
        if decision not in context.decision_log:
            context.log_decision(decision)

        return context, decision

    def execute(self, context: ContextWindow) -> tuple[ContextWindow, AgentDecision]:
        """
        Core agent logic — override in subclasses.

        Args:
            context: Current pipeline state.

        Returns:
            Tuple of (updated context, decision made by this agent).

        Note: Not marked @abstractmethod to keep backward compat with
        legacy agents that only implement process().
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute(ctx).")

    def validate_resources(self, context: ContextWindow) -> bool:
        """
        Check whether the context has enough resources for this agent.

        Override in subclasses for custom resource requirements.
        Default: require at least 1 MB memory and 0.5s compute.
        """
        return (
            context.budget.remaining_memory_mb >= 1
            and context.budget.remaining_compute_seconds >= 0.5
        )

    # -- Legacy compat -----------------------------------------------------

    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Legacy process() method for backward compatibility.

        New agents should implement execute() instead.
        """
        return AgentResponse(
            success=False,
            message=f"{self.name}: process() not implemented — use execute(ctx).",
            data={},
        )
