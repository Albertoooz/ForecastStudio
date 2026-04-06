"""
Conditional edge functions for the forecast graph.

Each function receives the full graph state and returns the name of the next node.
This keeps all routing logic in one place, separate from node business logic.

Phase 1 has a mostly linear graph (no retries or parallel branches yet).
These functions are designed to be extended in Phase 2 (retries, fallbacks, A/B paths).
"""

from __future__ import annotations

from forecaster.graph.state import ForecastGraphState


def after_train_evaluate(state: ForecastGraphState) -> str:
    """
    After training, route to forecast OR end early if all models failed.

    In Phase 1 this always goes to generate_forecast.
    Phase 2 could add a retry_with_simpler_model branch.
    """
    trained_model = state.get("trained_model")
    if trained_model is None:
        # All models failed — still go to generate_forecast which will
        # return a clean error rather than crashing the graph.
        return "generate_forecast"
    return "generate_forecast"


def after_generate_forecast(state: ForecastGraphState) -> str:
    """
    Terminal routing — always END.

    Phase 2 hook: could loop back to train_evaluate if health score < threshold
    and a different model hasn't been tried yet.
    """
    from langgraph.graph import END

    return END
