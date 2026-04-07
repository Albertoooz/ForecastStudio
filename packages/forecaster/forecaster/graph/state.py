"""
ForecastGraphState — the single source of truth flowing through the pipeline graph.

Design rules:
  - Config fields (datetime_column, target_column, …) are set once at START and never mutated.
  - Accumulator fields (step_results, decision_log, errors, progress_messages) use
    Annotated reducers so each node's output is merged/appended, not overwritten.
  - Heavy objects (DataFrames, trained models) are kept as Any; MemorySaver stores
    them in-process so JSON-serialisability is not required for Phase 1.
  - For Phase 3 (PostgresSaver), these will be replaced with blob references.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


def _merge_dicts(existing: dict, update: dict) -> dict:
    """Reducer for step_results — new entries override existing by key."""
    return {**existing, **update}


class ForecastGraphState(TypedDict):
    """State object passed between every node in the forecast graph."""

    # ── Input config (frozen at graph invocation, never modified) ────────────
    datetime_column: str
    target_column: str
    group_columns: list[str]
    horizon: int
    gap: int
    frequency: str | None  # may be updated by analyze_data node once

    # ── Data (DataFrames — in-memory only, not JSON-serialisable) ────────────
    dataframe: Any  # original pl.DataFrame from the user
    prepared_dataframe: Any | None
    future_exog_df: Any | None  # uploaded future exogenous values

    # ── Per-node results (accumulated across all nodes) ───────────────────────
    #    Each node contributes one entry keyed by its own name.
    step_results: Annotated[dict[str, dict], _merge_dicts]

    # Append-only audit trail — every node appends its AgentDecisions as dicts
    decision_log: Annotated[list[dict], operator.add]

    # ── Intermediate analysis results (set once, read by later nodes) ─────────
    data_profile: dict | None  # compact DataProfile from DataAnalyzerAgent
    analysis_data: dict | None  # richer stats from analyze_for_wizard
    feature_config: dict | None  # lags, rolling_windows, date_features, use_ewm
    recommended_model: str | None

    # ── Model training results ────────────────────────────────────────────────
    model_results: dict | None  # {model_name: {holdout_rmse, holdout_mape, …}}
    best_model_name: str | None
    trained_model: Any | None  # fitted model object

    # ── Final forecast ────────────────────────────────────────────────────────
    forecast_result: Any | None  # ForecastResult (Pydantic model)

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    #    Populated after the user answers an interrupt() question.
    user_responses: dict | None  # {question_key: "yes"|"no"|…}

    # ── Workflow health ───────────────────────────────────────────────────────
    workflow_status: str  # "running"|"done"|"failed"
    errors: Annotated[list[str], operator.add]  # non-fatal errors, accumulated
    progress_messages: Annotated[list[dict], operator.add]  # [{text, step, status, ts}]
