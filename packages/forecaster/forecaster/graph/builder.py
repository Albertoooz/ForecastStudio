"""
Build and compile the forecast StateGraph.

Call build_forecast_graph() once at app startup and store the result.
The compiled graph is thread-safe and reusable across sessions — each
session is isolated by its thread_id in the checkpointer.

Phase 1: MemorySaver (in-process, lost on restart).
Phase 3: swap for PostgresSaver(conn) for crash-recovery + multi-device.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from forecaster.graph.checkpointer import PickleSerde
from forecaster.graph.nodes import (
    analyze_data,
    engineer_features,
    generate_forecast,
    prepare_data,
    select_model,
    train_evaluate,
)
from forecaster.graph.state import ForecastGraphState


def build_forecast_graph():
    """
    Compile the forecast pipeline as a LangGraph StateGraph.

    Graph topology (Phase 1 — linear with HITL interrupt in analyze_data):

        START
          │
          ▼
      analyze_data  ←── may interrupt() once for user questions
          │
          ▼
      prepare_data
          │
          ▼
      engineer_features
          │
          ▼
      select_model
          │
          ▼
      train_evaluate
          │
          ▼
      generate_forecast
          │
          ▼
         END

    All edges are direct in Phase 1. Conditional routing hooks exist in
    forecaster/graph/edges.py and will be wired in Phase 2.

    Returns:
        CompiledStateGraph — thread-safe, ready to invoke/stream.
    """
    graph = StateGraph(ForecastGraphState)

    # Register nodes
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("prepare_data", prepare_data)
    graph.add_node("engineer_features", engineer_features)
    graph.add_node("select_model", select_model)
    graph.add_node("train_evaluate", train_evaluate)
    graph.add_node("generate_forecast", generate_forecast)

    # Wire edges (linear for Phase 1)
    graph.add_edge(START, "analyze_data")
    graph.add_edge("analyze_data", "prepare_data")
    graph.add_edge("prepare_data", "engineer_features")
    graph.add_edge("engineer_features", "select_model")
    graph.add_edge("select_model", "train_evaluate")
    graph.add_edge("train_evaluate", "generate_forecast")
    graph.add_edge("generate_forecast", END)

    # Checkpointer — required for interrupt() to work.
    # PickleSerde handles DataFrames and fitted model objects that msgpack rejects.
    checkpointer = MemorySaver(serde=PickleSerde())

    return graph.compile(checkpointer=checkpointer)
