"""
ForecastOrchestrator — deterministic pipeline coordinator.

Manages the agent pipeline:
  1. MemoryManager → reserve resources
  2. DataAnalyzer → profile data, suggest cleaning
  3. ExternalData → suggest joins (requires confirmation)
  4. FeatureEngineer → generate features
  5. ModelSelector → pick model
  6. Train → run forecast, log metrics

Yields (AgentDecision, requires_confirmation) at each step.
NEVER auto-proceeds on requires_confirmation=True.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from forecaster.agents.external_data_agent import ExternalDataAgent
from forecaster.agents.feature_engineer import FeatureEngineerAgent
from forecaster.core.context import AgentDecision, ContextWindow

if TYPE_CHECKING:
    from forecaster.agents.base import BaseAgent


# ---------------------------------------------------------------------------
#  Pipeline Step result
# ---------------------------------------------------------------------------


class PipelineStep:
    """Result of a single pipeline step."""

    def __init__(
        self,
        agent_name: str,
        decision: AgentDecision,
        context: ContextWindow,
        requires_confirmation: bool = False,
    ):
        self.agent_name = agent_name
        self.decision = decision
        self.context = context
        self.requires_confirmation = requires_confirmation

    @property
    def success(self) -> bool:
        return self.decision.decision_type != "error"

    def __repr__(self) -> str:
        status = "⚠️ CONFIRM" if self.requires_confirmation else ("✓" if self.success else "✗")
        return f"[{self.agent_name}] {self.decision.action} {status}"


# ---------------------------------------------------------------------------
#  Forecast Orchestrator
# ---------------------------------------------------------------------------


class ForecastOrchestrator:
    """
    Deterministic pipeline orchestrator.

    Usage:
        ctx = ContextWindow(...)
        ctx.register_data("sales", df)
        orch = ForecastOrchestrator()

        for step in orch.run(ctx):
            if step.requires_confirmation:
                # Show to user, wait for approval
                ...
            else:
                # Auto-proceed
                ...

        # Final context has forecast_result, decision_log, cost estimate
    """

    def __init__(
        self,
        agents: list[BaseAgent] | None = None,
        skip_external_data: bool = False,
    ):
        """
        Initialize orchestrator.

        Args:
            agents: Custom agent list (order matters). If None, uses default pipeline.
            skip_external_data: Skip external data agent entirely.
        """
        if agents is not None:
            self._agents = agents
        else:
            self._agents = self._default_agents(skip_external_data)

    @staticmethod
    def _default_agents(skip_external: bool = False) -> list[BaseAgent]:
        from forecaster.agents.data_analyzer import DataAnalyzerAgent
        from forecaster.agents.external_data_agent import ExternalDataAgent
        from forecaster.agents.feature_engineer import FeatureEngineerAgent
        from forecaster.agents.memory_manager import MemoryManagerAgent
        from forecaster.agents.model_selector import ModelSelectorAgent

        agents: list[BaseAgent] = [
            MemoryManagerAgent(),
            DataAnalyzerAgent(),
        ]
        if not skip_external:
            agents.append(ExternalDataAgent())
        agents.extend(
            [
                FeatureEngineerAgent(),
                ModelSelectorAgent(),
            ]
        )
        return agents

    # -- Main pipeline -----------------------------------------------------

    def run(self, context: ContextWindow) -> Generator[PipelineStep, str | None, ContextWindow]:
        """
        Run the full forecast pipeline as a generator.

        Yields PipelineStep at each stage. The caller can send() user decisions
        back for steps that require_confirmation.

        Returns the final ContextWindow with forecast results.
        """
        for agent in self._agents:
            context, decision = agent.run(context)

            step = PipelineStep(
                agent_name=agent.name,
                decision=decision,
                context=context,
                requires_confirmation=decision.requires_confirmation,
            )

            if decision.decision_type == "error":
                # Non-recoverable errors yield but don't stop the pipeline
                # unless it's a budget exceeded
                if decision.action in ("resource_budget_exceeded", "budget_exceeded"):
                    context.pending_decision = decision
                    context.requires_confirmation = True
                    yield step
                    return context
                yield step
                continue

            if decision.requires_confirmation:
                context.pending_decision = decision
                context.requires_confirmation = True
                user_response = yield step
                context.requires_confirmation = False
                context.pending_decision = None

                # Handle user response for external data joins
                if user_response and user_response.lower() in ("approve", "yes", "tak"):
                    if isinstance(agent, ExternalDataAgent) and decision.parameters.get(
                        "table_name"
                    ):
                        context = agent.apply_join(context, decision.parameters["table_name"])
                elif user_response and user_response.lower() in ("cancel", "no", "nie"):
                    context.log_decision(
                        AgentDecision(
                            agent_name=agent.name,
                            decision_type=decision.decision_type,
                            action=f"user_rejected: {decision.action}",
                            reasoning="User chose not to proceed with this step.",
                        )
                    )
                # "modify" → caller should set context fields and continue
            else:
                yield step

        # After all agents: run the actual training
        context = self._train_and_forecast(context)

        # Final step — training result
        train_decision = (
            context.decision_log[-1]
            if context.decision_log
            else AgentDecision(
                agent_name="orchestrator",
                decision_type="error",
                action="no_training",
                reasoning="Pipeline completed without training.",
            )
        )

        yield PipelineStep(
            agent_name="training",
            decision=train_decision,
            context=context,
        )

        return context

    def run_auto(self, context: ContextWindow) -> ContextWindow:
        """
        Run pipeline without user interaction (auto-approve everything).

        Useful for CLI / batch mode.
        """
        gen = self.run(context)
        try:
            step = next(gen)
            while True:
                if step.requires_confirmation:
                    step = gen.send("approve")
                else:
                    step = next(gen)
        except StopIteration as e:
            return e.value if e.value else step.context

    # -- Training ----------------------------------------------------------

    def _train_and_forecast(self, context: ContextWindow) -> ContextWindow:
        """Execute model training and return results."""
        context.advance_phase("training")
        start = time.time()

        df = context.get_primary_data()
        if df is None:
            context.log_decision(
                AgentDecision(
                    agent_name="orchestrator",
                    decision_type="error",
                    action="no_data_for_training",
                    reasoning="No data available after pipeline execution.",
                )
            )
            return context

        model_spec = context.model_spec
        if model_spec is None:
            context.log_decision(
                AgentDecision(
                    agent_name="orchestrator",
                    decision_type="error",
                    action="no_model_spec",
                    reasoning="ModelSelector did not produce a model specification.",
                )
            )
            return context

        # Apply features
        if context.feature_specs and context.datetime_column and context.target_column:
            holidays_df = context.get_data("holidays_pl")
            df = FeatureEngineerAgent.apply_features(
                df,
                context.feature_specs,
                context.datetime_column,
                context.target_column,
                holidays_df=holidays_df,
            )
            # Update registry with featured data
            first_key = next(iter(context.data_registry))
            context.data_registry[first_key] = df

        # Run the actual model
        try:
            result = self._execute_model(
                df=df,
                model_type=model_spec.model_type,
                datetime_column=context.datetime_column,
                target_column=context.target_column,
                group_columns=context.group_columns,
                horizon=context.horizon,
                gap=context.gap,
                hyperparameters=model_spec.hyperparameters,
            )

            elapsed = time.time() - start
            context.forecast_result = result

            context.log_decision(
                AgentDecision(
                    agent_name="model_trainer",
                    decision_type="model_training",
                    action=f"trained: {model_spec.model_type}",
                    parameters={
                        "metrics": result.get("metrics", {}),
                        "horizon": context.horizon,
                        "model_type": model_spec.model_type,
                    },
                    confidence=0.9,
                    reasoning=(
                        f"Training complete in {elapsed:.1f}s. "
                        + (
                            f"MAE={result['metrics'].get('mae', '?')}"
                            if result.get("metrics")
                            else ""
                        )
                    ),
                    estimated_compute_seconds=elapsed,
                    duration_ms=elapsed * 1000,
                )
            )

        except Exception as e:
            context.log_decision(
                AgentDecision(
                    agent_name="model_trainer",
                    decision_type="error",
                    action="training_failed",
                    reasoning=str(e),
                )
            )

        return context

    @staticmethod
    def _execute_model(
        df: pd.DataFrame,
        model_type: str,
        datetime_column: str | None,
        target_column: str | None,
        group_columns: list[str],
        horizon: int,
        gap: int,
        hyperparameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute the actual model training.

        Maps ModelSelector names to ModelAgent names and delegates.
        """
        from forecaster.agents.model_agent import ModelAgent

        # Map pipeline model names → ModelAgent model names
        MODEL_NAME_MAP = {  # noqa: N806
            "simple_ewm": "naive",
            "lightgbm_default": "lightgbm",
            "mlforecast_global": "lightgbm",
            "prophet": "prophet",
            "naive": "naive",
            "linear": "linear",
        }

        mapped_type = MODEL_NAME_MAP.get(model_type, "auto")

        agent = ModelAgent()
        result = agent.forecast(
            filepath=None,
            datetime_column=datetime_column,
            target_column=target_column,
            horizon=horizon,
            gap=gap,
            model_type=mapped_type,
            group_by_columns=group_columns,
            dataframe=df,
        )

        return result


# ---------------------------------------------------------------------------
#  Legacy Orchestrator (backward compat with app.py)
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Legacy orchestrator — kept for backward compatibility with Streamlit app.

    Delegates to ForecastOrchestrator for the new pipeline,
    but maintains the old API surface.
    """

    def __init__(self):
        from forecaster.core.session import ForecastSession

        self.sessions: dict[str, ForecastSession] = {}

    def create_session(self):
        from forecaster.core.session import ForecastSession

        session_id = str(uuid.uuid4())[:8]
        session = ForecastSession(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def handle_file_upload(self, session, filepath: Path, filename: str) -> dict[str, Any]:
        from forecaster.data.analyzer import analyze_file

        try:
            data_info = analyze_file(filepath, filename)
            session.uploaded_file = filepath
            session.data_info = data_info
            session.clear_forecast()
            response = {
                "success": True,
                "message": f"Loaded file '{filename}' ({data_info.n_rows} rows)",
                "data_info": data_info,
                "questions": data_info.questions,
            }
            if data_info.issues:
                response["warnings"] = data_info.issues
            return response
        except Exception as e:
            return {
                "success": False,
                "message": f"Error analyzing file: {e}",
                "data_info": None,
                "questions": [],
            }

    def handle_user_message(self, session, user_message: str, **kwargs) -> dict[str, Any]:
        session.add_message("user", user_message)
        from forecaster.agents.chat_v2 import get_chat_agent_v2

        chat_agent = get_chat_agent_v2()
        result = chat_agent.process(session, user_message)
        session.add_message("assistant", result["response"])
        return result

    def set_datetime_column(self, session, column: str):
        if session.data_info:
            session.data_info.datetime_column = column
            session.clear_forecast()

    def set_target_column(self, session, column: str):
        if session.data_info:
            session.data_info.target_column = column
            session.clear_forecast()

    def set_horizon(self, session, horizon: int):
        from forecaster.core.session import ModelConfig

        if session.forecast_config is None:
            session.forecast_config = ModelConfig()
        session.forecast_config.horizon = horizon

    def set_gap(self, session, gap: int):
        from forecaster.core.session import ModelConfig

        if session.forecast_config is None:
            session.forecast_config = ModelConfig()
        session.forecast_config.gap = gap

    def execute_data_operation(
        self, session, operation: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        from forecaster.utils.streamlit_optional import get_session_state

        ss = get_session_state()
        current_df = getattr(session, "current_df", None) or ss.get("current_dataframe")
        if current_df is None:
            return {"success": False, "message": "No data loaded in memory"}
        from forecaster.agents.data_operations_v2 import DataOperations

        ops = DataOperations()
        result = ops.execute_operation(current_df, operation, parameters)
        if result["success"]:
            session.current_df = result["dataframe"]
            from forecaster.data.analyzer import analyze_dataframe

            data_info = analyze_dataframe(
                result["dataframe"],
                session.data_info.filename if session.data_info else "data.csv",
                session.uploaded_file,
            )
            session.data_info = data_info
            session.clear_forecast()
        return result

    def combine_datetime_columns(self, session, date_column, time_column, output_column="datetime"):
        return self.execute_data_operation(
            session,
            "combine_datetime",
            {
                "date_column": date_column,
                "time_column": time_column,
                "output_column": output_column,
            },
        )

    def _run_forecast(self, session, config: dict[str, Any]) -> dict[str, Any]:
        from forecaster.agents.model_agent import ModelAgent

        if session.data_info is None:
            return {"success": False, "message": "No data loaded.", "forecast": None}
        if session.data_info.datetime_column is None:
            return {"success": False, "message": "Date column not selected.", "forecast": None}
        if session.data_info.target_column is None:
            return {"success": False, "message": "Target column not selected.", "forecast": None}

        horizon = config.get("horizon", 7)
        if session.forecast_config:
            horizon = session.forecast_config.horizon
        try:
            horizon = int(horizon)
        except (ValueError, TypeError):
            horizon = 7

        gap = 0
        if session.forecast_config:
            gap = session.forecast_config.gap

        try:
            from forecaster.utils.streamlit_optional import get_session_state

            ss = get_session_state()
            override_df = config.get("dataframe")
            current_df = (
                override_df
                if override_df is not None
                else (getattr(session, "current_df", None) or ss.get("current_dataframe"))
            )
            model_agent = ModelAgent()
            model_type = config.get("model_type", "auto")
            if (
                session.forecast_config
                and session.forecast_config.model_type
                and model_type == "auto"
            ):
                if session.forecast_config.model_type not in ("auto", "linear"):
                    model_type = session.forecast_config.model_type
            result = model_agent.forecast(
                filepath=session.data_info.filepath,
                datetime_column=session.data_info.datetime_column,
                target_column=session.data_info.target_column,
                horizon=horizon,
                gap=gap,
                model_type=model_type,
                group_by_column=session.data_info.group_by_column,
                group_by_columns=session.data_info.group_by_columns,
                dataframe=current_df,
            )
            if result["success"]:
                session.forecast_result = result["forecast"]
                session.is_trained = True
                if "model" in result:
                    session.trained_model = result["model"]
            return result
        except Exception as e:
            return {"success": False, "message": f"Error during forecast: {e}", "forecast": None}

    def get_session_summary(self, session) -> dict[str, Any]:
        summary = {
            "has_data": session.data_info is not None,
            "has_forecast": session.forecast_result is not None,
            "n_messages": len(session.messages),
        }
        if session.data_info:
            summary["data"] = {
                "filename": session.data_info.filename,
                "n_rows": session.data_info.n_rows,
                "datetime_column": session.data_info.datetime_column,
                "target_column": session.data_info.target_column,
            }
        if session.forecast_config:
            summary["model"] = {
                "type": session.forecast_config.model_type,
                "horizon": session.forecast_config.horizon,
            }
        return summary
