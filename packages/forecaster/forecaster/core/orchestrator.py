"""Orchestrator for coordinating agents and managing flow."""

import uuid
from pathlib import Path
from typing import Any

from forecaster.core.session import (
    ForecastSession,
    ModelConfig,
)


class Orchestrator:
    """
    Main coordinator for the forecasting platform.

    Manages session state and coordinates between:
    - User input (via UI)
    - Data Agent (file analysis)
    - Chat Agent (conversation)
    - Model Agent (training/forecasting)
    """

    def __init__(self):
        self.sessions: dict[str, ForecastSession] = {}

    def create_session(self) -> ForecastSession:
        """Create a new forecasting session."""
        session_id = str(uuid.uuid4())[:8]
        session = ForecastSession(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ForecastSession | None:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def handle_file_upload(
        self, session: ForecastSession, filepath: Path, filename: str
    ) -> dict[str, Any]:
        """
        Handle file upload - analyze and store.

        Returns dict with:
        - success: bool
        - message: str
        - data_info: DataInfo (if success)
        - questions: list[str] (if clarification needed)
        """
        from forecaster.data.analyzer import analyze_file

        try:
            # Analyze the file
            data_info = analyze_file(filepath, filename)

            # Update session
            session.uploaded_file = filepath
            session.data_info = data_info
            session.clear_forecast()  # Clear old forecast

            # Build response
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
                "message": f"Error analyzing file: {str(e)}",
                "data_info": None,
                "questions": [],
            }

    def handle_user_message(
        self, session: ForecastSession, user_message: str, **_kwargs
    ) -> dict[str, Any]:
        """
        Handle user message - interpret intent and respond.

        Args:
            session: Current session
            user_message: User's message

        Returns dict with:
        - response: str (message to display)
        - thinking: Optional[str] (agent's reasoning)
        - actions: list (actions to execute)
        - tool_calls: list (tools that were called)
        """
        # Add user message to history
        session.add_message("user", user_message)

        from forecaster.agents.chat_v2 import get_chat_agent_v2

        chat_agent = get_chat_agent_v2()
        result = chat_agent.process(session, user_message)

        # Add assistant response to history
        session.add_message("assistant", result["response"])

        return result

    def set_datetime_column(self, session: ForecastSession, column: str) -> None:
        """Set the datetime column for the dataset."""
        if session.data_info:
            session.data_info.datetime_column = column
            session.clear_forecast()

    def set_target_column(self, session: ForecastSession, column: str) -> None:
        """Set the target column for forecasting."""
        if session.data_info:
            session.data_info.target_column = column
            session.clear_forecast()

    def set_horizon(self, session: ForecastSession, horizon: int) -> None:
        """Set the forecast horizon."""
        if session.forecast_config is None:
            session.forecast_config = ModelConfig()
        session.forecast_config.horizon = horizon

    def set_gap(self, session: ForecastSession, gap: int) -> None:
        """Set the forecast gap (periods between now and first forecast point)."""
        if session.forecast_config is None:
            session.forecast_config = ModelConfig()
        session.forecast_config.gap = gap

    def execute_data_operation(
        self,
        session: ForecastSession,
        operation: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a data operation on in-memory DataFrame.

        Args:
            session: Current session
            operation: Name of operation (e.g., "combine_datetime", "normalize")
            parameters: Operation-specific parameters

        Returns:
            {
                "success": bool,
                "message": str,
                "dataframe": DataFrame (if success)
            }
        """
        from forecaster.utils.streamlit_optional import get_session_state

        ss = get_session_state()
        current_df = session.current_df or ss.get("current_dataframe")
        if current_df is None:
            return {
                "success": False,
                "message": "No data loaded in memory",
            }

        from forecaster.agents.data_operations_v2 import DataOperations

        ops = DataOperations()
        result = ops.execute_operation(current_df, operation, parameters)

        # If successful, update session
        if result["success"]:
            session.current_df = result["dataframe"]

            # Re-analyze to update metadata
            from forecaster.data.analyzer import analyze_dataframe

            data_info = analyze_dataframe(
                result["dataframe"],
                session.data_info.filename if session.data_info else "data.csv",
                session.uploaded_file,
            )
            session.data_info = data_info
            session.clear_forecast()  # Clear old forecast

            # Debug: Log the update
            print(
                f"[Orchestrator] DataFrame updated: {len(result['dataframe'])} rows, {len(result['dataframe'].columns)} columns"
            )
            print(f"[Orchestrator] New columns: {list(result['dataframe'].columns)}")

        return result

    def combine_datetime_columns(
        self,
        session: ForecastSession,
        date_column: str,
        time_column: str,
        output_column: str = "datetime",
    ) -> dict[str, Any]:
        """
        Combine date and time columns (legacy method, use execute_data_operation).
        """
        return self.execute_data_operation(
            session,
            "combine_datetime",
            {
                "date_column": date_column,
                "time_column": time_column,
                "output_column": output_column,
            },
        )

    def _run_forecast(self, session: ForecastSession, config: dict[str, Any]) -> dict[str, Any]:
        """
        Trigger the LangGraph forecast pipeline.

        Delegates to the compiled StateGraph via Streamlit session state so
        all execution goes through the same graph path as the sidebar button.
        The graph is invoked by app._run_workflow_with_progress() on the next
        Streamlit rerun (triggered by setting workflow_pending=True).

        Returns a lightweight acknowledgement dict; the real result is written
        to session.forecast_result by _run_workflow_with_progress().
        """
        # Validate pre-conditions before queuing
        if session.data_info is None:
            return {
                "success": False,
                "message": "No data loaded. Please upload a file first.",
                "forecast": None,
            }
        if session.data_info.datetime_column is None:
            return {
                "success": False,
                "message": "Date column not selected. Which column contains dates?",
                "forecast": None,
            }
        if session.data_info.target_column is None:
            return {
                "success": False,
                "message": "Target column not selected. Which column do you want to forecast?",
                "forecast": None,
            }

        # Apply any config overrides before queuing
        if config.get("horizon"):
            self.set_horizon(session, int(config["horizon"]))
        if config.get("gap") is not None:
            self.set_gap(session, int(config["gap"]))
        if config.get("model_type") and session.forecast_config:
            session.forecast_config.model_type = config["model_type"]

        return {
            "success": True,
            "message": "Forecast configuration updated. Run the forecast workflow from the API or UI.",
            "forecast": None,
        }

    def get_session_summary(self, session: ForecastSession) -> dict[str, Any]:
        """Get summary of current session state."""
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
