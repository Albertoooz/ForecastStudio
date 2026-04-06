"""Conversational interface for forecasting requests."""

from typing import Any

import pandas as pd

from forecaster.agents.planner import ForecastingPlanner
from forecaster.data import load_time_series, validate_time_series
from forecaster.models.automl import select_best_model
from forecaster.models.base import ForecastResult


class ForecastingConversation:
    """
    Main conversational interface for forecasting.

    Handles user requests, coordinates agents and models,
    and returns structured results.
    """

    def __init__(self, planner: ForecastingPlanner | None = None):
        self.planner = planner or ForecastingPlanner()
        self.current_data: pd.DataFrame | None = None
        self.current_model = None

    def load_data(self, path: str, **kwargs) -> dict[str, Any]:
        """
        Load time series data.

        Args:
            path: Path to data file
            **kwargs: Additional arguments for load_time_series

        Returns:
            Dictionary with status and validation results
        """
        try:
            df = load_time_series(path, **kwargs)
            validation = validate_time_series(df)

            self.current_data = df

            return {
                "success": True,
                "message": "Data loaded successfully",
                "validation": validation,
                "n_points": len(df),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to load data: {str(e)}",
                "validation": None,
                "n_points": 0,
            }

    def request_forecast(
        self,
        user_request: str,
        horizon: int | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Process forecasting request.

        Args:
            user_request: Natural language request from user
            horizon: Optional explicit forecast horizon (overrides planner)
            model_name: Optional explicit model name (overrides planner)

        Returns:
            Dictionary with forecast results and metadata
        """
        if self.current_data is None:
            return {
                "success": False,
                "message": "No data loaded. Call load_data() first.",
                "forecast": None,
            }

        # Get recommendation from planner
        validation = validate_time_series(self.current_data)
        planner_input = {
            "user_request": user_request,
            "data_summary": validation,
            "n_points": len(self.current_data),
        }

        plan_response = self.planner.process(planner_input)

        if not plan_response.success:
            return {
                "success": False,
                "message": f"Planning failed: {plan_response.message}",
                "forecast": None,
                "plan": None,
            }

        # Extract plan
        plan = plan_response.data

        # Use explicit parameters if provided, otherwise use plan
        final_horizon = horizon or plan.get("recommended_horizon", 7)
        final_model_name = model_name or plan.get("recommended_model", "naive")

        # Execute forecast (this is the only execution - everything else is planning)
        try:
            result = self._execute_forecast(final_model_name, final_horizon)

            return {
                "success": True,
                "message": "Forecast generated",
                "forecast": {
                    "predictions": result.predictions,
                    "dates": result.dates,
                    "model": result.model_name,
                    "metrics": result.metrics,
                },
                "plan": plan,
                "warnings": plan.get("warnings", []),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Forecast execution failed: {str(e)}",
                "forecast": None,
                "plan": plan,
            }

    def _execute_forecast(self, model_name: str, horizon: int) -> ForecastResult:
        """
        Execute forecast with specified model.

        This is the only place where execution happens.
        All other logic is planning/recommendation.
        """
        # Select and fit model
        result = select_best_model(self.current_data, test_size=0.2)

        # If specific model requested, use it (simple approach)
        if model_name == "naive":
            from forecaster.models.simple import NaiveForecaster

            model = NaiveForecaster()
            model.fit(self.current_data)
        elif model_name == "linear":
            from forecaster.models.simple import LinearForecaster

            model = LinearForecaster()
            model.fit(self.current_data)
        else:
            # Use AutoML selected model
            model = result["best_model"]

        # Generate forecast
        forecast = model.predict(horizon)

        return forecast
