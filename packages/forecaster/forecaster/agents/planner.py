"""Forecasting planner agent - provides recommendations, never executes."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from forecaster.agents.base import AgentResponse, BaseAgent
from forecaster.utils.llm_env import get_openai_compatible_settings

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Try default location


class ForecastingPlan(BaseModel):
    """Structured forecasting plan from the planner."""

    recommended_model: str
    recommended_horizon: int
    reasoning: str
    data_requirements: list[str]
    warnings: list[str] = []


class ForecastingPlanner(BaseAgent):
    """
    Planning agent for forecasting tasks.

    This agent analyzes user requests and data characteristics,
    then provides structured recommendations. It never executes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__("forecasting_planner")

        env_key, env_base, env_model = get_openai_compatible_settings()

        if api_key is None:
            api_key = env_key
        if base_url is None:
            base_url = env_base
        if model is None:
            model = env_model

        if not api_key:
            raise ValueError(
                "API key not found. Set LLM_API_KEY (or legacy DEEPSEEK_API_KEY / OPENAI_API_KEY) "
                "or pass api_key parameter"
            )

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = model
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai-compatible")

    def process(self, input_data: dict[str, Any]) -> AgentResponse:
        """
        Analyze forecasting request and return structured plan.

        Expected input_data:
        {
            "user_request": str,
            "data_summary": Dict[str, Any],  # From validate_time_series
            "n_points": int,
        }
        """
        try:
            # Validate input
            if "user_request" not in input_data:
                return AgentResponse(
                    success=False,
                    message="Missing 'user_request' in input",
                    data={},
                    errors=["user_request is required"],
                )

            user_request = input_data["user_request"]
            data_summary = input_data.get("data_summary", {})
            n_points = input_data.get("n_points", 0)

            # Create prompt for structured output
            prompt = self._create_planning_prompt(user_request, data_summary, n_points)

            # Get structured recommendation from LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a forecasting advisor. Provide structured recommendations only. Never execute actions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            # Parse structured response
            plan_data = json.loads(response.choices[0].message.content)

            # Validate structure
            plan = ForecastingPlan(**plan_data)

            return AgentResponse(
                success=True,
                message="Forecasting plan generated",
                data=plan.model_dump(),
                errors=[],
            )

        except json.JSONDecodeError as e:
            return AgentResponse(
                success=False,
                message="Failed to parse LLM response",
                data={},
                errors=[f"JSON decode error: {str(e)}"],
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message="Error in planning",
                data={},
                errors=[str(e)],
            )

    def _create_planning_prompt(
        self,
        user_request: str,
        data_summary: dict[str, Any],
        n_points: int,
    ) -> str:
        """Create prompt for structured planning."""
        return f"""Analyze this forecasting request and provide a structured recommendation.

User Request: {user_request}

Data Summary:
- Number of points: {n_points}
- Has missing values: {data_summary.get("has_missing", False)}
- Number of missing: {data_summary.get("n_missing", 0)}
- Validation errors: {data_summary.get("errors", [])}

Provide a JSON response with this exact structure:
{{
    "recommended_model": "naive" or "linear",
    "recommended_horizon": <integer number of periods>,
    "reasoning": "<brief explanation>",
    "data_requirements": ["<requirement 1>", "<requirement 2>"],
    "warnings": ["<warning 1 if any>", "<warning 2 if any>"]
}}

Guidelines:
- For < 30 points, recommend "naive"
- For >= 30 points, recommend "linear"
- Recommended horizon should be reasonable (typically 1-30 periods)
- Include warnings if data quality is poor
- Be explicit about any limitations
"""
