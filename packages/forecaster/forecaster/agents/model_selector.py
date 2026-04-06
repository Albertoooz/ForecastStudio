"""
ModelSelectorAgent — deterministic rules engine for model selection.

NO LLM involved. Pure rules based on data characteristics.
"""

from __future__ import annotations

from forecaster.agents.base import BaseAgent
from forecaster.core.context import (
    AgentDecision,
    ContextWindow,
    ModelSpec,
)


class ModelSelectorAgent(BaseAgent):
    """
    Deterministic model selector based on data profile.

    Rules:
      series_length < 100                         → simple_ewm
      strong seasonality AND holidays available    → prophet + regressors
      multiple groups                              → mlforecast_global (LightGBM)
      else                                        → lightgbm_default
    """

    name: str = "model_selector"

    def execute(self, context: ContextWindow) -> tuple[ContextWindow, AgentDecision]:
        context.advance_phase("model_selection")

        profile = context.data_profile
        if profile is None:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="error",
                action="no_profile",
                reasoning="DataProfile not available — run DataAnalyzerAgent first.",
            )

        # Gather decision inputs
        n_rows = profile.n_rows
        has_seasonality = profile.has_seasonality
        seasonality_period = profile.seasonality_period
        has_trend = profile.has_trend
        has_holidays = "holidays_pl" in context.data_registry
        has_groups = len(profile.group_columns) > 0 or len(context.group_columns) > 0
        n_features = len(context.feature_specs)

        # Apply rules
        model_spec, reasoning, confidence = self._select(
            n_rows=n_rows,
            has_seasonality=has_seasonality,
            seasonality_period=seasonality_period,
            has_trend=has_trend,
            has_holidays=has_holidays,
            has_groups=has_groups,
            n_features=n_features,
        )

        context.model_spec = model_spec

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="model_selection",
            action=f"selected: {model_spec.model_type}",
            parameters={
                "model_type": model_spec.model_type,
                "hyperparameters": model_spec.hyperparameters,
                "tuning_ranges": model_spec.tuning_ranges,
            },
            confidence=confidence,
            reasoning=reasoning,
            requires_confirmation=False,
            estimated_compute_seconds=model_spec.estimated_compute_seconds,
            estimated_memory_mb=model_spec.estimated_memory_mb,
        )

        return context, decision

    # -- Selection rules ---------------------------------------------------

    def _select(
        self,
        n_rows: int,
        has_seasonality: bool,
        seasonality_period: int | None,
        has_trend: bool,
        has_holidays: bool,
        has_groups: bool,
        n_features: int,
    ) -> tuple[ModelSpec, str, float]:
        """
        Deterministic rules engine.

        Returns (ModelSpec, reasoning, confidence).
        """

        # Rule 1: Very small dataset → simple EWM
        if n_rows < 100:
            return (
                ModelSpec(
                    model_type="simple_ewm",
                    reasoning="Series too short for complex models.",
                    hyperparameters={"span": min(n_rows // 2, 20)},
                    estimated_compute_seconds=0.5,
                    estimated_memory_mb=10,
                ),
                f"Only {n_rows} data points — using Exponential Weighted Mean (simple, robust).",
                0.7,
            )

        # Rule 2: Strong seasonality + holidays → Prophet with regressors
        if has_seasonality and has_holidays:
            return (
                ModelSpec(
                    model_type="prophet",
                    reasoning="Strong seasonality + holiday data available.",
                    hyperparameters={
                        "yearly_seasonality": True
                        if (seasonality_period and seasonality_period >= 365)
                        else "auto",
                        "weekly_seasonality": True
                        if (seasonality_period and seasonality_period in (7, 14))
                        else "auto",
                        "daily_seasonality": False,
                        "add_holidays": True,
                    },
                    tuning_ranges={
                        "changepoint_prior_scale": [0.001, 0.5],
                        "seasonality_prior_scale": [0.01, 10.0],
                    },
                    estimated_compute_seconds=5.0,
                    estimated_memory_mb=100,
                ),
                f"Seasonality detected (period={seasonality_period}) with holiday data — "
                "Prophet can model both via regressors.",
                0.88,
            )

        # Rule 3: Multiple groups → MLForecast global model
        if has_groups:
            return (
                ModelSpec(
                    model_type="mlforecast_global",
                    reasoning="Multiple groups detected — global model shares patterns.",
                    hyperparameters={
                        "n_estimators": 200,
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "max_depth": -1,
                    },
                    tuning_ranges={
                        "n_estimators": [100, 500],
                        "learning_rate": [0.01, 0.1],
                        "num_leaves": [15, 63],
                    },
                    estimated_compute_seconds=10.0,
                    estimated_memory_mb=200,
                ),
                "Multiple time series groups — MLForecast global model with LightGBM "
                "shares patterns across groups.",
                0.85,
            )

        # Rule 4: Seasonality but no holidays → Prophet (simpler)
        if has_seasonality and n_rows >= 200:
            return (
                ModelSpec(
                    model_type="prophet",
                    reasoning="Seasonality detected, sufficient data for Prophet.",
                    hyperparameters={
                        "yearly_seasonality": "auto",
                        "weekly_seasonality": "auto",
                        "daily_seasonality": False,
                    },
                    tuning_ranges={
                        "changepoint_prior_scale": [0.001, 0.5],
                    },
                    estimated_compute_seconds=3.0,
                    estimated_memory_mb=80,
                ),
                f"Seasonality detected (period={seasonality_period}) with {n_rows} rows — "
                "Prophet is suitable.",
                0.82,
            )

        # Rule 5: Default → LightGBM
        return (
            ModelSpec(
                model_type="lightgbm_default",
                reasoning="Standard time series — LightGBM provides strong baseline.",
                hyperparameters={
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": -1,
                    "min_child_samples": 10,
                },
                tuning_ranges={
                    "n_estimators": [100, 500],
                    "learning_rate": [0.01, 0.1],
                    "num_leaves": [15, 63],
                    "max_depth": [-1, 8, 12],
                },
                estimated_compute_seconds=5.0,
                estimated_memory_mb=150,
            ),
            f"{n_rows} rows, {'trend' if has_trend else 'no trend'}, "
            f"{'weak seasonality' if has_seasonality else 'no clear seasonality'} — "
            "LightGBM is a strong default.",
            0.80,
        )
