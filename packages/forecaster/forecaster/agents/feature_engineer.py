"""
FeatureEngineerAgent — generates feature specifications based on data profile.

Pure function: reads DataProfile from context, produces list[FeatureSpec].
Does NOT mutate the DataFrame — only specifies WHAT features to create.
Actual feature creation happens downstream (e.g., during model training).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster.agents.base import BaseAgent
from forecaster.core.context import (
    AgentDecision,
    ContextWindow,
    FeatureSpec,
)


class FeatureEngineerAgent(BaseAgent):
    """
    Generates feature specifications based on data characteristics.

    Features:
      - Lag features (1, 7, 30)
      - Rolling statistics (7d mean/std)
      - Date-part features (day_of_week, month, is_weekend)
      - Holiday flags (if holidays_pl in context.data_registry)
      - SHAP-based impact preview on sample (optional)
    """

    name: str = "feature_engineer"

    def execute(self, context: ContextWindow) -> tuple[ContextWindow, AgentDecision]:
        context.advance_phase("feature_engineering")

        profile = context.data_profile
        if profile is None:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="error",
                action="no_profile",
                reasoning="DataProfile not available — run DataAnalyzerAgent first.",
            )

        specs: list[FeatureSpec] = []

        # 1. Lag features
        lag_values = self._select_lags(profile)
        for lag in lag_values:
            specs.append(
                FeatureSpec(
                    name=f"lag_{lag}",
                    feature_type="lag",
                    parameters={"lag": lag},
                    estimated_impact=self._estimate_lag_impact(lag, profile),
                )
            )

        # 2. Rolling statistics
        windows = self._select_rolling_windows(profile)
        for w in windows:
            specs.append(
                FeatureSpec(
                    name=f"rolling_mean_{w}",
                    feature_type="rolling_mean",
                    parameters={"window": w},
                    estimated_impact=0.3,
                )
            )
            specs.append(
                FeatureSpec(
                    name=f"rolling_std_{w}",
                    feature_type="rolling_std",
                    parameters={"window": w},
                    estimated_impact=0.15,
                )
            )

        # 3. Date-part features
        date_features = self._select_date_features(profile)
        for feat_name, params in date_features:
            specs.append(
                FeatureSpec(
                    name=feat_name,
                    feature_type="date_part",
                    parameters=params,
                    estimated_impact=0.25 if profile.has_seasonality else 0.1,
                )
            )

        # 4. Holiday features (if available)
        if "holidays_pl" in context.data_registry:
            specs.append(
                FeatureSpec(
                    name="is_holiday",
                    feature_type="holiday",
                    parameters={"source": "holidays_pl"},
                    estimated_impact=0.35 if profile.has_seasonality else 0.15,
                )
            )

        # 5. Compute SHAP preview on sample (lightweight)
        specs = self._estimate_shap_impacts(specs, context)

        context.feature_specs = specs

        # Resource estimates
        n_features = len(specs)
        est_compute = 0.1 * n_features
        est_memory = max(10, 2 * n_features)

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="feature_engineering",
            action="features_specified",
            parameters={
                "n_features": n_features,
                "feature_names": [s.name for s in specs],
                "lag_values": lag_values,
                "rolling_windows": windows,
            },
            confidence=0.82,
            reasoning=self._build_reasoning(specs, profile),
            requires_confirmation=False,
            estimated_compute_seconds=est_compute,
            estimated_memory_mb=est_memory,
        )

        return context, decision

    # -- Feature creation (materialize specs into DataFrame) ---------------

    @staticmethod
    def apply_features(
        df: pd.DataFrame,
        specs: list[FeatureSpec],
        datetime_column: str,
        target_column: str,
        holidays_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Materialize feature specs into actual DataFrame columns.

        This is a pure function — does not modify the original DataFrame.
        Called by the orchestrator after the agent produces specs.
        """
        result = df.copy()

        # Ensure datetime is proper type
        if datetime_column in result.columns:
            result[datetime_column] = pd.to_datetime(result[datetime_column])

        for spec in specs:
            try:
                if spec.feature_type == "lag" and target_column in result.columns:
                    lag = spec.parameters.get("lag", 1)
                    result[spec.name] = result[target_column].shift(lag)

                elif spec.feature_type == "rolling_mean" and target_column in result.columns:
                    w = spec.parameters.get("window", 7)
                    result[spec.name] = result[target_column].rolling(window=w).mean()

                elif spec.feature_type == "rolling_std" and target_column in result.columns:
                    w = spec.parameters.get("window", 7)
                    result[spec.name] = result[target_column].rolling(window=w).std()

                elif spec.feature_type == "date_part" and datetime_column in result.columns:
                    dt = result[datetime_column]
                    part = spec.parameters.get("part")
                    if part == "dayofweek":
                        result[spec.name] = dt.dt.dayofweek
                    elif part == "month":
                        result[spec.name] = dt.dt.month
                    elif part == "day":
                        result[spec.name] = dt.dt.day
                    elif part == "is_weekend":
                        result[spec.name] = (dt.dt.dayofweek >= 5).astype(int)
                    elif part == "quarter":
                        result[spec.name] = dt.dt.quarter

                elif spec.feature_type == "holiday":
                    if holidays_df is not None and datetime_column in result.columns:
                        result = _apply_holiday_feature(
                            result, holidays_df, datetime_column, spec.name
                        )
                    else:
                        result[spec.name] = 0

            except Exception:
                # Skip features that fail gracefully
                result[spec.name] = np.nan

        return result

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _select_lags(profile) -> list[int]:
        """Select lag values based on data characteristics."""
        lags = [1]  # Always include lag-1

        freq = profile.frequency
        if freq == "D":
            lags.extend([7, 14, 30])
        elif freq == "W":
            lags.extend([4, 8, 13])
        elif freq == "M":
            lags.extend([3, 6, 12])
        elif freq and "h" in freq:
            lags.extend([24, 168])  # daily, weekly in hours
        else:
            lags.extend([7, 30])

        # Filter lags that are too large for the data
        max_lag = max(1, profile.n_rows // 3)
        return sorted({l for l in lags if l <= max_lag})  # noqa: E741

    @staticmethod
    def _select_rolling_windows(profile) -> list[int]:
        """Select rolling window sizes."""
        freq = profile.frequency
        if freq == "D":
            windows = [7, 14, 30]
        elif freq == "W":
            windows = [4, 8]
        elif freq == "M":
            windows = [3, 6]
        else:
            windows = [7, 14]

        max_w = max(1, profile.n_rows // 4)
        return [w for w in windows if w <= max_w]

    @staticmethod
    def _select_date_features(profile) -> list[tuple[str, dict]]:
        """Select relevant date-part features."""
        features = []
        freq = profile.frequency

        if freq in ("D", None) or (freq and ("h" in freq or "min" in freq)):
            features.append(("day_of_week", {"part": "dayofweek"}))
            features.append(("is_weekend", {"part": "is_weekend"}))
        if freq in ("D", "W", None):
            features.append(("month", {"part": "month"}))
        if freq in ("D", None):
            features.append(("day_of_month", {"part": "day"}))

        return features

    @staticmethod
    def _estimate_lag_impact(lag: int, profile) -> float:
        """Rough estimate of lag feature importance."""
        if lag == 1:
            return 0.6
        if profile.has_seasonality and profile.seasonality_period == lag:
            return 0.5
        if lag <= 7:
            return 0.35
        return 0.2

    def _estimate_shap_impacts(
        self, specs: list[FeatureSpec], context: ContextWindow
    ) -> list[FeatureSpec]:
        """
        Lightweight SHAP-inspired impact estimation on a small sample.
        Falls back to heuristic estimates if data is insufficient.
        """
        df = context.get_primary_data()
        target = context.target_column

        if df is None or target is None or len(df) < 50:
            return specs  # Keep heuristic estimates

        try:
            # Use correlation as a quick proxy for importance
            sample = df.head(min(500, len(df))).copy()
            if target not in sample.columns:
                return specs

            dt_col = context.datetime_column
            if dt_col and dt_col in sample.columns:
                sample[dt_col] = pd.to_datetime(sample[dt_col])

            # Create temp features and measure correlation
            materialized = self.apply_features(sample, specs, dt_col or "", target)

            target_series = materialized[target].dropna()
            for spec in specs:
                if spec.name in materialized.columns:
                    feat_series = materialized[spec.name].dropna()
                    common_idx = target_series.index.intersection(feat_series.index)
                    if len(common_idx) > 10:
                        corr = abs(target_series.loc[common_idx].corr(feat_series.loc[common_idx]))
                        if not np.isnan(corr):
                            spec.estimated_impact = round(corr, 3)
        except Exception:
            pass  # Keep heuristic estimates

        return specs

    @staticmethod
    def _build_reasoning(specs: list[FeatureSpec], profile) -> str:
        parts = [f"Generated {len(specs)} features"]

        lag_specs = [s for s in specs if s.feature_type == "lag"]
        if lag_specs:
            parts.append(f"lags [{','.join(str(s.parameters.get('lag', '?')) for s in lag_specs)}]")

        rolling_specs = [s for s in specs if "rolling" in s.feature_type]
        if rolling_specs:
            parts.append(f"{len(rolling_specs)} rolling stats")

        date_specs = [s for s in specs if s.feature_type == "date_part"]
        if date_specs:
            parts.append(f"{len(date_specs)} date features")

        holiday_specs = [s for s in specs if s.feature_type == "holiday"]
        if holiday_specs:
            parts.append("holiday flags")

        return "; ".join(parts) + "."


def _apply_holiday_feature(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    datetime_column: str,
    feature_name: str,
) -> pd.DataFrame:
    """Join holiday flags onto the main DataFrame."""
    result = df.copy()
    try:
        # Find date column in holidays
        hol_date_col = None
        for c in holidays_df.columns:
            if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(holidays_df[c]):
                hol_date_col = c
                break
        if hol_date_col is None:
            result[feature_name] = 0
            return result

        hol_dates = set(pd.to_datetime(holidays_df[hol_date_col]).dt.normalize())
        result[feature_name] = (
            pd.to_datetime(result[datetime_column]).dt.normalize().isin(hol_dates).astype(int)
        )
    except Exception:
        result[feature_name] = 0

    return result
