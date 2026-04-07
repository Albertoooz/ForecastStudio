"""
FeatureEngineerAgent — generates feature specifications based on data profile (Polars).

Pure function: reads DataProfile from context, produces list[FeatureSpec].
Does NOT mutate the DataFrame — only specifies WHAT features to create.
Actual feature creation happens downstream (e.g., during model training).
"""

from __future__ import annotations

import numpy as np
import polars as pl

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
      - Correlation-based impact preview on sample (optional)
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

        if "holidays_pl" in context.data_registry:
            specs.append(
                FeatureSpec(
                    name="is_holiday",
                    feature_type="holiday",
                    parameters={"source": "holidays_pl"},
                    estimated_impact=0.35 if profile.has_seasonality else 0.15,
                )
            )

        specs = self._estimate_impacts(specs, context)
        context.feature_specs = specs

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

    @staticmethod
    def apply_features(
        df: pl.DataFrame,
        specs: list[FeatureSpec],
        datetime_column: str,
        target_column: str,
        holidays_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """
        Materialize feature specs into actual DataFrame columns (Polars).

        Pure function — does not modify the original DataFrame.
        """
        result = df.clone()

        if datetime_column in result.columns and not result[datetime_column].dtype.is_temporal():
            result = result.with_columns(pl.col(datetime_column).cast(pl.Datetime, strict=False))

        for spec in specs:
            try:
                if spec.feature_type == "lag" and target_column in result.columns:
                    lag = spec.parameters.get("lag", 1)
                    result = result.with_columns(pl.col(target_column).shift(lag).alias(spec.name))

                elif spec.feature_type == "rolling_mean" and target_column in result.columns:
                    w = spec.parameters.get("window", 7)
                    result = result.with_columns(
                        pl.col(target_column)
                        .rolling_mean(window_size=w, min_periods=1)
                        .alias(spec.name)
                    )

                elif spec.feature_type == "rolling_std" and target_column in result.columns:
                    w = spec.parameters.get("window", 7)
                    result = result.with_columns(
                        pl.col(target_column)
                        .rolling_std(window_size=w, min_periods=1)
                        .alias(spec.name)
                    )

                elif spec.feature_type == "date_part" and datetime_column in result.columns:
                    part = spec.parameters.get("part")
                    dt = pl.col(datetime_column).cast(pl.Datetime)
                    if part == "dayofweek":
                        result = result.with_columns(dt.dt.weekday().alias(spec.name))
                    elif part == "month":
                        result = result.with_columns(dt.dt.month().alias(spec.name))
                    elif part == "day":
                        result = result.with_columns(dt.dt.day().alias(spec.name))
                    elif part == "is_weekend":
                        result = result.with_columns(
                            (dt.dt.weekday() >= 5).cast(pl.Int8).alias(spec.name)
                        )
                    elif part == "quarter":
                        result = result.with_columns(dt.dt.quarter().alias(spec.name))

                elif spec.feature_type == "holiday":
                    if holidays_df is not None and datetime_column in result.columns:
                        result = _apply_holiday_feature(
                            result, holidays_df, datetime_column, spec.name
                        )
                    else:
                        result = result.with_columns(pl.lit(0).alias(spec.name))

            except Exception:
                result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(spec.name))

        return result

    @staticmethod
    def _select_lags(profile) -> list[int]:
        lags = [1]
        freq = profile.frequency
        if freq == "D":
            lags.extend([7, 14, 30])
        elif freq == "W":
            lags.extend([4, 8, 13])
        elif freq == "M":
            lags.extend([3, 6, 12])
        elif freq and "h" in freq:
            lags.extend([24, 168])
        else:
            lags.extend([7, 30])
        max_lag = max(1, profile.n_rows // 3)
        return sorted({la for la in lags if la <= max_lag})

    @staticmethod
    def _select_rolling_windows(profile) -> list[int]:
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
        if lag == 1:
            return 0.6
        if profile.has_seasonality and profile.seasonality_period == lag:
            return 0.5
        if lag <= 7:
            return 0.35
        return 0.2

    def _estimate_impacts(
        self, specs: list[FeatureSpec], context: ContextWindow
    ) -> list[FeatureSpec]:
        """Correlation-based impact estimation on a small sample."""
        df = context.get_primary_data()
        target = context.target_column

        if df is None or target is None or df.height < 50:
            return specs

        try:
            sample_size = min(500, df.height)
            sample = df.head(sample_size)
            if target not in sample.columns:
                return specs

            dt_col = context.datetime_column
            materialized = self.apply_features(sample, specs, dt_col or "", target)

            target_arr = materialized[target].drop_nulls().to_numpy().astype(float)
            for spec in specs:
                if spec.name in materialized.columns:
                    feat_arr = materialized[spec.name].drop_nulls().to_numpy().astype(float)
                    n = min(len(target_arr), len(feat_arr))
                    if n > 10:
                        corr = float(np.corrcoef(target_arr[:n], feat_arr[:n])[0, 1])
                        if not np.isnan(corr):
                            spec.estimated_impact = round(abs(corr), 3)
        except Exception:
            pass

        return specs

    @staticmethod
    def _build_reasoning(specs: list[FeatureSpec], profile) -> str:
        parts = [f"Generated {len(specs)} features"]
        lag_specs = [s for s in specs if s.feature_type == "lag"]
        if lag_specs:
            parts.append(f"lags [{','.join(str(s.parameters.get('lag', '?')) for s in lag_specs)}]")
        rolling = [s for s in specs if "rolling" in s.feature_type]
        if rolling:
            parts.append(f"{len(rolling)} rolling stats")
        date_s = [s for s in specs if s.feature_type == "date_part"]
        if date_s:
            parts.append(f"{len(date_s)} date features")
        holiday_s = [s for s in specs if s.feature_type == "holiday"]
        if holiday_s:
            parts.append("holiday flags")
        return "; ".join(parts) + "."


def _apply_holiday_feature(
    df: pl.DataFrame,
    holidays_df: pl.DataFrame,
    datetime_column: str,
    feature_name: str,
) -> pl.DataFrame:
    """Join holiday flags onto the main DataFrame (Polars)."""
    try:
        hol_date_col = next(
            (
                c
                for c in holidays_df.columns
                if "date" in c.lower() or holidays_df[c].dtype.is_temporal()
            ),
            None,
        )
        if hol_date_col is None:
            return df.with_columns(pl.lit(0).alias(feature_name))

        hol_dates = holidays_df[hol_date_col].cast(pl.Date, strict=False).drop_nulls().unique()
        hol_set = set(hol_dates.to_list())

        return df.with_columns(
            pl.col(datetime_column)
            .cast(pl.Date, strict=False)
            .map_elements(lambda d: int(d in hol_set), return_dtype=pl.Int8)
            .alias(feature_name)
        )
    except Exception:
        return df.with_columns(pl.lit(0).alias(feature_name))
