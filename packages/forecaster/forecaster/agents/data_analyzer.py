"""
DataAnalyzerAgent — detects data characteristics and recommends cleaning.

Pure function: ContextWindow in → ContextWindow (with DataProfile) + AgentDecision out.
No side effects, no API calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecaster.agents.base import BaseAgent
from forecaster.core.context import (
    AgentDecision,
    ContextWindow,
    DataProfile,
)


class DataAnalyzerAgent(BaseAgent):
    """
    Analyzes time-series data and produces a DataProfile.

    Detects:
      - Frequency (daily, weekly, monthly, sub-daily)
      - Missing value percentage
      - Outliers via IQR
      - Seasonality via autocorrelation (ACF)
      - Trend via simple Mann-Kendall sign test
    """

    name: str = "data_analyzer"

    # Resource estimate: ~0.5s CPU, ~50MB per 10k rows
    _BASE_COMPUTE_S = 0.5
    _BASE_MEMORY_MB = 50
    _ROWS_PER_UNIT = 10_000

    def execute(self, context: ContextWindow) -> tuple[ContextWindow, AgentDecision]:
        context.advance_phase("data_analysis")

        df = context.get_primary_data()
        if df is None:
            decision = AgentDecision(
                agent_name=self.name,
                decision_type="error",
                action="no_data",
                reasoning="No data registered in context.",
            )
            return context, decision

        # Estimate resources
        n_rows = len(df)
        units = max(1, n_rows / self._ROWS_PER_UNIT)
        est_compute = self._BASE_COMPUTE_S * units
        est_memory = int(self._BASE_MEMORY_MB * units)

        # Reserve resources
        if not context.reserve_compute(est_compute, self.name):
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="error",
                action="compute_budget_exceeded",
                reasoning=f"Need {est_compute:.1f}s but only {context.budget.remaining_compute_seconds:.1f}s left.",
            )

        # Build profile
        profile = DataProfile(n_rows=n_rows, n_columns=len(df.columns))

        # Column types
        profile.column_types = {col: str(df[col].dtype) for col in df.columns}
        profile.numeric_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]

        # Auto-detect datetime column
        dt_col = context.datetime_column or self._detect_datetime_col(df)
        profile.datetime_column = dt_col
        if dt_col and not context.datetime_column:
            context.datetime_column = dt_col

        # Auto-detect target column
        target_col = context.target_column or self._detect_target_col(df, dt_col)
        profile.target_column = target_col
        if target_col and not context.target_column:
            context.target_column = target_col

        # Group columns
        profile.group_columns = context.group_columns

        # Frequency detection
        if dt_col and dt_col in df.columns:
            profile.frequency = self._detect_frequency(df, dt_col)
            profile.date_range = self._get_date_range(df, dt_col)

        # Missing values
        if target_col and target_col in df.columns:
            target = df[target_col]
            profile.missing_pct = round(target.isna().mean() * 100, 2)

            # Outliers (IQR)
            profile.outlier_pct = self._detect_outliers_pct(target.dropna())

            # Seasonality (ACF)
            profile.has_seasonality, profile.seasonality_period = self._detect_seasonality(
                target.dropna(), profile.frequency
            )

            # Trend (Mann-Kendall sign test — simplified)
            profile.has_trend = self._detect_trend(target.dropna())

        # Cleaning recommendations
        recommendations = self._build_recommendations(profile)
        profile.cleaning_recommendations = recommendations

        context.data_profile = profile

        # Determine if confirmation needed
        needs_confirm = profile.missing_pct > 5.0 or profile.outlier_pct > 5.0

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="data_analysis",
            action="profile_complete",
            parameters={
                "frequency": profile.frequency,
                "missing_pct": profile.missing_pct,
                "outlier_pct": profile.outlier_pct,
                "has_seasonality": profile.has_seasonality,
                "seasonality_period": profile.seasonality_period,
                "has_trend": profile.has_trend,
                "n_recommendations": len(recommendations),
            },
            confidence=0.85 if dt_col and target_col else 0.5,
            reasoning=self._build_reasoning(profile),
            requires_confirmation=needs_confirm,
            estimated_compute_seconds=est_compute,
            estimated_memory_mb=est_memory,
        )

        return context, decision

    # -- Detection helpers -------------------------------------------------

    @staticmethod
    def _detect_datetime_col(df: pd.DataFrame) -> str | None:
        """Auto-detect the datetime column."""
        # Already datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        # Name heuristic + parse check
        dt_keywords = ["date", "time", "timestamp", "dt", "datetime", "data", "dzien"]
        for col in df.columns:
            if any(kw in col.lower() for kw in dt_keywords):
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    return col
                except Exception:
                    continue

        # Brute-force parse
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    return col
                except Exception:
                    continue

        return None

    @staticmethod
    def _detect_target_col(df: pd.DataFrame, dt_col: str | None) -> str | None:
        """Auto-detect the target (value) column."""
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != dt_col]
        if len(numeric) == 1:
            return numeric[0]

        target_kw = ["value", "sales", "revenue", "price", "amount", "count", "target", "y"]
        for col in numeric:
            if any(kw in col.lower() for kw in target_kw):
                return col

        return numeric[0] if numeric else None

    @staticmethod
    def _detect_frequency(df: pd.DataFrame, dt_col: str) -> str | None:
        """Detect time series frequency."""
        try:
            dates = pd.to_datetime(df[dt_col]).sort_values()
            diffs = dates.diff().dropna()
            if len(diffs) == 0:
                return None
            mode_diff = diffs.mode().iloc[0]
            total_sec = mode_diff.total_seconds()
            if total_sec <= 0:
                return None
            if total_sec < 60:
                return f"{int(total_sec)}s"
            if total_sec < 3600:
                return f"{int(total_sec // 60)}min"
            if total_sec < 86400:
                h = total_sec / 3600
                return f"{int(h)}h" if h == int(h) else f"{int(total_sec // 60)}min"
            days = mode_diff.days
            if days == 1:
                return "D"
            if days == 7:
                return "W"
            if 28 <= days <= 31:
                return "M"
            if 365 <= days <= 366:
                return "Y"
            return f"{days}D"
        except Exception:
            return None

    @staticmethod
    def _get_date_range(df: pd.DataFrame, dt_col: str) -> tuple[str, str] | None:
        try:
            dates = pd.to_datetime(df[dt_col])
            return (dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d"))
        except Exception:
            return None

    @staticmethod
    def _detect_outliers_pct(series: pd.Series) -> float:
        """Outlier percentage using IQR method."""
        if len(series) < 4:
            return 0.0
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        return round(outliers / len(series) * 100, 2)

    @staticmethod
    def _detect_seasonality(series: pd.Series, frequency: str | None) -> tuple[bool, int | None]:
        """Detect seasonality via autocorrelation."""
        if len(series) < 14:
            return False, None

        try:
            # Determine lag to check based on frequency
            max_lag = min(len(series) // 2, 365)
            if max_lag < 7:
                return False, None

            # Compute ACF manually (no statsmodels dependency required)
            s = series.values.astype(float)
            s = s - s.mean()
            n = len(s)
            acf_vals = np.correlate(s, s, mode="full")[n - 1 :]
            acf_vals = acf_vals / acf_vals[0] if acf_vals[0] != 0 else acf_vals

            # Look for peaks in ACF (skip lag 0)
            candidate_lags = []
            if frequency in ("D", None):
                candidate_lags = [7, 14, 30, 365]
            elif frequency == "W":
                candidate_lags = [4, 13, 52]
            elif frequency == "M":
                candidate_lags = [3, 6, 12]
            elif frequency and "h" in frequency:
                candidate_lags = [24, 168]  # daily, weekly
            elif frequency and "min" in frequency:
                candidate_lags = [96, 672]  # daily (15min), weekly

            threshold = 0.3
            for lag in candidate_lags:
                if lag < len(acf_vals) and acf_vals[lag] > threshold:
                    return True, lag

            # Fallback: find first significant peak after lag 1
            for lag in range(2, min(max_lag, len(acf_vals))):
                if acf_vals[lag] > threshold:
                    return True, lag

            return False, None
        except Exception:
            return False, None

    @staticmethod
    def _detect_trend(series: pd.Series) -> bool:
        """Simplified Mann-Kendall trend test."""
        if len(series) < 10:
            return False
        try:
            n = len(series)
            # Use a sample for large series
            if n > 1000:
                idx = np.linspace(0, n - 1, 1000, dtype=int)
                s = series.values[idx]
            else:
                s = series.values.astype(float)

            # Count concordant vs discordant pairs (sampled)
            sample_n = min(len(s), 500)
            rng = np.random.RandomState(42)
            pairs = rng.choice(len(s), size=(sample_n, 2), replace=True)
            signs = np.sign(s[pairs[:, 1]] - s[pairs[:, 0]]) * np.sign(pairs[:, 1] - pairs[:, 0])
            stat = signs.sum() / sample_n

            return abs(stat) > 0.3  # threshold for "has trend"
        except Exception:
            return False

    # -- Recommendation builder --------------------------------------------

    @staticmethod
    def _build_recommendations(profile: DataProfile) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []

        if profile.missing_pct > 0:
            method = "seasonal_interp" if profile.has_seasonality else "linear_interp"
            if profile.missing_pct > 20:
                method = "drop_rows"
            recs.append(
                {
                    "type": "fill_missing",
                    "method": method,
                    "severity": "high"
                    if profile.missing_pct > 10
                    else "medium"
                    if profile.missing_pct > 2
                    else "low",
                    "detail": f"{profile.missing_pct}% missing values",
                }
            )

        if profile.outlier_pct > 2:
            recs.append(
                {
                    "type": "handle_outliers",
                    "method": "clip_iqr",
                    "severity": "medium" if profile.outlier_pct < 10 else "high",
                    "detail": f"{profile.outlier_pct}% outliers detected (IQR)",
                }
            )

        if profile.n_rows < 30:
            recs.append(
                {
                    "type": "warning",
                    "method": "insufficient_data",
                    "severity": "high",
                    "detail": f"Only {profile.n_rows} rows — forecast reliability will be low.",
                }
            )

        return recs

    @staticmethod
    def _build_reasoning(profile: DataProfile) -> str:
        parts = []
        if profile.frequency:
            parts.append(f"Detected {profile.frequency} frequency")
        if profile.missing_pct > 0:
            parts.append(f"{profile.missing_pct}% missing values")
        if profile.has_seasonality:
            parts.append(f"seasonality detected (period={profile.seasonality_period})")
        if profile.has_trend:
            parts.append("trend detected")
        if profile.outlier_pct > 0:
            parts.append(f"{profile.outlier_pct}% outliers")
        return (
            "; ".join(parts) if parts else "Data analysis complete, no significant patterns found."
        )
