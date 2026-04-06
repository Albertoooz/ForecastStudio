"""
ExternalDataAgent — discovers and suggests external data joins.

Currently supports DuckDB tables (holidays_pl, macro_eu) stored in ./data/external/.
Gracefully degrades when DuckDB is not available or tables don't exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from forecaster.agents.base import BaseAgent
from forecaster.core.context import AgentDecision, ContextWindow

# Default path for external data directory (DuckDB or CSV fallback)
_EXTERNAL_DATA_DIR = Path(__file__).parent.parent / "data" / "external"


class ExternalDataAgent(BaseAgent):
    """
    Agent that discovers external datasets and suggests joins.

    Phase 1 (current):  Looks for CSV/Parquet files in data/external/
    Phase 2 (future):   DuckDB connection for holidays_pl, macro_eu tables

    Auto-detects date overlaps and suggests joins with confidence score.
    """

    name: str = "external_data"

    def __init__(self, external_dir: Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.external_dir = external_dir or _EXTERNAL_DATA_DIR

    def execute(self, context: ContextWindow) -> tuple[ContextWindow, AgentDecision]:
        context.advance_phase("external_data")

        df = context.get_primary_data()
        dt_col = context.datetime_column

        if df is None or dt_col is None:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="external_data_join",
                action="skip",
                reasoning="No primary data or datetime column — cannot evaluate joins.",
                confidence=1.0,
            )

        # Parse primary dates
        try:
            primary_dates = pd.to_datetime(df[dt_col]).dt.normalize()
            primary_set = set(primary_dates.dropna().dt.strftime("%Y-%m-%d"))
        except Exception:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="external_data_join",
                action="skip",
                reasoning="Could not parse datetime column for overlap analysis.",
                confidence=1.0,
            )

        # Discover available external tables
        available_tables = self._discover_tables()

        if not available_tables:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="external_data_join",
                action="no_external_data",
                reasoning=f"No external data found in {self.external_dir}. "
                "Create CSV/Parquet files there to enable auto-joins.",
                confidence=1.0,
            )

        # Evaluate each table for join suitability
        suggestions: list[dict[str, Any]] = []
        for table_name, table_df in available_tables.items():
            suggestion = self._evaluate_join(table_name, table_df, primary_set, context)
            if suggestion:
                suggestions.append(suggestion)

        if not suggestions:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="external_data_join",
                action="no_suitable_joins",
                reasoning="External tables found but no suitable date overlap for joining.",
                confidence=0.9,
            )

        # Best suggestion
        best = max(suggestions, key=lambda s: s["overlap_pct"])

        # If overlap > 90% and seasonality detected → high confidence
        seasonality = context.data_profile.has_seasonality if context.data_profile else False
        confidence = min(best["overlap_pct"] / 100.0, 0.95)
        if seasonality and best["overlap_pct"] > 80:
            confidence = min(confidence + 0.1, 0.98)

        # Needs confirmation for any join
        decision = AgentDecision(
            agent_name=self.name,
            decision_type="external_data_join",
            action=f"join_external: {best['table_name']} ON date",
            parameters={
                "table_name": best["table_name"],
                "join_key": "date",
                "overlap_pct": best["overlap_pct"],
                "n_features": best["n_features"],
                "feature_columns": best["feature_columns"],
                "all_suggestions": suggestions,
            },
            confidence=confidence,
            reasoning=(
                f"Found {best['overlap_pct']:.0f}% date overlap with {best['table_name']} "
                f"({best['n_features']} features). "
                + (
                    "Seasonality detected — external features likely helpful."
                    if seasonality
                    else ""
                )
            ),
            requires_confirmation=True,  # Always ask user before joining external data
            estimated_compute_seconds=0.2,
            estimated_memory_mb=max(5, best.get("size_mb", 5)),
        )

        return context, decision

    # -- Join execution (called after user confirms) -----------------------

    def apply_join(self, context: ContextWindow, table_name: str) -> ContextWindow:
        """
        Apply the external data join after user confirmation.

        Returns updated context with joined data registered.
        """
        tables = self._discover_tables()
        if table_name not in tables:
            context.log_decision(
                AgentDecision(
                    agent_name=self.name,
                    decision_type="error",
                    action=f"table_not_found: {table_name}",
                    reasoning=f"Table {table_name} not found in external data directory.",
                )
            )
            return context

        ext_df = tables[table_name]
        df = context.get_primary_data()
        dt_col = context.datetime_column

        if df is None or dt_col is None:
            return context

        try:
            # Normalize dates for join
            df_copy = df.copy()
            df_copy["__join_date"] = pd.to_datetime(df_copy[dt_col]).dt.normalize()

            # Find date column in external table
            ext_date_col = self._find_date_col(ext_df)
            if ext_date_col is None:
                return context

            ext_df_copy = ext_df.copy()
            ext_df_copy["__join_date"] = pd.to_datetime(ext_df_copy[ext_date_col]).dt.normalize()

            # Get feature columns (non-date)
            feature_cols = [c for c in ext_df.columns if c != ext_date_col]

            # Left join
            merged = df_copy.merge(
                ext_df_copy[["__join_date"] + feature_cols],
                on="__join_date",
                how="left",
            )
            merged.drop(columns=["__join_date"], inplace=True)

            # Register the external data and update primary
            context.register_data(table_name, ext_df)
            first_key = next(iter(context.data_registry))
            context.data_registry[first_key] = merged

            context.log_decision(
                AgentDecision(
                    agent_name=self.name,
                    decision_type="external_data_join",
                    action=f"joined: {table_name}",
                    parameters={"features_added": feature_cols, "rows": len(merged)},
                    confidence=0.95,
                    reasoning=f"Joined {len(feature_cols)} features from {table_name}.",
                )
            )
        except Exception as e:
            context.log_decision(
                AgentDecision(
                    agent_name=self.name,
                    decision_type="error",
                    action=f"join_failed: {table_name}",
                    reasoning=str(e),
                )
            )

        return context

    # -- Discovery helpers -------------------------------------------------

    def _discover_tables(self) -> dict[str, pd.DataFrame]:
        """Discover available external data tables."""
        tables: dict[str, pd.DataFrame] = {}

        if not self.external_dir.exists():
            return tables

        for fp in self.external_dir.iterdir():
            if fp.suffix == ".csv":
                try:
                    tables[fp.stem] = pd.read_csv(fp)
                except Exception:
                    continue
            elif fp.suffix == ".parquet":
                try:
                    tables[fp.stem] = pd.read_parquet(fp)
                except Exception:
                    continue

        return tables

    def _evaluate_join(
        self,
        table_name: str,
        table_df: pd.DataFrame,
        primary_dates: set[str],
        context: ContextWindow,
    ) -> dict[str, Any] | None:
        """Evaluate a single external table for join suitability."""
        date_col = self._find_date_col(table_df)
        if date_col is None:
            return None

        try:
            ext_dates = pd.to_datetime(table_df[date_col]).dt.normalize()
            ext_set = set(ext_dates.dropna().dt.strftime("%Y-%m-%d"))
        except Exception:
            return None

        overlap = primary_dates & ext_set
        if len(primary_dates) == 0:
            return None

        overlap_pct = len(overlap) / len(primary_dates) * 100
        if overlap_pct < 10:
            return None  # Too little overlap

        feature_cols = [c for c in table_df.columns if c != date_col]

        return {
            "table_name": table_name,
            "overlap_pct": round(overlap_pct, 1),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "size_mb": table_df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

    @staticmethod
    def _find_date_col(df: pd.DataFrame) -> str | None:
        """Find the date column in a DataFrame."""
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        dt_kw = ["date", "time", "timestamp", "dt", "data"]
        for col in df.columns:
            if any(kw in col.lower() for kw in dt_kw):
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    return col
                except Exception:
                    continue
        return None
