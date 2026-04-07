"""
ExternalDataAgent — discovers and suggests external data joins (Polars).

Currently supports CSV/Parquet files stored in ./data/external/.
Gracefully degrades when no external files exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from forecaster.agents.base import BaseAgent
from forecaster.core.context import AgentDecision, ContextWindow

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

        try:
            primary_set = set(
                df[dt_col].cast(pl.Date, strict=False).drop_nulls().cast(pl.Utf8).to_list()
            )
        except Exception:
            return context, AgentDecision(
                agent_name=self.name,
                decision_type="external_data_join",
                action="skip",
                reasoning="Could not parse datetime column for overlap analysis.",
                confidence=1.0,
            )

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

        best = max(suggestions, key=lambda s: s["overlap_pct"])
        seasonality = context.data_profile.has_seasonality if context.data_profile else False
        confidence = min(best["overlap_pct"] / 100.0, 0.95)
        if seasonality and best["overlap_pct"] > 80:
            confidence = min(confidence + 0.1, 0.98)

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
            requires_confirmation=True,
            estimated_compute_seconds=0.2,
            estimated_memory_mb=max(5, best.get("size_mb", 5)),
        )

        return context, decision

    def apply_join(self, context: ContextWindow, table_name: str) -> ContextWindow:
        """Apply the external data join after user confirmation."""
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
            ext_date_col = self._find_date_col(ext_df)
            if ext_date_col is None:
                return context

            feature_cols = [c for c in ext_df.columns if c != ext_date_col]

            # Normalize both date columns to Date for join
            df_join = df.with_columns(
                pl.col(dt_col).cast(pl.Date, strict=False).alias("__join_date")
            )
            ext_join = ext_df.with_columns(
                pl.col(ext_date_col).cast(pl.Date, strict=False).alias("__join_date")
            ).select(["__join_date"] + feature_cols)

            merged = df_join.join(ext_join, on="__join_date", how="left").drop("__join_date")

            context.register_data(table_name, ext_df)
            first_key = next(iter(context.data_registry))
            context.data_registry[first_key] = merged

            context.log_decision(
                AgentDecision(
                    agent_name=self.name,
                    decision_type="external_data_join",
                    action=f"joined: {table_name}",
                    parameters={"features_added": feature_cols, "rows": merged.height},
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

    def _discover_tables(self) -> dict[str, pl.DataFrame]:
        tables: dict[str, pl.DataFrame] = {}
        if not self.external_dir.exists():
            return tables
        for fp in self.external_dir.iterdir():
            if fp.suffix == ".csv":
                try:
                    tables[fp.stem] = pl.read_csv(fp)
                except Exception:
                    continue
            elif fp.suffix == ".parquet":
                try:
                    tables[fp.stem] = pl.read_parquet(fp)
                except Exception:
                    continue
        return tables

    def _evaluate_join(
        self,
        table_name: str,
        table_df: pl.DataFrame,
        primary_dates: set[str],
        context: ContextWindow,
    ) -> dict[str, Any] | None:
        date_col = self._find_date_col(table_df)
        if date_col is None:
            return None

        try:
            ext_set = set(
                table_df[date_col].cast(pl.Date, strict=False).drop_nulls().cast(pl.Utf8).to_list()
            )
        except Exception:
            return None

        overlap = primary_dates & ext_set
        if len(primary_dates) == 0:
            return None

        overlap_pct = len(overlap) / len(primary_dates) * 100
        if overlap_pct < 10:
            return None

        feature_cols = [c for c in table_df.columns if c != date_col]
        size_mb = table_df.estimated_size("mb")

        return {
            "table_name": table_name,
            "overlap_pct": round(overlap_pct, 1),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "size_mb": size_mb,
        }

    @staticmethod
    def _find_date_col(df: pl.DataFrame) -> str | None:
        for col in df.columns:
            if df[col].dtype.is_temporal():
                return col
        dt_kw = ["date", "time", "timestamp", "dt", "data"]
        for col in df.columns:
            if any(kw in col.lower() for kw in dt_kw):
                try:
                    df[col].drop_nulls().head(5).cast(pl.Utf8).str.to_datetime(strict=False)
                    return col
                except Exception:
                    continue
        return None
