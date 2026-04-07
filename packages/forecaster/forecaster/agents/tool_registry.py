"""
Extensible Tool Registry for Agent Chat System.

Architecture:
  - Each tool is a lightweight descriptor → returns dict with action/operation
  - app.py handles execution (side effects, session state, re-runs)
  - Tools are grouped into categories for clarity
  - New agents register their own tools via register_agent_tools()

Tool result contract:
  {"action": "<name>", ...}          → config change handled by app.py
  {"operation": "<name>", "parameters": {...}}  → data operation via DataOperations
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for agent tools with OpenAI function calling schemas.

    Features:
    - Automatic schema validation
    - Usage tracking & success-rate metrics
    - Extensible — agents can register their own tools
    """

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}
        self._usage_stats: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: dict[str, Any],
        category: str = "general",
        examples: list[str] | None = None,
    ):
        """Register a tool with its OpenAI function-calling schema."""
        if name in self.tools:
            logger.debug(f"Tool '{name}' overwritten.")

        self.tools[name] = {
            "function": function,
            "category": category,
            "schema": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
            "examples": examples or [],
        }
        self._usage_stats[name] = {"calls": 0, "successes": 0, "failures": 0}

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI function-calling format."""
        return [{"type": "function", "function": t["schema"]} for t in self.tools.values()]

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            available = ", ".join(sorted(self.tools.keys()))
            return {
                "success": False,
                "error": f"Unknown tool '{tool_name}'. Available: {available}",
            }

        tool = self.tools[tool_name]
        self._usage_stats[tool_name]["calls"] += 1

        try:
            result = tool["function"](**arguments)
            self._usage_stats[tool_name]["successes"] += 1
            return {"success": True, "result": result, "tool_name": tool_name}
        except Exception as e:
            self._usage_stats[tool_name]["failures"] += 1
            logger.error(f"Tool '{tool_name}' failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "tool_name": tool_name}

    def list_tools(self, category: str | None = None) -> list[str]:
        """List tool names, optionally filtered by category."""
        if category:
            return [n for n, t in self.tools.items() if t.get("category") == category]
        return list(self.tools.keys())

    def get_tool_description(self, tool_name: str) -> str:
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        return self.tools[tool_name]["schema"]["description"]

    def get_tool_stats(self, tool_name: str) -> dict[str, Any]:
        stats = self._usage_stats.get(tool_name, {})
        total = stats.get("calls", 0)
        return {
            **stats,
            "success_rate": stats.get("successes", 0) / total if total else 0.0,
        }


# ======================================================================
#  Global singleton
# ======================================================================
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get (or create) the global tool registry singleton."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _register_all_tools(_registry)
    return _registry


# ======================================================================
#  Tool definitions — grouped by category
# ======================================================================


def _register_all_tools(r: ToolRegistry):
    """Register every built-in tool."""
    _register_data_tools(r)
    _register_config_tools(r)
    _register_missing_data_tools(r)
    _register_feature_tools(r)
    _register_model_tools(r)
    _register_pipeline_tools(r)
    _register_inspect_tools(r)
    _register_dataset_tools(r)
    logger.info(f"Registered {len(r.list_tools())} tools")


# ------------------------------------------------------------------
# 1. DATA MANIPULATION
# ------------------------------------------------------------------
def _register_data_tools(r: ToolRegistry):

    r.register_tool(
        name="combine_datetime",
        category="data",
        function=lambda date_column, time_column, output_column="datetime": {
            "operation": "combine_datetime",
            "parameters": {
                "date_column": date_column,
                "time_column": time_column,
                "output_column": output_column,
            },
        },
        description="Combine separate date and time columns into a single datetime column",
        parameters={
            "type": "object",
            "properties": {
                "date_column": {"type": "string", "description": "Column with dates"},
                "time_column": {"type": "string", "description": "Column with times"},
                "output_column": {
                    "type": "string",
                    "description": "Name for new column (default: datetime)",
                    "default": "datetime",
                },
            },
            "required": ["date_column", "time_column"],
        },
    )

    r.register_tool(
        name="create_column",
        category="data",
        function=lambda column_name, expression, description="": {
            "operation": "add_column",
            "parameters": {"column_name": column_name, "expression": expression},
        },
        description=(
            "Create a new calculated column using a Polars expression (e.g. pl.col('a') + pl.col('b')). "
            "For grouped data use df.groupby(...)['col'].transform(func). "
            "Example: df['price'] * df['quantity']"
        ),
        parameters={
            "type": "object",
            "properties": {
                "column_name": {"type": "string", "description": "Name for new column"},
                "expression": {"type": "string", "description": "Pandas expression to evaluate"},
            },
            "required": ["column_name", "expression"],
        },
    )

    r.register_tool(
        name="drop_column",
        category="data",
        function=lambda column: {
            "operation": "drop_column",
            "parameters": {"column": column},
        },
        description="Remove a column from the dataset",
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Column to remove"},
            },
            "required": ["column"],
        },
    )

    r.register_tool(
        name="rename_column",
        category="data",
        function=lambda old_name, new_name: {
            "operation": "rename_column",
            "parameters": {"old_name": old_name, "new_name": new_name},
        },
        description="Rename a column",
        parameters={
            "type": "object",
            "properties": {
                "old_name": {"type": "string"},
                "new_name": {"type": "string"},
            },
            "required": ["old_name", "new_name"],
        },
    )

    r.register_tool(
        name="filter_rows",
        category="data",
        function=lambda column, condition, value: {
            "operation": "filter",
            "parameters": {"column": column, "condition": condition, "value": value},
        },
        description="Filter rows: column > / < / == / >= / <= value",
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "condition": {"type": "string", "enum": [">", "<", "==", ">=", "<=", "!="]},
                "value": {
                    "type": "string",
                    "description": "Value to compare (auto-cast to number if possible)",
                },
            },
            "required": ["column", "condition", "value"],
        },
    )

    r.register_tool(
        name="filter_date_range",
        category="data",
        function=lambda start_date=None, end_date=None: {
            "operation": "filter_date_range",
            "parameters": {"start_date": start_date, "end_date": end_date},
        },
        description="Filter data to a date range. Use ISO format: 2024-01-01",
        parameters={
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date (inclusive), e.g. 2024-06-01",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (inclusive), e.g. 2025-01-01",
                },
            },
        },
    )

    r.register_tool(
        name="sort_data",
        category="data",
        function=lambda column, ascending=True: {
            "operation": "sort",
            "parameters": {"column": column, "ascending": ascending},
        },
        description="Sort the dataset by a column",
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "ascending": {"type": "boolean", "default": True},
            },
            "required": ["column"],
        },
    )

    r.register_tool(
        name="resample_data",
        category="data",
        function=lambda frequency, agg_function="mean": {
            "operation": "resample",
            "parameters": {"frequency": frequency, "agg_function": agg_function},
        },
        description="Resample (change frequency) of the time series. Freq: 15min, h, D, W, M",
        parameters={
            "type": "object",
            "properties": {
                "frequency": {
                    "type": "string",
                    "description": "Target frequency: 15min, h, D, W, M",
                },
                "agg_function": {
                    "type": "string",
                    "enum": ["mean", "sum", "min", "max", "first", "last"],
                    "default": "mean",
                },
            },
            "required": ["frequency"],
        },
    )


# ------------------------------------------------------------------
# 2. CONFIGURATION (datetime, target, grouping, horizon, gap)
# ------------------------------------------------------------------
def _register_config_tools(r: ToolRegistry):

    r.register_tool(
        name="set_datetime_column",
        category="config",
        function=lambda column: {"action": "set_datetime_column", "column": column},
        description="Set which column is the datetime index for forecasting",
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Datetime column name"},
            },
            "required": ["column"],
        },
    )

    r.register_tool(
        name="set_target_column",
        category="config",
        function=lambda column: {"action": "set_target_column", "column": column},
        description="Set which column to forecast (target variable)",
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Target column name"},
            },
            "required": ["column"],
        },
    )

    r.register_tool(
        name="set_grouping_columns",
        category="config",
        function=lambda columns: {"action": "set_group_by", "columns": columns},
        description="Set grouping columns for multi-series forecasting (e.g. store_id, product_id)",
        parameters={
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names to group by",
                },
            },
            "required": ["columns"],
        },
    )

    r.register_tool(
        name="set_horizon",
        category="config",
        function=lambda horizon: {"action": "set_horizon", "horizon": int(horizon)},
        description="Set how many periods ahead to forecast",
        parameters={
            "type": "object",
            "properties": {
                "horizon": {"type": "integer", "description": "Number of periods to forecast"},
            },
            "required": ["horizon"],
        },
    )

    r.register_tool(
        name="set_gap",
        category="config",
        function=lambda gap: {"action": "set_gap", "gap": int(gap)},
        description="Set the gap (delay) between now and first forecast point. Lags 1..gap are excluded.",
        parameters={
            "type": "object",
            "properties": {
                "gap": {"type": "integer", "description": "Gap in periods (0 = no gap)"},
            },
            "required": ["gap"],
        },
    )


# ------------------------------------------------------------------
# 3. MISSING DATA HANDLING
# ------------------------------------------------------------------
def _register_missing_data_tools(r: ToolRegistry):

    r.register_tool(
        name="fill_missing",
        category="missing_data",
        function=lambda column, method="ffill", value=None: {
            "operation": "fill_missing",
            "parameters": {"column": column, "method": method, "value": value},
        },
        description=(
            "Fill missing values in a column. Methods: "
            "ffill (forward-fill), bfill (back-fill), mean, median, zero, interpolate, value (custom). "
            "Use column='__all__' to fill all columns."
        ),
        parameters={
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Column name or '__all__' for all columns",
                },
                "method": {
                    "type": "string",
                    "enum": ["ffill", "bfill", "mean", "median", "zero", "interpolate", "value"],
                    "default": "ffill",
                    "description": "Fill strategy",
                },
                "value": {"description": "Custom fill value (only used when method='value')"},
            },
            "required": ["column"],
        },
    )

    r.register_tool(
        name="drop_missing_rows",
        category="missing_data",
        function=lambda column=None, threshold=None: {
            "operation": "drop_missing",
            "parameters": {"column": column, "threshold": threshold},
        },
        description="Drop rows with missing values. Optionally only for a specific column or threshold.",
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Column to check (None = any column)"},
                "threshold": {
                    "type": "number",
                    "description": "Min non-null values required per row",
                },
            },
        },
    )

    r.register_tool(
        name="show_missing_summary",
        category="missing_data",
        function=lambda: {"action": "show_missing_summary"},
        description="Show a summary of missing values per column (count and percentage)",
        parameters={"type": "object", "properties": {}},
    )


# ------------------------------------------------------------------
# 4. FEATURE ENGINEERING OVERRIDES
# ------------------------------------------------------------------
def _register_feature_tools(r: ToolRegistry):

    r.register_tool(
        name="set_lags",
        category="features",
        function=lambda lags: {"action": "set_lags", "lags": [int(l) for l in lags]},  # noqa: E741
        description=(
            "Override the automatically-generated lag values for the next forecast run. "
            "Example: [1, 4, 96] for 15-min data (1-period, 1-hour, 1-day). "
            "THIS OVERRIDES FeatureEngineerAgent — set to [] to reset to auto."
        ),
        parameters={
            "type": "object",
            "properties": {
                "lags": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of lag periods (e.g. [1, 7, 14, 30])",
                },
            },
            "required": ["lags"],
        },
    )

    r.register_tool(
        name="set_rolling_windows",
        category="features",
        function=lambda windows: {
            "action": "set_rolling_windows",
            "windows": [int(w) for w in windows],
        },
        description="Override rolling window sizes (mean/std). Set to [] to reset to auto.",
        parameters={
            "type": "object",
            "properties": {
                "windows": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Rolling window sizes (e.g. [7, 14, 30])",
                },
            },
            "required": ["windows"],
        },
    )

    r.register_tool(
        name="set_date_features",
        category="features",
        function=lambda features: {"action": "set_date_features", "features": features},
        description=(
            "Override which date-part features to extract. "
            "Options: dayofweek, month, day, dayofyear, quarter, hour, minute, year, weekday. "
            "Set to [] to reset to auto."
        ),
        parameters={
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Date features to extract",
                },
            },
            "required": ["features"],
        },
    )

    r.register_tool(
        name="set_ewm",
        category="features",
        function=lambda enabled: {"action": "set_ewm", "enabled": bool(enabled)},
        description="Enable or disable Exponentially Weighted Mean (EWM) feature",
        parameters={
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "description": "True to enable EWM"},
            },
            "required": ["enabled"],
        },
    )


# ------------------------------------------------------------------
# 5. MODEL CONFIGURATION
# ------------------------------------------------------------------
def _register_model_tools(r: ToolRegistry):

    r.register_tool(
        name="set_model",
        category="model",
        function=lambda model_type, hyperparameters=None: {
            "action": "set_model",
            "model_type": model_type,
            "hyperparameters": hyperparameters or {},
        },
        description=(
            "Set the forecasting model and (optionally) its hyperparameters. "
            "Models: auto, naive, linear, lightgbm, prophet, automl. "
            "For lightgbm you can set: n_estimators, learning_rate, num_leaves, max_depth, subsample, reg_alpha, reg_lambda. "
            "For prophet: changepoint_prior_scale, seasonality_prior_scale, seasonality_mode. "
            "Use model_type='auto' to let ModelSelectorAgent decide."
        ),
        parameters={
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "enum": ["auto", "naive", "linear", "lightgbm", "prophet", "automl"],
                    "description": "Model type",
                },
                "hyperparameters": {
                    "type": "object",
                    "description": "Model hyperparameters (key-value pairs). Leave empty for defaults.",
                },
            },
            "required": ["model_type"],
        },
    )

    r.register_tool(
        name="get_model_info",
        category="model",
        function=lambda: {"action": "get_model_info"},
        description=(
            "Get detailed info about the trained model: hyperparameters, feature importance, "
            "performance metrics, training mode. Use when user asks about the model."
        ),
        parameters={"type": "object", "properties": {}},
    )


# ------------------------------------------------------------------
# 6. PIPELINE / WORKFLOW CONTROL
# ------------------------------------------------------------------
def _register_pipeline_tools(r: ToolRegistry):

    r.register_tool(
        name="run_forecast",
        category="pipeline",
        function=lambda: {"action": "run_forecast", "config": {}},
        description=(
            "Run the full multi-agent forecast pipeline. "
            "Uses current configuration (datetime, target, grouping, horizon, gap, model). "
            "Agents automatically analyze data, engineer features, select model, train & evaluate."
        ),
        parameters={"type": "object", "properties": {}},
    )

    r.register_tool(
        name="rerun_pipeline_step",
        category="pipeline",
        function=lambda step: {"action": "rerun_step", "step": step},
        description=(
            "Re-run a specific pipeline step with current settings. "
            "Steps: analysis, features, model_selection, training, evaluation, forecast"
        ),
        parameters={
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "enum": [
                        "analysis",
                        "features",
                        "model_selection",
                        "training",
                        "evaluation",
                        "forecast",
                    ],
                    "description": "Pipeline step to re-execute",
                },
            },
            "required": ["step"],
        },
    )

    r.register_tool(
        name="show_pipeline_status",
        category="pipeline",
        function=lambda: {"action": "show_pipeline_status"},
        description="Show the status of the last pipeline run — steps, timing, decisions, health score",
        parameters={"type": "object", "properties": {}},
    )


# ------------------------------------------------------------------
# 7. DATA INSPECTION / EXPLORATION
# ------------------------------------------------------------------
def _register_inspect_tools(r: ToolRegistry):

    r.register_tool(
        name="describe_data",
        category="inspect",
        function=lambda columns=None: {"action": "describe_data", "columns": columns},
        description="Show statistical summary (mean, std, min, max, quartiles) of numeric columns",
        parameters={
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to describe (default: all numeric)",
                },
            },
        },
    )

    r.register_tool(
        name="show_dtypes",
        category="inspect",
        function=lambda: {"action": "show_dtypes"},
        description="Show data types and non-null counts for all columns",
        parameters={"type": "object", "properties": {}},
    )

    r.register_tool(
        name="show_head",
        category="inspect",
        function=lambda n=10: {"action": "show_head", "n": int(n)},
        description="Show the first N rows of the dataset",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 10, "description": "Number of rows"},
            },
        },
    )

    r.register_tool(
        name="value_counts",
        category="inspect",
        function=lambda column, top_n=20: {
            "action": "value_counts",
            "column": column,
            "top_n": int(top_n),
        },
        description="Show value counts (frequency distribution) for a column",
        parameters={
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "top_n": {"type": "integer", "default": 20},
            },
            "required": ["column"],
        },
    )

    r.register_tool(
        name="show_correlation",
        category="inspect",
        function=lambda columns=None, top_n=10: {
            "action": "show_correlation",
            "columns": columns,
            "top_n": int(top_n),
        },
        description="Show correlation matrix between numeric columns (or specific columns with the target)",
        parameters={
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to correlate (default: all numeric)",
                },
                "top_n": {"type": "integer", "default": 10},
            },
        },
    )


# ------------------------------------------------------------------
# 8. DATASETS / DATA SOURCES (catalog, switch, re-sync)
# ------------------------------------------------------------------
def _register_dataset_tools(r: ToolRegistry):
    r.register_tool(
        name="list_datasets",
        category="datasets",
        function=lambda: {"action": "list_datasets"},
        description=(
            "List all datasets available in the workspace (names, ids, source: PostgreSQL vs file, "
            "sync status, row counts). Use when the user asks what data exists or which sources are connected."
        ),
        parameters={"type": "object", "properties": {}},
    )

    r.register_tool(
        name="switch_dataset",
        category="datasets",
        function=lambda dataset_id: {
            "action": "switch_dataset",
            "dataset_id": str(dataset_id),
        },
        description=(
            "Switch the active dataset for this chat to the given dataset UUID. "
            "Inspect tools and data operations will then apply to that snapshot. "
            "Use list_datasets first if you need the id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "UUID of the dataset row (from list_datasets or CONTEXT)",
                },
            },
            "required": ["dataset_id"],
        },
    )

    r.register_tool(
        name="resync_dataset",
        category="datasets",
        function=lambda dataset_id=None: {
            "action": "resync_dataset",
            **({"dataset_id": str(dataset_id)} if dataset_id else {}),
        },
        description=(
            "Re-fetch data from the external PostgreSQL source into the stored snapshot (blob). "
            "Only works for datasets connected via Postgres. Omit dataset_id to refresh the active dataset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Optional UUID; defaults to the currently active dataset",
                },
            },
        },
    )
