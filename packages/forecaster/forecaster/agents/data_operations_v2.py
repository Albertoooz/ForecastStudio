"""Data transformation operations - works with DataFrames in memory (no file I/O)."""

from typing import Any

import pandas as pd

from forecaster.utils.observability import langfuse_observation


class DataOperations:
    """Handles generic data transformation operations on DataFrames."""

    def __init__(self):
        self.available_operations = {
            "combine_datetime": self._combine_datetime,
            "add_column": self._add_column,
            "normalize": self._normalize,
            "aggregate": self._aggregate,
            "resample": self._resample,
            "drop_column": self._drop_column,
            "rename_column": self._rename_column,
            "filter": self._filter,
            "sort": self._sort,
            "fill_missing": self._fill_missing,
            "drop_missing": self._drop_missing,
            "filter_date_range": self._filter_date_range,
        }

    def execute_operation(
        self,
        df: pd.DataFrame,
        operation: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a data transformation operation on DataFrame in memory.

        Args:
            df: DataFrame to transform
            operation: Name of the operation to execute
            parameters: Operation-specific parameters

        Returns:
            {
                "success": bool,
                "message": str,
                "dataframe": pd.DataFrame (if success) - transformed DataFrame
            }
        """
        with langfuse_observation(
            as_type="span",
            name=f"data_operation.{operation}",
            input={
                "operation": operation,
                "parameters": parameters,
                "rows": len(df),
                "columns": list(df.columns),
            },
        ) as obs:
            if operation not in self.available_operations:
                result = {
                    "success": False,
                    "message": f"Unknown operation: {operation}. Available: {', '.join(self.available_operations.keys())}",
                    "dataframe": None,
                }
                if obs is not None:
                    obs.update(output={"success": False, "message": result["message"]})
                return result

            try:
                # Validate INPUT columns exist before operation (skip output columns)
                output_column_names = [
                    "output_column",
                    "new_name",
                    "column_name",
                ]  # These are outputs, not inputs
                for param_name, col_name in parameters.items():
                    if (
                        "column" in param_name
                        and param_name not in output_column_names
                        and isinstance(col_name, str)
                    ):
                        if col_name not in df.columns:
                            raise ValueError(
                                f"Column '{col_name}' required for operation '{operation}' not found in data."
                            )

                # Execute operation on DataFrame
                op_func = self.available_operations[operation]
                result_df = op_func(
                    df.copy(), parameters
                )  # Work on copy to avoid modifying original

                # Build detailed message
                message = f"Executed operation '{operation}'"
                if operation == "combine_datetime":
                    output_col = parameters.get("output_column", "datetime")
                    message += f". Created new datetime column '{output_col}'"
                elif operation == "add_column":
                    col_name = parameters.get("column_name", "")
                    if col_name:
                        message += f". Created new column '{col_name}'"

                message += f". DataFrame now has {len(result_df.columns)} columns."

                result = {
                    "success": True,
                    "message": message,
                    "dataframe": result_df,
                }
                if obs is not None:
                    obs.update(
                        output={
                            "success": True,
                            "message": message,
                            "rows_after": len(result_df),
                            "columns_after": list(result_df.columns),
                        }
                    )
                return result

            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                print(f"[DataOperations] Error: {str(e)}")
                print(f"[DataOperations] Traceback: {error_trace}")
                result = {
                    "success": False,
                    "message": f"Error during operation '{operation}': {str(e)}",
                    "dataframe": None,
                }
                if obs is not None:
                    obs.update(output={"success": False, "error": str(e)})
                return result

    # Operation implementations

    def _combine_datetime(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Combine date and time columns."""
        date_col = params.get("date_column")
        time_col = params.get("time_column")
        output_col = params.get("output_column", "datetime")

        if not date_col or not time_col:
            raise ValueError("date_column and time_column are required")

        # Check if columns exist
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in data")
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in data")

        # Combine date and time
        df[output_col] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))

        return df

    def _add_column(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Add a new calculated column."""
        column_name = params.get("column_name")
        expression = params.get("expression")

        if not column_name or not expression:
            raise ValueError("column_name and expression are required")

        # Evaluate expression - allow full pandas operations including groupby
        try:
            import numpy as np
            import pandas as pd

            # Provide df, pd, np in eval context for complex operations
            result = eval(expression, {"df": df, "pd": pd, "np": np})
            df[column_name] = result
        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            raise ValueError(  # noqa: B904
                f"Could not evaluate expression: {expression}\n"
                f"Error: {str(e)}\n"
                f"Details: {error_detail}"
            )

        return df

    def _normalize(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Normalize a column."""
        column = params.get("column")
        method = params.get("method", "min_max")
        output_column = params.get("output_column", f"{column}_normalized")

        if not column:
            raise ValueError("column is required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        if method == "min_max":
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val == min_val:
                df[output_column] = 0
            else:
                df[output_column] = (df[column] - min_val) / (max_val - min_val)
        elif method == "z_score":
            mean = df[column].mean()
            std = df[column].std()
            if std == 0:
                df[output_column] = 0
            else:
                df[output_column] = (df[column] - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return df

    def _aggregate(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Aggregate data by group."""
        group_by = params.get("group_by")
        agg_column = params.get("agg_column")
        agg_function = params.get("agg_function", "mean")

        if not group_by or not agg_column:
            raise ValueError("group_by and agg_column are required")

        result = df.groupby(group_by)[agg_column].agg(agg_function).reset_index()
        return result

    def _resample(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Resample time series to different frequency."""
        datetime_column = params.get("datetime_column")
        value_column = params.get("value_column")
        frequency = params.get("frequency", "D")
        agg_function = params.get("agg_function", "mean")

        if not datetime_column or not value_column:
            raise ValueError("datetime_column and value_column are required")

        # Set datetime as index
        df_copy = df.copy()
        df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])
        df_copy = df_copy.set_index(datetime_column)

        # Resample
        resampled = df_copy[value_column].resample(frequency).agg(agg_function)
        return resampled.reset_index()

    def _drop_column(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Remove a column."""
        column = params.get("column")

        if not column:
            raise ValueError("column is required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        return df.drop(columns=[column])

    def _rename_column(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Rename a column."""
        old_name = params.get("old_name")
        new_name = params.get("new_name")

        if not old_name or not new_name:
            raise ValueError("old_name and new_name are required")

        if old_name not in df.columns:
            raise ValueError(f"Column '{old_name}' not found")

        return df.rename(columns={old_name: new_name})

    def _filter(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Filter rows based on condition."""
        column = params.get("column")
        condition = params.get("condition")
        value = params.get("value")

        if not column or not condition:
            raise ValueError("column and condition are required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        if condition == ">":
            return df[df[column] > value]
        elif condition == "<":
            return df[df[column] < value]
        elif condition == "==":
            return df[df[column] == value]
        elif condition == ">=":
            return df[df[column] >= value]
        elif condition == "<=":
            return df[df[column] <= value]
        else:
            raise ValueError(f"Unknown condition: {condition}")

    def _sort(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Sort by column."""
        column = params.get("column")
        ascending = params.get("ascending", True)

        if not column:
            raise ValueError("column is required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        return df.sort_values(by=column, ascending=ascending)

    def _fill_missing(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Fill missing values with various strategies."""
        column = params.get("column", "__all__")
        method = params.get("method", "ffill")
        value = params.get("value")

        if column == "__all__":
            cols = df.columns.tolist()
        else:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            cols = [column]

        for col in cols:
            if method == "ffill":
                df[col] = df[col].ffill()
            elif method == "bfill":
                df[col] = df[col].bfill()
            elif method == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
            elif method == "zero":
                df[col] = df[col].fillna(0)
            elif method == "interpolate":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].interpolate(method="linear")
            elif method == "value":
                if value is not None:
                    df[col] = df[col].fillna(value)
                else:
                    raise ValueError("method='value' requires 'value' parameter")
            else:
                raise ValueError(f"Unknown fill method: {method}")

        filled_cols = ", ".join(cols[:5])
        if len(cols) > 5:
            filled_cols += f" (+{len(cols) - 5} more)"
        return df

    def _drop_missing(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Drop rows with missing values."""
        column = params.get("column")
        threshold = params.get("threshold")

        before = len(df)
        if column:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            df = df.dropna(subset=[column])
        elif threshold is not None:
            df = df.dropna(thresh=int(threshold))
        else:
            df = df.dropna()

        dropped = before - len(df)
        print(f"[DataOperations] Dropped {dropped} rows with missing values")
        return df

    def _filter_date_range(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Filter data by date range."""
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        # Find datetime column
        dt_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_col = col
                break

        if dt_col is None:
            from forecaster.utils.streamlit_optional import get_session_state

            legacy_session = get_session_state().get("session")
            if (
                legacy_session
                and legacy_session.data_info
                and legacy_session.data_info.datetime_column
            ):
                dt_col = legacy_session.data_info.datetime_column
                if dt_col in df.columns:
                    df[dt_col] = pd.to_datetime(df[dt_col])

        if dt_col is None:
            raise ValueError("No datetime column found. Set datetime column first.")

        before = len(df)
        if start_date:
            df = df[df[dt_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[dt_col] <= pd.to_datetime(end_date)]

        filtered = before - len(df)
        print(f"[DataOperations] Filtered {filtered} rows by date range")
        return df

    def get_available_operations(self) -> dict[str, dict]:
        """Get list of available operations with descriptions."""
        return {
            "combine_datetime": {
                "description": "Combines date and time columns into a datetime column",
                "parameters": {
                    "date_column": "str - name of the date column",
                    "time_column": "str - name of the time column",
                    "output_column": "str - name of the new column (default: datetime)",
                },
            },
            "add_column": {
                "description": "Adds a new calculated column",
                "parameters": {
                    "column_name": "str - name of the new column",
                    "expression": "str - pandas expression (e.g., 'col1 + col2')",
                },
            },
            "normalize": {
                "description": "Normalizes a column",
                "parameters": {
                    "column": "str - name of the column to normalize",
                    "method": "str - 'min_max' or 'z_score' (default: min_max)",
                    "output_column": "str - name of the new column",
                },
            },
            "aggregate": {
                "description": "Aggregates data by group",
                "parameters": {
                    "group_by": "str - column to group by",
                    "agg_column": "str - column to aggregate",
                    "agg_function": "str - aggregation function (mean/sum/count)",
                },
            },
            "resample": {
                "description": "Changes time series frequency",
                "parameters": {
                    "datetime_column": "str - name of the datetime column",
                    "value_column": "str - name of the value column",
                    "frequency": "str - D/H/W/M (default: D)",
                    "agg_function": "str - aggregation function (default: mean)",
                },
            },
            "drop_column": {
                "description": "Removes a column",
                "parameters": {"column": "str - name of the column to remove"},
            },
            "rename_column": {
                "description": "Renames a column",
                "parameters": {
                    "old_name": "str - current name",
                    "new_name": "str - new name",
                },
            },
            "filter": {
                "description": "Filters rows based on condition",
                "parameters": {
                    "column": "str - column name",
                    "condition": "str - >/</==/>=/<=",
                    "value": "value to compare",
                },
            },
            "sort": {
                "description": "Sorts by column",
                "parameters": {
                    "column": "str - column to sort by",
                    "ascending": "bool - sort order (default: true)",
                },
            },
        }
