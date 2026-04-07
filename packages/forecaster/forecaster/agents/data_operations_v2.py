"""Data transformation operations — Polars DataFrames in memory (no file I/O)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from forecaster.utils.observability import langfuse_observation


def _pd_freq_to_polars(frequency: str) -> str:
    f = (frequency or "D").strip()
    u = f.upper()
    mapping = {
        "D": "1d",
        "1D": "1d",
        "H": "1h",
        "1H": "1h",
        "W": "1w",
        "1W": "1w",
        "M": "1mo",
        "MS": "1mo",
        "T": "1m",
        "MIN": "1m",
    }
    return mapping.get(u, f if any(c.isdigit() for c in f) else "1d")


def _agg_expr(agg_column: str, agg_function: str) -> pl.Expr:
    c = pl.col(agg_column)
    fn = (agg_function or "mean").lower()
    if fn == "mean":
        return c.mean()
    if fn == "sum":
        return c.sum()
    if fn == "count":
        return c.count()
    if fn == "min":
        return c.min()
    if fn == "max":
        return c.max()
    raise ValueError(f"Unknown aggregation function: {agg_function}")


def _parse_date_bound(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


class DataOperations:
    """Handles generic data transformation operations on Polars DataFrames."""

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
        df: pl.DataFrame,
        operation: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        with langfuse_observation(
            as_type="span",
            name=f"data_operation.{operation}",
            input={
                "operation": operation,
                "parameters": parameters,
                "rows": df.height,
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
                output_column_names = ("output_column", "new_name", "column_name")
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

                op_func = self.available_operations[operation]
                result_df = op_func(df.clone(), parameters)

                message = f"Executed operation '{operation}'"
                if operation == "combine_datetime":
                    output_col = parameters.get("output_column", "datetime")
                    message += f". Created new datetime column '{output_col}'"
                elif operation == "add_column":
                    col_name = parameters.get("column_name", "")
                    if col_name:
                        message += f". Created new column '{col_name}'"

                message += f". DataFrame now has {result_df.width} columns."

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
                            "rows_after": result_df.height,
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

    def _combine_datetime(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        date_col = params.get("date_column")
        time_col = params.get("time_column")
        output_col = params.get("output_column", "datetime")

        if not date_col or not time_col:
            raise ValueError("date_column and time_column are required")

        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in data")
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in data")

        combined = (
            pl.col(date_col).cast(pl.Utf8) + pl.lit(" ") + pl.col(time_col).cast(pl.Utf8)
        ).str.to_datetime(strict=False)
        return df.with_columns(combined.alias(output_col))

    def _add_column(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        column_name = params.get("column_name")
        expression = params.get("expression")

        if not column_name or not expression:
            raise ValueError("column_name and expression are required")

        try:
            import numpy as np

            result = eval(
                expression,
                {"__builtins__": {}},
                {"pl": pl, "df": df, "np": np},
            )
        except Exception as e:
            import traceback

            raise ValueError(
                f"Could not evaluate expression: {expression}\n"
                f"Error: {str(e)}\n"
                f"Details: {traceback.format_exc()}"
            ) from e

        if isinstance(result, pl.Expr):
            return df.with_columns(result.alias(column_name))
        raise ValueError(
            "Expression must evaluate to a Polars expression (e.g. pl.col('a') + pl.col('b'))"
        )

    def _normalize(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        column = params.get("column")
        method = params.get("method", "min_max")
        output_column = params.get("output_column", f"{column}_normalized")

        if not column:
            raise ValueError("column is required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        c = pl.col(column)
        if method == "min_max":
            min_val = df[column].min()
            max_val = df[column].max()
            if min_val is None or max_val is None or min_val == max_val:
                return df.with_columns(pl.lit(0.0).alias(output_column))
            return df.with_columns(((c - min_val) / (max_val - min_val)).alias(output_column))
        if method == "z_score":
            mean = df[column].mean()
            std = df[column].std()
            if std is None or std == 0:
                return df.with_columns(pl.lit(0.0).alias(output_column))
            return df.with_columns(((c - mean) / std).alias(output_column))
        raise ValueError(f"Unknown normalization method: {method}")

    def _aggregate(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        group_by = params.get("group_by")
        agg_column = params.get("agg_column")
        agg_function = params.get("agg_function", "mean")

        if not group_by or not agg_column:
            raise ValueError("group_by and agg_column are required")

        return df.group_by(group_by).agg(_agg_expr(agg_column, agg_function))

    def _resample(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        datetime_column = params.get("datetime_column")
        value_column = params.get("value_column")
        frequency = params.get("frequency", "D")
        agg_function = params.get("agg_function", "mean")

        if not datetime_column or not value_column:
            raise ValueError("datetime_column and value_column are required")

        every = _pd_freq_to_polars(frequency)
        sorted_df = df.sort(datetime_column)
        return sorted_df.group_by_dynamic(datetime_column, every=every).agg(
            _agg_expr(value_column, agg_function)
        )

    def _drop_column(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        column = params.get("column")

        if not column:
            raise ValueError("column is required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        return df.drop(column)

    def _rename_column(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        old_name = params.get("old_name")
        new_name = params.get("new_name")

        if not old_name or not new_name:
            raise ValueError("old_name and new_name are required")

        if old_name not in df.columns:
            raise ValueError(f"Column '{old_name}' not found")

        return df.rename({old_name: new_name})

    def _filter(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        column = params.get("column")
        condition = params.get("condition")
        value = params.get("value")

        if not column or not condition:
            raise ValueError("column and condition are required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        c = pl.col(column)
        cond = condition.strip() if isinstance(condition, str) else condition
        if cond == ">":
            return df.filter(c > value)
        if cond == "<":
            return df.filter(c < value)
        if cond == "==":
            return df.filter(c == value)
        if cond == "!=":
            return df.filter(c != value)
        if cond == ">=":
            return df.filter(c >= value)
        if cond == "<=":
            return df.filter(c <= value)
        raise ValueError(f"Unknown condition: {condition}")

    def _sort(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        column = params.get("column")
        ascending = params.get("ascending", True)

        if not column:
            raise ValueError("column is required")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        return df.sort(column, descending=not ascending)

    def _fill_missing(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        column = params.get("column", "__all__")
        method = params.get("method", "ffill")
        value = params.get("value")

        if column == "__all__":
            cols = list(df.columns)
        else:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            cols = [column]

        out = df
        for col in cols:
            c = pl.col(col)
            if method == "ffill":
                out = out.with_columns(c.forward_fill())
            elif method == "bfill":
                out = out.with_columns(c.backward_fill())
            elif method == "mean":
                if out[col].dtype.is_numeric():
                    m = out[col].mean()
                    out = out.with_columns(c.fill_null(m))
            elif method == "median":
                if out[col].dtype.is_numeric():
                    med = out[col].median()
                    out = out.with_columns(c.fill_null(med))
            elif method == "zero":
                out = out.with_columns(c.fill_null(0))
            elif method == "interpolate":
                if out[col].dtype.is_numeric():
                    out = out.with_columns(c.interpolate())
            elif method == "value":
                if value is not None:
                    out = out.with_columns(c.fill_null(value))
                else:
                    raise ValueError("method='value' requires 'value' parameter")
            else:
                raise ValueError(f"Unknown fill method: {method}")

        return out

    def _drop_missing(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        column = params.get("column")
        threshold = params.get("threshold")

        before = df.height
        if column:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            out = df.drop_nulls(subset=[column])
        elif threshold is not None:
            n = int(threshold)
            non_null = pl.sum_horizontal([pl.col(c).is_not_null() for c in df.columns])
            out = df.filter(non_null >= n)
        else:
            out = df.drop_nulls()

        dropped = before - out.height
        print(f"[DataOperations] Dropped {dropped} rows with missing values")
        return out

    def _filter_date_range(self, df: pl.DataFrame, params: dict[str, Any]) -> pl.DataFrame:
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        dt_col = None
        for col in df.columns:
            if df[col].dtype.is_temporal():
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
                cand = legacy_session.data_info.datetime_column
                if cand in df.columns:
                    dt_col = cand

        if dt_col is None:
            raise ValueError("No datetime column found. Set datetime column first.")

        tsc = pl.col(dt_col).cast(pl.Datetime, strict=False)
        cond = pl.lit(True)
        if start_date is not None:
            cond = cond & (tsc >= pl.lit(_parse_date_bound(start_date)))
        if end_date is not None:
            cond = cond & (tsc <= pl.lit(_parse_date_bound(end_date)))

        before = df.height
        out = df.filter(cond)
        filtered = before - out.height
        print(f"[DataOperations] Filtered {filtered} rows by date range")
        return out

    def get_available_operations(self) -> dict[str, dict]:
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
                    "expression": "str - Polars expression (e.g. pl.col('a') + pl.col('b'))",
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
                    "condition": "str - >/</==/!=/>=/<=",
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
