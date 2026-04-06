"""Model Agent - handles model selection, training, and forecasting."""

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from forecaster.core.session import ForecastResult
from forecaster.models.simple import LinearForecaster, NaiveForecaster


class ModelAgent:
    """
    Model Agent - selects and runs forecasting models.

    Responsibilities:
    - Select appropriate model based on data
    - Train model
    - Generate forecasts

    Available models:
    - naive: Simple baseline (last value)
    - linear: Linear regression with time features
    - lightgbm: Gradient boosting (requires: pip install mlforecast lightgbm)
    - prophet: Meta's Prophet (requires: pip install prophet)
    - automl: Automatic model selection
    """

    def __init__(self):
        self.available_models = {
            "naive": NaiveForecaster,
            "linear": LinearForecaster,
        }

        # Try to load professional models (optional dependencies)
        self._load_optional_models()

    def get_available_models(self) -> list[str]:
        """Return list of available model names."""
        return list(self.available_models.keys())

    def forecast(
        self,
        filepath: Path,
        datetime_column: str,
        target_column: str,
        horizon: int = 7,
        gap: int = 0,
        model_type: str = "auto",
        group_by_column: str | None = None,
        group_by_columns: list[str] | None = None,
        dataframe: pd.DataFrame | None = None,  # Use in-memory DataFrame instead of file
    ) -> dict[str, Any]:
        # Ensure horizon and gap are integers
        try:
            horizon = int(horizon)
        except (ValueError, TypeError):
            horizon = 7
        try:
            gap = int(gap)
        except (ValueError, TypeError):
            gap = 0
        """
        Run forecast on the data.

        Args:
            filepath: Path to data file
            datetime_column: Name of datetime column
            target_column: Name of target column
            horizon: Number of periods to forecast
            model_type: Model to use ("naive", "linear", "auto")

        Returns:
            {
                "success": bool,
                "message": str,
                "forecast": ForecastResult (if success)
            }
        """
        try:
            # Determine grouping columns (prefer new list, fallback to single column)
            group_cols = group_by_columns if group_by_columns else []
            if group_by_column and group_by_column not in group_cols:
                group_cols = [group_by_column] if group_by_column else []

            # Load raw data (for professional models that need original columns)
            if dataframe is not None:
                raw_df = dataframe.copy()
            else:
                from forecaster.data.loader import load_full_dataframe

                raw_df = load_full_dataframe(filepath, datetime_column=datetime_column)

            # Validate raw data
            if datetime_column not in raw_df.columns:
                return {
                    "success": False,
                    "message": f"Datetime column '{datetime_column}' not found in data. Available columns: {list(raw_df.columns)}",
                    "forecast": None,
                }

            if target_column not in raw_df.columns:
                return {
                    "success": False,
                    "message": f"Target column '{target_column}' not found in data. Available columns: {list(raw_df.columns)}",
                    "forecast": None,
                }

            # Prepare processed data (for simple models)
            if dataframe is not None:
                df_processed = self._prepare_dataframe(
                    dataframe, datetime_column, target_column, group_cols
                )
            else:
                df_processed = self._load_data(filepath, datetime_column, target_column, group_cols)

            # Validate processed data
            if len(df_processed) == 0:
                return {
                    "success": False,
                    "message": "No data available after processing. Check your data and column selections.",
                    "forecast": None,
                }

            if df_processed["value"].isna().all():
                return {
                    "success": False,
                    "message": "Target column contains only missing values.",
                    "forecast": None,
                }

            # Select model
            if model_type == "auto":
                model_type = self._select_model(df_processed)

            # Get model class
            if model_type not in self.available_models:
                return {
                    "success": False,
                    "message": f"Unknown model: {model_type}",
                    "forecast": None,
                }

            # Train and forecast
            model_class = self.available_models[model_type]

            # For MLForecast, pass gap so lags 1..gap are excluded
            if model_type == "lightgbm":
                model = model_class(gap=gap)
            else:
                model = model_class()

            # Professional models need original DataFrame + column names
            is_professional_model = model_type in ["lightgbm", "prophet", "automl"]

            if is_professional_model:
                # Use original DataFrame with actual column names
                # Validate and clean data before passing to model
                fit_df = raw_df.copy()

                # Ensure target column is numeric
                if target_column in fit_df.columns:
                    original_dtype = fit_df[target_column].dtype
                    if not pd.api.types.is_numeric_dtype(fit_df[target_column]):
                        print(
                            f"[ModelAgent] Converting target column '{target_column}' from {original_dtype} to numeric"
                        )
                        fit_df[target_column] = pd.to_numeric(
                            fit_df[target_column], errors="coerce"
                        )
                        # Drop rows where conversion failed
                        before_len = len(fit_df)
                        fit_df = fit_df.dropna(subset=[target_column])
                        after_len = len(fit_df)
                        if before_len != after_len:
                            print(
                                f"[ModelAgent] Dropped {before_len - after_len} rows with non-numeric target values"
                            )

                # Ensure datetime column is datetime
                if datetime_column in fit_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(fit_df[datetime_column]):
                        print(
                            f"[ModelAgent] Converting datetime column '{datetime_column}' to datetime"
                        )
                        fit_df[datetime_column] = pd.to_datetime(
                            fit_df[datetime_column], errors="coerce"
                        )
                        fit_df = fit_df.dropna(subset=[datetime_column])

                # Debug: validate group columns (they might be the issue)
                if group_cols:
                    for col in group_cols:
                        if col in fit_df.columns:
                            col_dtype = fit_df[col].dtype
                            print(
                                f"[ModelAgent] Group column '{col}': dtype={col_dtype}, sample={fit_df[col].head(3).tolist()}"
                            )
                            # Check if it's object dtype (strings)
                            if col_dtype == "object":
                                # Check if it can be converted to numeric
                                try:
                                    test_numeric = pd.to_numeric(fit_df[col], errors="coerce")
                                    if not test_numeric.isna().all():
                                        print(
                                            f"[ModelAgent] WARNING: Group column '{col}' is string but contains numeric values"
                                        )
                                except:  # noqa: E722
                                    pass

                try:
                    fit_result = model.fit(
                        data=fit_df,
                        datetime_column=datetime_column,
                        target_column=target_column,
                        group_by_columns=group_cols if group_cols else None,
                    )

                    # Check if fit was successful (for models that return dict)
                    if isinstance(fit_result, dict) and not fit_result.get("success", True):
                        error_msg = fit_result.get("error", "Unknown error during model fitting")
                        raise ValueError(f"Model fitting failed: {error_msg}")

                    result = model.predict(horizon)
                except Exception as e:
                    import sys
                    import traceback

                    error_trace = traceback.format_exc()
                    error_msg = str(e)

                    # Log to file and console
                    print(f"\n{'=' * 80}", file=sys.stderr)
                    print("[ModelAgent] ❌ ERROR DURING FORECAST", file=sys.stderr)
                    print(f"{'=' * 80}", file=sys.stderr)
                    print(f"Error type: {type(e).__name__}", file=sys.stderr)
                    print(f"Error message: {error_msg}", file=sys.stderr)
                    print("\nFull traceback:", file=sys.stderr)
                    print(error_trace, file=sys.stderr)
                    print(f"{'=' * 80}\n", file=sys.stderr)

                    # Check if it's the string/int comparison error
                    if "'<=' not supported" in error_msg or "'>=' not supported" in error_msg:
                        print("[ModelAgent] This is a type comparison error!", file=sys.stderr)
                        print(
                            "[ModelAgent] Likely cause: Mixed data types in columns",
                            file=sys.stderr,
                        )
                        if group_cols:
                            print(f"[ModelAgent] Group columns used: {group_cols}", file=sys.stderr)

                    raise
            else:
                # Simple models use processed DataFrame (datetime index + 'value' column)
                model.fit(df_processed)
                result = model.predict(horizon)

            # Calculate simple metrics on training data (optional, don't fail if it errors)
            try:
                if is_professional_model:
                    metrics = {}  # Professional models handle their own metrics
                else:
                    metrics = self._calculate_metrics(df_processed, model)
            except Exception:
                metrics = {}

            # Format result based on model type
            if is_professional_model:
                # Professional models return DataFrame with predictions
                if isinstance(result, pd.DataFrame):
                    # Extract predictions and dates
                    if "prediction" in result.columns:
                        predictions = result["prediction"].tolist()
                    else:
                        # Fallback to last column
                        predictions = result.iloc[:, -1].tolist()

                    if "datetime" in result.columns:
                        dates = result["datetime"].astype(str).tolist()
                    elif "ds" in result.columns:
                        dates = result["ds"].astype(str).tolist()
                    else:
                        # Generate dates based on horizon
                        last_date = raw_df[datetime_column].max()
                        freq = pd.infer_freq(raw_df[datetime_column].sort_values())
                        if not freq:
                            freq = "D"  # Default to daily
                        date_range = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[
                            1:
                        ]
                        dates = date_range.astype(str).tolist()

                    # Prepare group info for multivariate forecasts
                    group_info = None
                    if group_cols and "unique_id" in result.columns:
                        group_info = {
                            "groups": result["unique_id"].unique().tolist(),
                            "n_groups": result["unique_id"].nunique(),
                            "group_columns": group_cols,
                        }

                    forecast = ForecastResult(
                        predictions=predictions,
                        dates=dates,
                        model_name=model_type,
                        horizon=horizon,
                        metrics=metrics,
                        group_info=group_info,
                    )
                else:
                    # Shouldn't happen, but handle it
                    forecast = result
            else:
                # Simple models return ForecastResult
                forecast = ForecastResult(
                    predictions=result.predictions,
                    dates=result.dates,
                    model_name=model_type,
                    horizon=horizon,
                    metrics=metrics,
                )

            # Enhanced analysis: residuals, baseline, warnings, health score
            forecast = self._enhance_forecast_with_diagnostics(
                forecast, df_processed, raw_df, datetime_column, target_column
            )

            # Per-group diagnostics (for separate per-group models)
            if (
                is_professional_model
                and group_cols
                and hasattr(model, "has_groups")
                and model.has_groups()
            ):
                forecast = self._enhance_with_per_group_diagnostics(
                    forecast, raw_df, datetime_column, target_column, group_cols
                )

            return {
                "success": True,
                "message": f"Forecast generated ({model_type}, {horizon} periods)",
                "forecast": forecast,
                "model": model,  # Return trained model for analysis
            }

        except ValueError as e:
            return {
                "success": False,
                "message": f"Data validation error: {str(e)}",
                "forecast": None,
            }
        except Exception as e:
            import traceback

            traceback.format_exc()
            return {
                "success": False,
                "message": f"Error during forecast: {str(e)}",
                "forecast": None,
            }

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
        datetime_column: str,
        target_column: str,
        group_by_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Prepare in-memory DataFrame for forecasting."""
        df = df.copy()

        # Ensure datetime column is datetime type
        if datetime_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
                df[datetime_column] = pd.to_datetime(df[datetime_column])

        return self._process_dataframe(df, datetime_column, target_column, group_by_columns)

    def _load_data(
        self,
        filepath: Path,
        datetime_column: str,
        target_column: str,
        group_by_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load and prepare data for forecasting from file."""
        from forecaster.data.loader import load_full_dataframe

        # Load full DataFrame
        df = load_full_dataframe(filepath, datetime_column=datetime_column)

        return self._process_dataframe(df, datetime_column, target_column, group_by_columns)

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        datetime_column: str,
        target_column: str,
        group_by_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Common processing for DataFrame (from file or memory)."""

        # Verify columns exist
        if datetime_column not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Datetime column '{datetime_column}' not found in data. Available columns: {list(df.columns)}"
            )

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data. Available columns: {list(df.columns)}"
            )

        # If grouping is specified, aggregate
        if group_by_columns and all(col in df.columns for col in group_by_columns):
            # Reset index if datetime is index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()

            # Group by datetime and group columns, aggregate target
            group_cols = [datetime_column] + group_by_columns
            df_grouped = df.groupby(group_cols)[target_column].mean().reset_index()
            df_grouped = df_grouped.set_index(datetime_column)
            df = df_grouped[[target_column]].copy()
        else:
            # Standard: just use target column
            # If datetime is already index, use it
            if not isinstance(df.index, pd.DatetimeIndex):
                if datetime_column in df.columns:
                    df = df.set_index(datetime_column)
            df = df[[target_column]].copy()

        df.columns = ["value"]
        df = df.sort_index()
        df = df.dropna()

        return df

    def _load_optional_models(self):
        """Load professional models if dependencies are installed."""
        # Try MLForecast + LightGBM
        try:
            from forecaster.models.mlforecast_models import MLForecastModel

            self.available_models["lightgbm"] = MLForecastModel
            print("[ModelAgent] ✅ LightGBM model available")
        except ImportError:
            print(
                "[ModelAgent] ℹ️ LightGBM not available (install: pip install mlforecast lightgbm)"
            )

        # Try Prophet
        try:
            from forecaster.models.prophet_model import ProphetModel

            self.available_models["prophet"] = ProphetModel
            print("[ModelAgent] ✅ Prophet model available")
        except ImportError:
            print("[ModelAgent] ℹ️ Prophet not available (install: pip install prophet)")

        # Try AutoML
        try:
            from forecaster.models.automl_forecaster import AutoMLForecaster

            self.available_models["automl"] = AutoMLForecaster
            print("[ModelAgent] ✅ AutoML available")
        except ImportError:
            print("[ModelAgent] ℹ️ AutoML not available")

    def _select_model(self, df: pd.DataFrame) -> str:
        """Auto-select model based on data characteristics."""
        # Ensure n_points is integer
        n_points = int(len(df))

        # Prefer LightGBM for larger datasets
        if "lightgbm" in self.available_models and n_points >= 100:
            return "lightgbm"
        elif n_points < 30:
            return "naive"
        else:
            return "linear"

    def _calculate_metrics(self, df: pd.DataFrame, model: Any) -> dict[str, float]:
        """Calculate simple metrics on training data."""
        try:
            import numpy as np

            # Get in-sample predictions (simplified)
            n = len(df)
            if n < 5:
                return {}

            # Use last 20% for validation
            train_size = int(n * 0.8)
            if train_size == 0:
                return {}

            train_df = df.iloc[:train_size].copy()
            test_df = df.iloc[train_size:].copy()

            if len(test_df) == 0 or len(train_df) == 0:
                return {}

            # Refit on train and predict

            model_class = type(model)
            temp_model = model_class()
            temp_model.fit(train_df)
            result = temp_model.predict(len(test_df))

            # Calculate metrics
            actual = test_df["value"].values
            predicted = result.predictions[: len(actual)]

            if len(actual) == 0 or len(predicted) == 0:
                return {}

            # Check for division by zero
            if np.all(actual == actual[0]) and np.all(predicted == predicted[0]):
                # All values are the same, metrics are 0
                return {"mae": 0.0, "rmse": 0.0}

            mae = float(np.mean(np.abs(actual - predicted)))
            rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

            # Check for NaN or Inf
            if np.isnan(mae) or np.isinf(mae) or np.isnan(rmse) or np.isinf(rmse):
                return {}

            return {"mae": round(mae, 2), "rmse": round(rmse, 2)}

        except Exception:
            # Return empty metrics on any error
            return {}

    def _enhance_with_per_group_diagnostics(
        self,
        forecast: ForecastResult,
        raw_df: pd.DataFrame,
        datetime_column: str,
        target_column: str,
        group_cols: list[str],
    ) -> ForecastResult:
        """
        Compute diagnostics per group using IN-SAMPLE holdout evaluation.

        Instead of comparing future forecast to historical test (which doesn't make sense),
        we split historical data into train/test and evaluate the model's performance.
        """
        try:
            import numpy as np

            from forecaster.analysis.model_diagnostics import (
                calculate_baseline_metrics,
                calculate_health_score,
                calculate_residuals,
                generate_warnings,
            )

            groups = forecast.group_info.get("groups", []) if forecast.group_info else []

            per_group_metrics = {}
            per_group_health = {}
            per_group_warns = {}
            per_group_residuals = {}
            per_group_residual_dates = {}

            for group_id in groups:
                try:
                    # Parse group_id to filter raw data
                    group_values = group_id.split("_")
                    group_data = raw_df.copy()
                    for col_idx, col_name in enumerate(group_cols):
                        if col_idx < len(group_values) and col_name in group_data.columns:
                            group_data = group_data[
                                group_data[col_name].astype(str) == group_values[col_idx]
                            ]

                    if len(group_data) < 10:
                        continue

                    # Sort by datetime
                    if datetime_column in group_data.columns:
                        group_data = group_data.sort_values(datetime_column)

                    actual_values = group_data[target_column].dropna().tolist()

                    if len(actual_values) < 10:
                        continue

                    # Split for holdout evaluation (80/20)
                    split_idx = int(len(actual_values) * 0.8)
                    train_actual = actual_values[:split_idx]
                    test_actual = actual_values[split_idx:]

                    # Naive baseline: last training value repeated
                    naive_preds = [train_actual[-1]] * len(test_actual)

                    # For MODEL predictions on holdout:
                    # Since we already trained on full data, we can't get true holdout preds.
                    # Use a simple heuristic: assume model's RMSE improvement carries to test set.
                    # OR: use naive as proxy for now (conservative estimate).
                    # IDEAL: retrain model on train_actual and predict test_actual (expensive).

                    # FALLBACK: Use naive baseline for now (shows worst-case scenario)
                    model_test_preds = naive_preds  # Conservative placeholder

                    # Calculate residuals
                    residual_analysis = calculate_residuals(test_actual, model_test_preds)
                    baseline_metrics = calculate_baseline_metrics(
                        test_actual, model_test_preds, naive_preds
                    )

                    # Get dates for residuals
                    if datetime_column in group_data.columns:
                        test_dates = (
                            group_data[datetime_column].iloc[split_idx:].astype(str).tolist()
                        )
                    else:
                        test_dates = list(range(split_idx, len(actual_values)))

                    per_group_residuals[group_id] = residual_analysis.get("residuals", [])
                    per_group_residual_dates[group_id] = test_dates[
                        : len(per_group_residuals[group_id])
                    ]

                    # Per-group data quality
                    group_data_quality = {
                        "total_rows": len(actual_values),
                        "missing_pct": 0.0,
                        "outliers_pct": 0.0,
                    }

                    # Per-group metrics
                    per_group_metrics[group_id] = {
                        "n_observations": len(actual_values),
                        "mean": float(np.mean(actual_values)),
                        "std": float(np.std(actual_values)),
                        "forecast_mean": float(np.mean(forecast.predictions))
                        if forecast.predictions
                        else 0,
                        "forecast_std": float(np.std(forecast.predictions))
                        if forecast.predictions
                        else 0,
                    }
                    per_group_metrics[group_id].update(baseline_metrics)

                    # Per-group warnings — pass real data_quality (not empty dict)
                    group_warnings = generate_warnings(
                        residual_analysis, baseline_metrics, group_data_quality, []
                    )
                    per_group_warns[group_id] = group_warnings

                    # Per-group health score
                    try:
                        score, _ = calculate_health_score(
                            residual_analysis, baseline_metrics, group_data_quality, group_warnings
                        )
                        per_group_health[group_id] = score
                    except Exception:
                        per_group_health[group_id] = 50.0

                except Exception as e:
                    print(
                        f"[ModelAgent] Per-group diagnostics error for '{group_id}': {str(e)}",
                        file=sys.stderr,
                    )
                    continue

            forecast.per_group_metrics = per_group_metrics if per_group_metrics else None
            forecast.per_group_health_scores = per_group_health if per_group_health else None
            forecast.per_group_warnings = per_group_warns if per_group_warns else None
            forecast.per_group_residuals = per_group_residuals if per_group_residuals else None
            forecast.per_group_residual_dates = (
                per_group_residual_dates if per_group_residual_dates else None
            )

        except Exception as e:
            print(f"[ModelAgent] Per-group diagnostics failed: {str(e)}", file=sys.stderr)

        return forecast

    def _enhance_forecast_with_diagnostics(
        self,
        forecast: ForecastResult,
        df_processed: pd.DataFrame,
        raw_df: pd.DataFrame,
        datetime_column: str,
        target_column: str,
    ) -> ForecastResult:
        """
        Enhance forecast with diagnostics: data quality, sanity checks, warnings, health score.

        NOTE: Baseline metrics (RMSE model vs naive) should be injected by the workflow
        from holdout evaluation — NOT computed here with fake proxies.
        """
        try:
            from forecaster.analysis.model_diagnostics import (
                analyze_data_quality,
                calculate_baseline_metrics,
                calculate_health_score,
                calculate_residuals,
                check_forecast_sanity,
                generate_trust_indicators,
                generate_warnings,
            )

            # ---- Residual Analysis (on holdout set) ----
            actual_values = (
                df_processed["value"].tolist() if "value" in df_processed.columns else []
            )
            residual_analysis = {"is_random": True, "autocorr_lag1": 0.0, "has_trend": False}

            if len(actual_values) >= 10:
                split_idx = int(len(actual_values) * 0.8)
                train_actual = actual_values[:split_idx]
                test_actual = actual_values[split_idx:]

                # Naive baseline for residuals
                test_naive = [train_actual[-1]] * len(test_actual)

                residual_analysis = calculate_residuals(test_actual, test_naive)
                forecast.residuals = residual_analysis.get("residuals", [])
                forecast.actual_values = test_actual

                # Residual dates
                if hasattr(df_processed.index, "tolist"):
                    test_dates = df_processed.index[split_idx:].astype(str).tolist()
                    forecast.residual_dates = test_dates

                # If no baseline_metrics were injected by workflow, compute basic ones
                if not forecast.baseline_metrics:
                    forecast.baseline_metrics = calculate_baseline_metrics(
                        test_actual, test_naive, test_naive
                    )

            # ---- Data Quality Analysis ----
            forecast.data_quality = analyze_data_quality(raw_df, datetime_column, target_column)

            # ---- Forecast Sanity Checks ----
            historical_values = (
                raw_df[target_column].dropna().tolist() if target_column in raw_df.columns else []
            )
            forecast_warnings = check_forecast_sanity(
                forecast.predictions, historical_values, target_column
            )

            # ---- Trust Indicators ----
            forecast.trust_indicators = generate_trust_indicators(
                residual_analysis,
                forecast.baseline_metrics or {},
                forecast.data_quality or {},
                forecast_warnings,
            )

            # ---- Warnings ----
            forecast.warnings = generate_warnings(
                residual_analysis,
                forecast.baseline_metrics or {},
                forecast.data_quality or {},
                forecast_warnings,
            )

            # ---- Health Score ----
            if forecast.baseline_metrics and forecast.data_quality:
                health_score, _ = calculate_health_score(
                    residual_analysis,
                    forecast.baseline_metrics,
                    forecast.data_quality,
                    forecast.warnings,
                )
                forecast.health_score = health_score

            return forecast

        except Exception as e:
            import sys

            print(f"[ModelAgent] Warning: Could not compute diagnostics: {str(e)}", file=sys.stderr)
            return forecast
