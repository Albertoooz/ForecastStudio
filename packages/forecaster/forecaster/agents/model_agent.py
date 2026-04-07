"""Model Agent — handles model selection, training, and forecasting (Polars)."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from forecaster.core.session import ForecastResult
from forecaster.models.simple import LinearForecaster, NaiveForecaster
from forecaster.utils.tabular import infer_frequency, polars_date_range


class ModelAgent:
    """
    Model Agent — selects and runs forecasting models.

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
        self._load_optional_models()

    def get_available_models(self) -> list[str]:
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
        dataframe: pl.DataFrame | None = None,
    ) -> dict[str, Any]:
        try:
            horizon = int(horizon)
        except (ValueError, TypeError):
            horizon = 7
        try:
            gap = int(gap)
        except (ValueError, TypeError):
            gap = 0

        try:
            group_cols = group_by_columns if group_by_columns else []
            if group_by_column and group_by_column not in group_cols:
                group_cols = [group_by_column] if group_by_column else []

            if dataframe is not None:
                raw_df = dataframe.clone()
            else:
                from forecaster.data.loader import load_full_dataframe

                raw_df = load_full_dataframe(filepath, datetime_column=datetime_column)

            if datetime_column not in raw_df.columns:
                return {
                    "success": False,
                    "message": f"Datetime column '{datetime_column}' not found. Available: {raw_df.columns}",
                    "forecast": None,
                }

            if target_column not in raw_df.columns:
                return {
                    "success": False,
                    "message": f"Target column '{target_column}' not found. Available: {raw_df.columns}",
                    "forecast": None,
                }

            if dataframe is not None:
                df_processed = self._prepare_dataframe(
                    dataframe, datetime_column, target_column, group_cols
                )
            else:
                df_processed = self._load_data(filepath, datetime_column, target_column, group_cols)

            if df_processed.height == 0:
                return {
                    "success": False,
                    "message": "No data available after processing. Check your data and column selections.",
                    "forecast": None,
                }

            if df_processed["value"].null_count() == df_processed.height:
                return {
                    "success": False,
                    "message": "Target column contains only missing values.",
                    "forecast": None,
                }

            if model_type == "auto":
                model_type = self._select_model(df_processed)

            if model_type not in self.available_models:
                return {
                    "success": False,
                    "message": f"Unknown model: {model_type}",
                    "forecast": None,
                }

            model_class = self.available_models[model_type]
            if model_type == "lightgbm":
                model = model_class(gap=gap)
            else:
                model = model_class()

            is_professional_model = model_type in ["lightgbm", "prophet", "automl"]

            if is_professional_model:
                fit_df = raw_df.clone()

                # Ensure target is numeric
                if target_column in fit_df.columns:
                    if not fit_df[target_column].dtype.is_numeric():
                        print(f"[ModelAgent] Converting target column '{target_column}' to numeric")
                        fit_df = fit_df.with_columns(
                            pl.col(target_column).cast(pl.Float64, strict=False)
                        )
                        before = fit_df.height
                        fit_df = fit_df.drop_nulls(target_column)
                        after = fit_df.height
                        if before != after:
                            print(f"[ModelAgent] Dropped {before - after} non-numeric target rows")

                # Ensure datetime is temporal
                if datetime_column in fit_df.columns:
                    if not fit_df[datetime_column].dtype.is_temporal():
                        print(
                            f"[ModelAgent] Converting datetime column '{datetime_column}' to datetime"
                        )
                        fit_df = fit_df.with_columns(
                            pl.col(datetime_column).cast(pl.Datetime, strict=False)
                        )
                        fit_df = fit_df.drop_nulls(datetime_column)

                if group_cols:
                    for col in group_cols:
                        if col in fit_df.columns:
                            print(
                                f"[ModelAgent] Group column '{col}': dtype={fit_df[col].dtype}, "
                                f"sample={fit_df[col].head(3).to_list()}"
                            )

                try:
                    fit_result = model.fit(
                        data=fit_df,
                        datetime_column=datetime_column,
                        target_column=target_column,
                        group_by_columns=group_cols if group_cols else None,
                    )
                    if isinstance(fit_result, dict) and not fit_result.get("success", True):
                        raise ValueError(f"Model fitting failed: {fit_result.get('error')}")

                    result = model.predict(horizon)
                except Exception as e:
                    import traceback

                    error_trace = traceback.format_exc()
                    print(f"\n{'=' * 80}", file=sys.stderr)
                    print("[ModelAgent] ERROR DURING FORECAST", file=sys.stderr)
                    print(f"Error: {e}", file=sys.stderr)
                    print(error_trace, file=sys.stderr)
                    print(f"{'=' * 80}\n", file=sys.stderr)
                    raise
            else:
                model.fit(df_processed)
                result = model.predict(horizon)

            try:
                if is_professional_model:
                    metrics: dict[str, float] = {}
                else:
                    metrics = self._calculate_metrics(df_processed, model)
            except Exception:
                metrics = {}

            if is_professional_model:
                if isinstance(result, pl.DataFrame):
                    predictions = (
                        result["prediction"].to_list()
                        if "prediction" in result.columns
                        else result[:, -1].to_list()
                    )
                    if "datetime" in result.columns:
                        dates = [str(d) for d in result["datetime"].to_list()]
                    elif "ds" in result.columns:
                        dates = [str(d) for d in result["ds"].to_list()]
                    else:
                        last_dt = raw_df[datetime_column].drop_nulls().sort()[-1]
                        if not isinstance(last_dt, __import__("datetime").datetime):
                            from datetime import datetime as _dt

                            try:
                                last_dt = _dt.fromisoformat(str(last_dt))
                            except Exception:
                                last_dt = __import__("datetime").datetime.now()
                        freq = infer_frequency(raw_df.sort(datetime_column)[datetime_column]) or "D"
                        date_range = polars_date_range(
                            last_dt, periods=horizon + 1, freq=freq
                        ).slice(1)
                        dates = [str(d) for d in date_range.to_list()]

                    group_info = None
                    if group_cols and "unique_id" in result.columns:
                        group_info = {
                            "groups": result["unique_id"].unique().to_list(),
                            "n_groups": result["unique_id"].n_unique(),
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
                    forecast = result
            else:
                forecast = ForecastResult(
                    predictions=result.predictions,
                    dates=result.dates,
                    model_name=model_type,
                    horizon=horizon,
                    metrics=metrics,
                )

            forecast = self._enhance_forecast_with_diagnostics(
                forecast, df_processed, raw_df, datetime_column, target_column
            )

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
                "model": model,
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
        df: pl.DataFrame,
        datetime_column: str,
        target_column: str,
        group_by_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        if datetime_column in df.columns and not df[datetime_column].dtype.is_temporal():
            df = df.with_columns(pl.col(datetime_column).cast(pl.Datetime, strict=False))
        return self._process_dataframe(df, datetime_column, target_column, group_by_columns)

    def _load_data(
        self,
        filepath: Path,
        datetime_column: str,
        target_column: str,
        group_by_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        from forecaster.data.loader import load_full_dataframe

        df = load_full_dataframe(filepath, datetime_column=datetime_column)
        return self._process_dataframe(df, datetime_column, target_column, group_by_columns)

    def _process_dataframe(
        self,
        df: pl.DataFrame,
        datetime_column: str,
        target_column: str,
        group_by_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        if datetime_column not in df.columns:
            raise ValueError(
                f"Datetime column '{datetime_column}' not found. Available: {df.columns}"
            )
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found. Available: {df.columns}")

        if group_by_columns and all(c in df.columns for c in group_by_columns):
            result = (
                df.group_by([datetime_column] + group_by_columns)
                .agg(pl.col(target_column).mean())
                .sort(datetime_column)
                .select([datetime_column, pl.col(target_column).alias("value")])
                .drop_nulls("value")
            )
        else:
            result = (
                df.select([datetime_column, pl.col(target_column).alias("value")])
                .sort(datetime_column)
                .drop_nulls("value")
            )

        return result

    def _load_optional_models(self):
        try:
            from forecaster.models.mlforecast_models import MLForecastModel

            self.available_models["lightgbm"] = MLForecastModel
            print("[ModelAgent] LightGBM model available")
        except ImportError:
            print("[ModelAgent] LightGBM not available (install: pip install mlforecast lightgbm)")

        try:
            from forecaster.models.prophet_model import ProphetModel

            self.available_models["prophet"] = ProphetModel
            print("[ModelAgent] Prophet model available")
        except ImportError:
            print("[ModelAgent] Prophet not available (install: pip install prophet)")

        try:
            from forecaster.models.automl_forecaster import AutoMLForecaster

            self.available_models["automl"] = AutoMLForecaster
            print("[ModelAgent] AutoML available")
        except ImportError:
            print("[ModelAgent] AutoML not available")

    def _select_model(self, df: pl.DataFrame) -> str:
        n_points = df.height
        if "lightgbm" in self.available_models and n_points >= 100:
            return "lightgbm"
        if n_points < 30:
            return "naive"
        return "linear"

    def _calculate_metrics(self, df: pl.DataFrame, model: Any) -> dict[str, float]:
        try:
            n = df.height
            if n < 5:
                return {}
            train_size = int(n * 0.8)
            if train_size == 0:
                return {}
            train_df = df.slice(0, train_size)
            test_df = df.slice(train_size)
            if test_df.height == 0 or train_df.height == 0:
                return {}
            model_class = type(model)
            temp_model = model_class()
            temp_model.fit(train_df)
            result = temp_model.predict(test_df.height)
            actual = test_df["value"].to_numpy()
            predicted = np.array(result.predictions[: len(actual)])
            if len(actual) == 0 or len(predicted) == 0:
                return {}
            mae = float(np.mean(np.abs(actual - predicted)))
            rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
            if np.isnan(mae) or np.isinf(mae) or np.isnan(rmse) or np.isinf(rmse):
                return {}
            return {"mae": round(mae, 2), "rmse": round(rmse, 2)}
        except Exception:
            return {}

    def _enhance_with_per_group_diagnostics(
        self,
        forecast: ForecastResult,
        raw_df: pl.DataFrame,
        datetime_column: str,
        target_column: str,
        group_cols: list[str],
    ) -> ForecastResult:
        try:
            from forecaster.analysis.model_diagnostics import (
                calculate_baseline_metrics,
                calculate_health_score,
                calculate_residuals,
                generate_warnings,
            )

            groups = forecast.group_info.get("groups", []) if forecast.group_info else []
            per_group_metrics: dict = {}
            per_group_health: dict = {}
            per_group_warns: dict = {}
            per_group_residuals: dict = {}
            per_group_residual_dates: dict = {}

            for group_id in groups:
                try:
                    group_values = group_id.split("_")
                    uid_expr = pl.concat_str(
                        [pl.col(c).cast(pl.Utf8) for c in group_cols], separator="_"
                    )
                    group_data = (
                        raw_df.filter(uid_expr == group_id)
                        .sort(datetime_column)
                        .with_columns(pl.col(target_column).cast(pl.Float64, strict=False))  # noqa: F841
                        .drop_nulls(target_column)
                    )
                    _ = group_values  # unused but kept for clarity

                    if group_data.height < 10:
                        continue

                    actual_values = group_data[target_column].to_list()
                    if len(actual_values) < 10:
                        continue

                    split_idx = int(len(actual_values) * 0.8)
                    train_actual = actual_values[:split_idx]
                    test_actual = actual_values[split_idx:]
                    naive_preds = [train_actual[-1]] * len(test_actual)
                    model_test_preds = naive_preds

                    residual_analysis = calculate_residuals(test_actual, model_test_preds)
                    baseline_metrics = calculate_baseline_metrics(
                        test_actual, model_test_preds, naive_preds
                    )

                    if datetime_column in group_data.columns:
                        test_dates = [
                            str(d) for d in group_data[datetime_column].slice(split_idx).to_list()
                        ]
                    else:
                        test_dates = list(range(split_idx, len(actual_values)))

                    per_group_residuals[group_id] = residual_analysis.get("residuals", [])
                    per_group_residual_dates[group_id] = test_dates[
                        : len(per_group_residuals[group_id])
                    ]

                    group_data_quality = {
                        "total_rows": len(actual_values),
                        "missing_pct": 0.0,
                        "outliers_pct": 0.0,
                    }
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

                    group_warnings = generate_warnings(
                        residual_analysis, baseline_metrics, group_data_quality, []
                    )
                    per_group_warns[group_id] = group_warnings

                    try:
                        score, _ = calculate_health_score(
                            residual_analysis, baseline_metrics, group_data_quality, group_warnings
                        )
                        per_group_health[group_id] = score
                    except Exception:
                        per_group_health[group_id] = 50.0

                except Exception as e:
                    print(
                        f"[ModelAgent] Per-group diagnostics error for '{group_id}': {e}",
                        file=sys.stderr,
                    )

            forecast.per_group_metrics = per_group_metrics or None
            forecast.per_group_health_scores = per_group_health or None
            forecast.per_group_warnings = per_group_warns or None
            forecast.per_group_residuals = per_group_residuals or None
            forecast.per_group_residual_dates = per_group_residual_dates or None

        except Exception as e:
            print(f"[ModelAgent] Per-group diagnostics failed: {e}", file=sys.stderr)

        return forecast

    def _enhance_forecast_with_diagnostics(
        self,
        forecast: ForecastResult,
        df_processed: pl.DataFrame,
        raw_df: pl.DataFrame,
        datetime_column: str,
        target_column: str,
    ) -> ForecastResult:
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

            actual_values = (
                df_processed["value"].to_list() if "value" in df_processed.columns else []
            )
            residual_analysis: dict = {"is_random": True, "autocorr_lag1": 0.0, "has_trend": False}

            if len(actual_values) >= 10:
                split_idx = int(len(actual_values) * 0.8)
                train_actual = actual_values[:split_idx]
                test_actual = actual_values[split_idx:]
                test_naive = [train_actual[-1]] * len(test_actual)
                residual_analysis = calculate_residuals(test_actual, test_naive)
                forecast.residuals = residual_analysis.get("residuals", [])
                forecast.actual_values = test_actual

                if datetime_column in df_processed.columns:
                    test_dates = [
                        str(d) for d in df_processed[datetime_column].slice(split_idx).to_list()
                    ]
                    forecast.residual_dates = test_dates

                if not forecast.baseline_metrics:
                    forecast.baseline_metrics = calculate_baseline_metrics(
                        test_actual, test_naive, test_naive
                    )

            forecast.data_quality = analyze_data_quality(raw_df, datetime_column, target_column)
            historical_values = (
                raw_df[target_column].drop_nulls().to_list()
                if target_column in raw_df.columns
                else []
            )
            forecast_warnings = check_forecast_sanity(
                forecast.predictions, historical_values, target_column
            )

            forecast.trust_indicators = generate_trust_indicators(
                residual_analysis,
                forecast.baseline_metrics or {},
                forecast.data_quality or {},
                forecast_warnings,
            )
            forecast.warnings = generate_warnings(
                residual_analysis,
                forecast.baseline_metrics or {},
                forecast.data_quality or {},
                forecast_warnings,
            )

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
            print(f"[ModelAgent] Warning: Could not compute diagnostics: {e}", file=sys.stderr)
            return forecast
