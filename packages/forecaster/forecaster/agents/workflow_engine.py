"""
Production-grade autonomous forecasting workflow.

Runs a complete pipeline autonomously:
1. Data analysis (quality, issues, statistics)
2. Data preparation (missing values, outliers, duplicates)
3. Feature engineering configuration
4. Multi-model training + hyperparameter tuning
5. Evaluation & model selection
6. Final forecast generation with diagnostics

Reports progress via callback for live UI updates.
"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecaster.core.session import ForecastResult

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class StepResult:
    """Result of a single workflow step."""

    status: str = "pending"  # pending, running, done, failed, skipped
    message: str = ""
    details: list[str] = field(default_factory=list)
    duration: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Complete workflow result."""

    success: bool = False
    steps: dict[str, StepResult] = field(default_factory=dict)
    best_model_name: str = ""
    best_model: Any = None
    forecast_result: ForecastResult | None = None
    prepared_df: pd.DataFrame | None = None
    all_model_results: dict[str, dict] = field(default_factory=dict)
    total_duration: float = 0.0
    features_config: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def run_forecast_workflow(
    df: pd.DataFrame,
    datetime_column: str,
    target_column: str,
    horizon: int,
    gap: int = 0,
    group_cols: list[str] | None = None,
    frequency: str | None = None,
    model_type: str = "auto",
    progress_callback: Callable | None = None,
) -> WorkflowResult:
    """
    Run complete autonomous forecasting workflow.

    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column
        target_column: Name of target column
        horizon: Forecast horizon (periods)
        gap: Gap between now and first forecast period
        group_cols: Columns for multivariate grouping
        frequency: Data frequency (e.g., "15min", "D")
        model_type: "auto" to try all, or specific name
        progress_callback: fn(step_name, status, message) for live updates

    Returns:
        WorkflowResult with all steps, best model, and forecast
    """
    result = WorkflowResult()
    start_time = time.time()
    group_cols = group_cols or []

    def report(step, status, msg=""):
        if progress_callback:
            try:
                progress_callback(step, status, msg)
            except Exception:
                pass

    # =========================================================================
    # STEP 1: DATA ANALYSIS
    # =========================================================================
    step = StepResult(status="running")
    result.steps["analysis"] = step
    report("analysis", "running", "Analyzing data quality...")
    t0 = time.time()

    try:
        from forecaster.agents.forecast_wizard import analyze_for_wizard

        analysis = analyze_for_wizard(
            df, datetime_column, target_column, group_cols, frequency, gap
        )

        stats = analysis.get("stats", {})
        issues = analysis.get("issues", [])
        group_stats = analysis.get("group_stats", {})

        step.details.append(f"{analysis['n_rows']:,} rows × {analysis['n_columns']} columns")
        step.details.append(
            f"Target: `{target_column}` — "
            f"mean={stats.get('mean', 0):,.2f}, std={stats.get('std', 0):,.2f}, "
            f"range=[{stats.get('min', 0):,.2f}, {stats.get('max', 0):,.2f}]"
        )

        if stats.get("zeros_pct", 0) > 10:
            step.details.append(f"⚠️ Sparse data: {stats['zeros_pct']:.0f}% zeros")
        if stats.get("negative_pct", 0) > 0:
            step.details.append(f"ℹ️ Negative values: {stats['negative_pct']:.1f}%")
        if group_stats:
            step.details.append(
                f"👥 {group_stats.get('n_groups', 0)} groups ({', '.join(group_cols)})"
                f" — sizes: {group_stats.get('min_group_size', 0):,}–{group_stats.get('max_group_size', 0):,}"
            )
        if frequency:
            step.details.append(f"📊 Frequency: {frequency}")
        if stats.get("date_range"):
            step.details.append(f"📅 Range: {stats['date_range']}")

        for issue in issues:
            step.details.append(f"{issue['severity']} {issue['message']}")

        # Detect potential exogenous variables
        reserved_cols = {datetime_column, target_column, "unique_id", "ds", "y"} | set(group_cols)
        potential_exog = [
            col
            for col in df.columns
            if col not in reserved_cols
            and pd.api.types.is_numeric_dtype(df[col])
            and not col.startswith("Unnamed")
            and col != "index"
        ]

        # Store potential_exog in analysis dict for later use
        analysis["potential_exog"] = potential_exog
        analysis["use_exog"] = False
        analysis["future_exog_df"] = None

        if potential_exog:
            step.details.append(
                f"ℹ️ Detected {len(potential_exog)} potential exogenous variables (e.g., {', '.join(potential_exog[:3])}{'...' if len(potential_exog) > 3 else ''})"
            )

            # Optional future exog (legacy Streamlit session; empty in API-only mode)
            from forecaster.utils.streamlit_optional import get_session_state

            ss = get_session_state()
            future_exog_df = ss.get("future_exog_df")
            has_future_exog = future_exog_df is not None

            if has_future_exog:
                step.details.append(
                    f"✅ Future exog data provided: {len(future_exog_df):,} rows × {len(future_exog_df.columns)} cols"
                )
                analysis["use_exog"] = True
                analysis["future_exog_df"] = future_exog_df
            else:
                step.details.append(
                    "⚠️ **No future exog data provided.** Using only autoregressive features (lags, rolling, date, EWM)."
                )
                step.details.append(
                    "💡 **Tip:** To use exogenous variables, upload a CSV with future values in the sidebar (📊 Future Exogenous Variables section)."
                )

        step.data = analysis
        step.status = "done"
        step.duration = time.time() - t0
        step.message = f"Analyzed {analysis['n_rows']:,} rows"
        report("analysis", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Analysis failed: {str(e)}"
        step.details = [step.message]
        report("analysis", "failed", step.message)
        result.total_duration = time.time() - start_time
        return result

    # =========================================================================
    # STEP 2: DATA PREPARATION
    # =========================================================================
    step = StepResult(status="running")
    result.steps["preparation"] = step
    report("preparation", "running", "Preparing data...")
    t0 = time.time()

    try:
        from forecaster.agents.forecast_wizard import apply_preparation, suggest_preparations

        prep_steps = suggest_preparations(analysis)
        prepared_df, prep_log = apply_preparation(
            df, prep_steps, datetime_column, target_column, frequency, group_cols
        )

        # Auto-clip negative values for non-negative metrics
        non_neg_kw = ["count", "quantity", "volume", "sales", "revenue", "price", "amount"]
        is_non_negative = any(k in target_column.lower() for k in non_neg_kw)

        if is_non_negative and stats.get("negative_pct", 0) > 0:
            if target_column in prepared_df.columns:
                n_neg = int((prepared_df[target_column] < 0).sum())
                if n_neg > 0:
                    prepared_df = prepared_df.copy()
                    prepared_df[target_column] = prepared_df[target_column].clip(lower=0)
                    prep_log.append(
                        f"✅ Clipped {n_neg:,} negative values to 0 (non-negative metric)"
                    )

        step.details = prep_log if prep_log else ["✅ Data is clean — no preparation needed"]
        step.data = {"n_rows_before": len(df), "n_rows_after": len(prepared_df)}
        result.prepared_df = prepared_df

        step.status = "done"
        step.duration = time.time() - t0
        step.message = f"Prepared ({len(prepared_df):,} rows)"
        report("preparation", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Preparation failed: {str(e)}"
        step.details = [f"⚠️ {step.message} — using original data"]
        result.prepared_df = df.copy()
        report("preparation", "failed", step.message)

    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    step = StepResult(status="running")
    result.steps["features"] = step
    report("features", "running", "Configuring features...")
    t0 = time.time()

    try:
        features_config = _configure_features(frequency, gap, analysis)
        result.features_config = features_config

        step.details.append(f"Lags: {features_config['lags']}")
        step.details.append(f"Rolling windows: mean/std {features_config['rolling_windows']}")
        step.details.append(f"Date features: {', '.join(features_config['date_features'])}")
        if features_config.get("use_ewm"):
            step.details.append("Exponentially weighted mean: enabled")

        n_estimated = (
            len(features_config["lags"])
            + len(features_config["lags"]) * len(features_config["rolling_windows"]) * 2
            + len(features_config["date_features"])
        )
        step.details.append(f"📊 Estimated features: ~{n_estimated}")

        step.data = features_config
        step.status = "done"
        step.duration = time.time() - t0
        step.message = f"{len(features_config['lags'])} lags, {len(features_config['date_features'])} date features"
        report("features", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Feature config failed: {str(e)}"
        step.details = [f"⚠️ {step.message} — using defaults"]
        features_config = _default_features_config()
        result.features_config = features_config
        report("features", "failed", step.message)

    # =========================================================================
    # STEP 4: MODEL TRAINING
    # =========================================================================
    step = StepResult(status="running")
    result.steps["training"] = step
    report("training", "running", "Training models...")
    t0 = time.time()

    train_df = result.prepared_df if result.prepared_df is not None else df.copy()

    # Determine which models to train
    from forecaster.agents.model_agent import ModelAgent

    agent = ModelAgent()
    available = agent.get_available_models()

    # For grouped data, ONLY use LightGBM (simple models don't support groups)
    if group_cols and len(group_cols) > 0:
        if "lightgbm" in available:
            models_to_train = ["lightgbm"]
            step_result = result.steps.get("training", StepResult())
            if step_result:
                step_result.details.insert(
                    0,
                    "ℹ️ Grouped data detected — using LightGBM (simple models don't support groups)",
                )
        else:
            # Fallback if LightGBM not available (shouldn't happen)
            models_to_train = ["naive", "linear"]
    elif model_type != "auto":
        models_to_train = (
            [model_type]
            if model_type in available
            else list(available.keys())
            if isinstance(available, dict)
            else available[:1]
        )
    else:
        models_to_train = [m for m in ["naive", "linear", "lightgbm"] if m in available]

    model_results = {}
    trained_models = {}

    for model_name in models_to_train:
        report("training", "running", f"Training {model_name.upper()}...")
        try:
            # Get future_exog_df from analysis (set in Step 1)
            future_exog_df = analysis.get("future_exog_df")

            mr = _evaluate_model_on_holdout(
                model_name,
                train_df,
                datetime_column,
                target_column,
                horizon,
                gap,
                group_cols,
                features_config,
                future_exog_df=future_exog_df,
            )

            model_results[model_name] = mr
            if mr.get("model"):
                trained_models[model_name] = mr["model"]

            rmse = mr.get("holdout_rmse", "N/A")
            mape = mr.get("holdout_mape", "N/A")

            rmse_str = (
                f"RMSE: {rmse:.2f}"
                if isinstance(rmse, (int, float)) and rmse < 1e10
                else "RMSE: N/A"
            )
            mape_str = (
                f"| MAPE: {mape:.1f}%" if isinstance(mape, (int, float)) and mape < 1e6 else ""
            )
            duration_str = f" ({mr.get('train_time', 0):.1f}s)" if mr.get("train_time") else ""

            report("training", "running", f"✅ {model_name.upper()} — {rmse_str} {mape_str}")
            step.details.append(f"✅ {model_name.upper()} — {rmse_str} {mape_str}{duration_str}")

        except Exception as e:
            report("training", "running", f"❌ {model_name.upper()} failed")
            step.details.append(f"❌ {model_name.upper()} — {str(e)}")
            print(f"[Workflow] {model_name} failed: {str(e)}", file=sys.stderr)

    # Hyperparameter tuning for LightGBM
    if "lightgbm" in trained_models and len(train_df) >= 500:
        report("training", "running", "Tuning LightGBM hyperparameters...")
        try:
            # Get future_exog_df from analysis
            future_exog_df = analysis.get("future_exog_df")

            tuned = _tune_lightgbm_hyperparameters(
                train_df,
                datetime_column,
                target_column,
                horizon,
                gap,
                group_cols,
                features_config,
                future_exog_df=future_exog_df,
            )
            if tuned and tuned.get("holdout_rmse", float("inf")) < model_results.get(
                "lightgbm", {}
            ).get("holdout_rmse", float("inf")):
                model_results["lightgbm_tuned"] = tuned
                if tuned.get("model"):
                    trained_models["lightgbm_tuned"] = tuned["model"]
                rmse = tuned["holdout_rmse"]
                params_str = ", ".join(
                    f"{k}={v}" for k, v in list(tuned.get("best_params", {}).items())[:3]
                )
                report("training", "running", f"✅ LightGBM (tuned) — RMSE: {rmse:.2f} ⭐")
                step.details.append(f"✅ LightGBM (tuned) — RMSE: {rmse:.2f} ⭐ ({params_str})")
            else:
                report("training", "running", "ℹ️ Tuning did not improve over default")
                step.details.append("ℹ️ Tuning did not improve over default LightGBM")
        except Exception as e:
            report("training", "running", f"⚠️ Tuning skipped: {str(e)}")
            step.details.append(f"⚠️ Tuning skipped: {str(e)}")

    step.status = "done"
    step.duration = time.time() - t0
    step.message = f"Trained {len(trained_models)} model(s) in {step.duration:.1f}s"
    result.all_model_results = model_results
    report("training", "done", step.message)

    # =========================================================================
    # STEP 5: EVALUATION & SELECTION
    # =========================================================================
    step = StepResult(status="running")
    result.steps["evaluation"] = step
    report("evaluation", "running", "Selecting best model...")
    t0 = time.time()

    try:
        best_name = None
        best_rmse = float("inf")

        for name, res in model_results.items():
            rmse = res.get("holdout_rmse")
            if rmse is not None and isinstance(rmse, (int, float)) and rmse < best_rmse:
                best_rmse = rmse
                best_name = name

        if best_name is None:
            best_name = models_to_train[0] if models_to_train else "naive"
            best_rmse = model_results.get(best_name, {}).get("holdout_rmse", 0)

        result.best_model_name = best_name
        result.best_model = trained_models.get(best_name)

        # Improvement vs naive
        naive_rmse = model_results.get("naive", {}).get("holdout_rmse")
        improvement = 0.0
        if (
            naive_rmse
            and isinstance(naive_rmse, (int, float))
            and naive_rmse > 0
            and isinstance(best_rmse, (int, float))
        ):
            improvement = (1 - best_rmse / naive_rmse) * 100

        step.details.append(f"🏆 **Best model: {best_name.upper()}**")
        if isinstance(best_rmse, (int, float)) and best_rmse < 1e10:
            step.details.append(f"RMSE: {best_rmse:.2f}")
        if improvement > 0:
            step.details.append(f"📈 +{improvement:.1f}% improvement vs naive baseline")
        elif improvement < 0:
            step.details.append(f"⚠️ {improvement:.1f}% worse than naive baseline")

        # Leaderboard
        step.details.append("")
        step.details.append("**Model leaderboard:**")
        for name, res in sorted(
            model_results.items(), key=lambda x: x[1].get("holdout_rmse", float("inf"))
        ):
            rmse = res.get("holdout_rmse", None)
            marker = " ⭐" if name == best_name else ""
            if isinstance(rmse, (int, float)) and rmse < 1e10:
                step.details.append(f"  {name.upper()}: RMSE={rmse:.2f}{marker}")

        step.status = "done"
        step.duration = time.time() - t0
        step.message = f"Best: {best_name.upper()}"
        step.data = {"best_model": best_name, "best_rmse": best_rmse, "improvement": improvement}
        report("evaluation", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Selection failed: {str(e)}"
        result.best_model_name = models_to_train[0] if models_to_train else "naive"
        result.best_model = trained_models.get(result.best_model_name)
        report("evaluation", "failed", step.message)

    # =========================================================================
    # STEP 6: FINAL FORECAST
    # =========================================================================
    step = StepResult(status="running")
    result.steps["forecast"] = step
    report("forecast", "running", "Generating final forecast...")
    t0 = time.time()

    try:
        best_model = result.best_model
        best_name = result.best_model_name

        if best_model is None:
            raise ValueError("No model available for forecasting")

        actual_model_type = best_name.replace("_tuned", "")

        # ---- Predict using the ALREADY-TRAINED best_model directly ----
        # (No re-training via agent.forecast — use the exact model that was evaluated)
        from forecaster.models.mlforecast_models import MLForecastModel

        if isinstance(best_model, MLForecastModel):
            pred_df = best_model.predict(horizon)
            predictions = pred_df["prediction"].tolist()
            if "datetime" in pred_df.columns:
                dates = pred_df["datetime"].astype(str).tolist()
            elif "ds" in pred_df.columns:
                dates = pred_df["ds"].astype(str).tolist()
            else:
                dates = list(range(len(predictions)))

            group_info = None
            if group_cols and "unique_id" in pred_df.columns:
                group_info = {
                    "groups": pred_df["unique_id"].unique().tolist(),
                    "n_groups": pred_df["unique_id"].nunique(),
                    "group_columns": group_cols,
                }
        else:
            # Simple models (naive / linear) — need to re-predict
            proc_df = train_df.copy()
            proc_df[datetime_column] = pd.to_datetime(proc_df[datetime_column], errors="coerce")
            proc_df = proc_df.dropna(subset=[datetime_column]).set_index(datetime_column)
            proc_df = proc_df[[target_column]].copy()
            proc_df.columns = ["value"]
            proc_df = proc_df.sort_index().dropna()
            best_model.fit(proc_df)
            simple_result = best_model.predict(horizon)
            predictions = simple_result.predictions
            dates = simple_result.dates
            group_info = None

        # ---- Build ForecastResult with REAL holdout metrics ----
        best_holdout = model_results.get(result.best_model_name, {})
        naive_holdout = model_results.get("naive", {})

        rmse_model = best_holdout.get("holdout_rmse", 0)
        rmse_naive = naive_holdout.get("holdout_rmse", 0)
        mape_model = best_holdout.get("holdout_mape", 0)
        mape_naive = naive_holdout.get("holdout_mape", 0)

        rmse_improvement = ((rmse_naive - rmse_model) / rmse_naive * 100) if rmse_naive > 0 else 0
        mape_improvement = ((mape_naive - mape_model) / mape_naive * 100) if mape_naive > 0 else 0

        real_baseline = {
            "rmse_model": rmse_model,
            "rmse_naive": rmse_naive,
            "mape_model": mape_model,
            "mape_naive": mape_naive,
            "rmse_improvement_pct": rmse_improvement,
            "mape_improvement_pct": mape_improvement,
            "beats_baseline": rmse_improvement > 0,
        }

        # ---- Data quality & sanity checks ----
        from forecaster.analysis.model_diagnostics import (
            analyze_data_quality,
            calculate_health_score,
            calculate_residuals,
            check_forecast_sanity,
            generate_trust_indicators,
            generate_warnings,
        )

        data_quality = analyze_data_quality(df, datetime_column, target_column)
        historical_values = (
            df[target_column].dropna().tolist() if target_column in df.columns else []
        )
        forecast_sanity = check_forecast_sanity(predictions, historical_values, target_column)
        residual_analysis = {"is_random": True, "autocorr_lag1": 0.0, "has_trend": False}

        warnings_list = generate_warnings(
            residual_analysis, real_baseline, data_quality, forecast_sanity
        )
        trust_list = generate_trust_indicators(
            residual_analysis, real_baseline, data_quality, forecast_sanity
        )
        health_score, _ = calculate_health_score(
            residual_analysis, real_baseline, data_quality, warnings_list
        )

        forecast = ForecastResult(
            predictions=predictions,
            dates=dates,
            model_name=actual_model_type,
            horizon=horizon,
            metrics={},
            group_info=group_info,
            baseline_metrics=real_baseline,
            data_quality=data_quality,
            warnings=warnings_list,
            trust_indicators=trust_list,
            health_score=health_score,
        )

        # ---- Per-group diagnostics ----
        if isinstance(best_model, MLForecastModel) and group_cols and best_model.has_groups():
            try:
                per_group_metrics = {}
                per_group_health = {}
                per_group_warns = {}
                per_group_residuals = {}
                per_group_residual_dates = {}

                groups = group_info.get("groups", []) if group_info else []
                for gid in groups:
                    try:
                        uid_col = "_unique_id_tmp"
                        gdf = df.copy()
                        gdf[uid_col] = gdf[group_cols].astype(str).agg("_".join, axis=1)
                        gdf = gdf[gdf[uid_col] == gid].sort_values(datetime_column)
                        gdf[target_column] = pd.to_numeric(gdf[target_column], errors="coerce")
                        actual_vals = gdf[target_column].dropna().tolist()

                        if len(actual_vals) < 20:
                            continue

                        split_idx = int(len(actual_vals) * 0.8)
                        train_act = actual_vals[:split_idx]
                        test_act = actual_vals[split_idx:]
                        naive_preds = [train_act[-1]] * len(test_act)

                        res_analysis = calculate_residuals(test_act, naive_preds)

                        # Use workflow-level holdout metrics for baseline comparison
                        group_bm = {
                            "rmse_model": rmse_model,
                            "rmse_naive": rmse_naive,
                            "mape_model": mape_model,
                            "mape_naive": mape_naive,
                            "rmse_improvement_pct": rmse_improvement,
                            "mape_improvement_pct": mape_improvement,
                            "beats_baseline": rmse_improvement > 0,
                        }

                        if datetime_column in gdf.columns:
                            test_dates = gdf[datetime_column].iloc[split_idx:].astype(str).tolist()
                        else:
                            test_dates = list(range(split_idx, len(actual_vals)))

                        per_group_residuals[gid] = res_analysis.get("residuals", [])
                        per_group_residual_dates[gid] = test_dates[: len(per_group_residuals[gid])]

                        group_dq = {
                            "total_rows": len(actual_vals),
                            "missing_pct": 0.0,
                            "outliers_pct": 0.0,
                        }

                        per_group_metrics[gid] = {
                            "n_observations": len(actual_vals),
                            "mean": float(np.mean(actual_vals)),
                            "std": float(np.std(actual_vals)),
                        }
                        per_group_metrics[gid].update(group_bm)

                        gw = generate_warnings(res_analysis, group_bm, group_dq, [])
                        per_group_warns[gid] = gw

                        try:
                            gs, _ = calculate_health_score(res_analysis, group_bm, group_dq, gw)
                            per_group_health[gid] = gs
                        except Exception:
                            per_group_health[gid] = 50.0

                    except Exception as ge:
                        print(
                            f"[Workflow] Per-group diagnostics error for '{gid}': {ge}",
                            file=sys.stderr,
                        )

                forecast.per_group_metrics = per_group_metrics or None
                forecast.per_group_health_scores = per_group_health or None
                forecast.per_group_warnings = per_group_warns or None
                forecast.per_group_residuals = per_group_residuals or None
                forecast.per_group_residual_dates = per_group_residual_dates or None

            except Exception as pg_err:
                print(f"[Workflow] Per-group diagnostics failed: {pg_err}", file=sys.stderr)

        result.forecast_result = forecast

        # ---- Step summary ----
        n_preds = len(predictions)
        n_groups = group_info.get("n_groups", 1) if group_info else 1

        step.details.append(f"✅ {n_preds:,} predictions generated")
        if n_groups > 1:
            step.details.append(f"📊 {n_groups} separate models (per group)")

        if forecast.health_score is not None:
            hs = forecast.health_score
            label = (
                "EXCELLENT" if hs >= 85 else "GOOD" if hs >= 70 else "FAIR" if hs >= 50 else "POOR"
            )
            color = "🟢" if hs >= 70 else "🟡" if hs >= 50 else "🔴"
            step.details.append(f"{color} Health Score: {hs:.0f}/100 ({label})")

        if forecast.warnings:
            step.details.append(f"⚠️ {len(forecast.warnings)} warning(s)")

        step.status = "done"
        step.message = f"{n_preds:,} predictions"

        step.duration = time.time() - t0
        report("forecast", step.status, step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Forecast failed: {str(e)}"
        step.details = [f"❌ {step.message}"]
        step.duration = time.time() - t0
        report("forecast", "failed", step.message)

    result.success = result.steps.get("forecast", StepResult()).status == "done"
    result.total_duration = time.time() - start_time

    return result


# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================


def _periods_per_day(frequency: str | None) -> int:
    """Calculate periods per day from frequency string."""
    if not frequency:
        return 1
    freq = frequency.strip()
    if freq.endswith("min"):
        mins = int(freq.replace("min", "") or "1")
        return max(1, 1440 // mins)
    elif freq.endswith("h"):
        hours = int(freq.replace("h", "") or "1")
        return max(1, 24 // hours)
    elif freq in ("D", "1D"):
        return 1
    return 1


def _configure_features(
    frequency: str | None,
    gap: int,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Determine optimal feature engineering configuration based on data."""
    config: dict[str, Any] = {
        "lags": [],
        "rolling_windows": [],
        "date_features": [],
        "use_ewm": False,
    }

    n_rows = analysis.get("n_rows", 0)
    ppd = _periods_per_day(frequency)

    if gap == 0:
        if ppd >= 24:
            # Sub-hourly (e.g., 15min → 96 ppd)
            config["lags"] = sorted({1, ppd // 4, ppd // 2, ppd, ppd * 7})
            config["rolling_windows"] = sorted({ppd // 4, ppd, ppd * 7})
            config["date_features"] = [
                "hour",
                "dayofweek",
                "month",
                "is_weekend",
                "dayofyear",
                "quarter",
            ]
        elif ppd >= 2:
            # Hourly
            config["lags"] = [1, 12, 24, 168]
            config["rolling_windows"] = [12, 24, 168]
            config["date_features"] = ["hour", "dayofweek", "month", "is_weekend", "dayofyear"]
        else:
            # Daily
            config["lags"] = [1, 7, 14, 28]
            config["rolling_windows"] = [7, 14, 28]
            config["date_features"] = ["dayofweek", "month", "dayofyear", "quarter", "is_weekend"]
    else:
        base = gap + 1
        config["lags"] = sorted(
            {
                base,
                base + max(7, ppd),
                base + max(14, ppd * 7),
                base + max(28, ppd * 14),
            }
        )
        config["rolling_windows"] = sorted({max(7, base), max(14, base + 7)})
        config["date_features"] = ["dayofweek", "month", "dayofyear"]

    # Limit lags based on smallest group size (avoid all-NaN features)
    min_group_size = analysis.get("group_stats", {}).get("min_group_size", n_rows)
    max_safe_lag = max(10, min_group_size // 3)
    config["lags"] = [l for l in config["lags"] if l < max_safe_lag and l > 0]  # noqa: E741
    config["rolling_windows"] = [w for w in config["rolling_windows"] if w < max_safe_lag and w > 0]

    # Ensure we always have some lags
    if not config["lags"]:
        config["lags"] = [1, 7, 14]
    if not config["rolling_windows"]:
        config["rolling_windows"] = [7, 14]

    # EWM for larger datasets
    config["use_ewm"] = n_rows >= 500

    return config


def _default_features_config() -> dict[str, Any]:
    return {
        "lags": [1, 7, 14],
        "rolling_windows": [7, 14],
        "date_features": ["dayofweek", "month", "dayofyear"],
        "use_ewm": False,
    }


# =============================================================================
# MODEL EVALUATION
# =============================================================================


def _evaluate_model_on_holdout(
    model_name: str,
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    horizon: int,
    gap: int,
    group_cols: list[str] | None,
    features_config: dict[str, Any],
    future_exog_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train a model and evaluate on holdout. Returns RMSE, MAPE, trained model."""
    t0 = time.time()

    # For evaluation: aggregate to single series (fast, consistent)
    eval_df = df.copy()
    if datetime_col in eval_df.columns:
        eval_df[datetime_col] = pd.to_datetime(eval_df[datetime_col], errors="coerce")
        eval_df = eval_df.dropna(subset=[datetime_col])
        eval_df = eval_df.sort_values(datetime_col)

    eval_df[target_col] = pd.to_numeric(eval_df[target_col], errors="coerce")
    eval_df = eval_df.dropna(subset=[target_col])

    n = len(eval_df)
    if n < 20:
        return {
            "holdout_rmse": float("inf"),
            "holdout_mape": float("inf"),
            "model": None,
            "train_time": 0,
        }

    split = int(n * 0.8)
    train_split = eval_df.iloc[:split]
    test_split = eval_df.iloc[split:]
    test_values = test_split[target_col].values

    if model_name in ("lightgbm", "lightgbm_tuned"):
        return _eval_lightgbm(
            df,
            train_split,
            test_values,
            datetime_col,
            target_col,
            horizon,
            gap,
            group_cols,
            features_config,
            t0,
            future_exog_df=future_exog_df,
        )
    elif model_name == "linear":
        return _eval_simple_model(
            "linear", train_split, test_values, datetime_col, target_col, df, t0
        )
    elif model_name == "naive":
        return _eval_simple_model(
            "naive", train_split, test_values, datetime_col, target_col, df, t0
        )
    else:
        return {
            "holdout_rmse": float("inf"),
            "holdout_mape": float("inf"),
            "model": None,
            "train_time": 0,
        }


def _eval_lightgbm(
    full_df,
    train_split,
    test_values,
    datetime_col,
    target_col,
    horizon,
    gap,
    group_cols,
    features_config,
    t0,
    lgb_params=None,
    future_exog_df=None,
):
    """Evaluate LightGBM on holdout and retrain on full data."""
    from forecaster.models.mlforecast_models import MLForecastModel

    # Holdout evaluation (single series, no groups — fast)
    eval_model = MLForecastModel(
        model_type="lightgbm",
        lags=features_config.get("lags", [1, 7, 14]),
        date_features=features_config.get("date_features", ["dayofweek", "month", "dayofyear"]),
        rolling_windows=features_config.get("rolling_windows", [7, 14]),
        use_ewm=features_config.get("use_ewm", False),
        gap=gap,
        lgb_params=lgb_params or {},
    )

    # Drop group columns from holdout eval to prevent them becoming exogenous features
    eval_train = train_split.copy()
    if group_cols:
        eval_train = eval_train.drop(
            columns=[c for c in group_cols if c in eval_train.columns], errors="ignore"
        )

    eval_model.fit(
        eval_train, datetime_col, target_col, group_by_columns=None, future_exog_df=future_exog_df
    )

    h = min(len(test_values), horizon * 2)
    preds_df = eval_model.predict(h)
    pred_values = preds_df["prediction"].values

    min_len = min(len(test_values), len(pred_values))
    if min_len > 0:
        actual = test_values[:min_len]
        predicted = pred_values[:min_len]
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mape = float(np.mean(np.abs((actual - predicted) / (np.abs(actual) + 1e-10))) * 100)
    else:
        rmse, mape = float("inf"), float("inf")

    # Retrain on full data WITH groups for the final model
    final_model = MLForecastModel(
        model_type="lightgbm",
        lags=features_config.get("lags", [1, 7, 14]),
        date_features=features_config.get("date_features", ["dayofweek", "month", "dayofyear"]),
        rolling_windows=features_config.get("rolling_windows", [7, 14]),
        use_ewm=features_config.get("use_ewm", False),
        gap=gap,
        lgb_params=lgb_params or {},
    )
    final_model.fit(
        full_df,
        datetime_col,
        target_col,
        group_by_columns=group_cols if group_cols else None,
        future_exog_df=future_exog_df,
    )

    return {
        "holdout_rmse": rmse,
        "holdout_mape": mape,
        "model": final_model,
        "model_name": "lightgbm",
        "train_time": time.time() - t0,
    }


def _eval_simple_model(model_name, train_split, test_values, datetime_col, target_col, full_df, t0):
    """Evaluate a simple model (naive, linear) on holdout."""
    from forecaster.models.simple import LinearForecaster, NaiveForecaster

    # Prepare for simple models (DatetimeIndex + 'value')
    train_processed = train_split.copy()
    if datetime_col in train_processed.columns:
        train_processed[datetime_col] = pd.to_datetime(
            train_processed[datetime_col], errors="coerce"
        )
        train_processed = train_processed.dropna(subset=[datetime_col])
        train_processed = train_processed.set_index(datetime_col)
    train_processed = train_processed[[target_col]].copy()
    train_processed.columns = ["value"]
    train_processed = train_processed.sort_index().dropna()

    if model_name == "naive":
        model = NaiveForecaster()
    else:
        model = LinearForecaster()

    model.fit(train_processed)
    result = model.predict(len(test_values))
    predicted = np.array(result.predictions[: len(test_values)])

    min_len = min(len(test_values), len(predicted))
    if min_len > 0:
        actual = test_values[:min_len]
        pred = predicted[:min_len]
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        mape = float(np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-10))) * 100)
    else:
        rmse, mape = float("inf"), float("inf")

    # For final model, retrain on full data
    full_processed = full_df.copy()
    if datetime_col in full_processed.columns:
        full_processed[datetime_col] = pd.to_datetime(full_processed[datetime_col], errors="coerce")
        full_processed = full_processed.dropna(subset=[datetime_col])
        full_processed = full_processed.set_index(datetime_col)
    full_processed = full_processed[[target_col]].copy()
    full_processed.columns = ["value"]
    full_processed = full_processed.sort_index().dropna()

    final_model = NaiveForecaster() if model_name == "naive" else LinearForecaster()
    final_model.fit(full_processed)

    return {
        "holdout_rmse": rmse,
        "holdout_mape": mape,
        "model": final_model,
        "model_name": model_name,
        "train_time": time.time() - t0,
    }


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================


def _tune_lightgbm_hyperparameters(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    horizon: int,
    gap: int,
    group_cols: list[str] | None,
    features_config: dict[str, Any],
    future_exog_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Simple grid search over LightGBM hyperparameters."""
    from forecaster.models.mlforecast_models import MLFORECAST_AVAILABLE

    if not MLFORECAST_AVAILABLE:
        return None

    # Quick holdout setup
    eval_df = df.copy()
    if datetime_col in eval_df.columns:
        eval_df[datetime_col] = pd.to_datetime(eval_df[datetime_col], errors="coerce")
        eval_df = eval_df.dropna(subset=[datetime_col]).sort_values(datetime_col)
    eval_df[target_col] = pd.to_numeric(eval_df[target_col], errors="coerce")
    eval_df = eval_df.dropna(subset=[target_col])

    n = len(eval_df)
    split = int(n * 0.8)
    test_values = eval_df[target_col].values[split:]

    if len(test_values) == 0:
        return None

    # Configurations to try (diverse set)
    configs = [
        {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31, "max_depth": 6},
        {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 63, "max_depth": 10},
        {"n_estimators": 1000, "learning_rate": 0.01, "num_leaves": 127, "max_depth": -1},
        {
            "n_estimators": 300,
            "learning_rate": 0.08,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
        },
    ]

    best_rmse = float("inf")
    best_config = None

    for cfg in configs:
        try:
            result = _eval_lightgbm(
                df,
                eval_df.iloc[:split],
                test_values,
                datetime_col,
                target_col,
                horizon,
                gap,
                group_cols,
                features_config,
                time.time(),
                lgb_params=cfg,
                future_exog_df=future_exog_df,
            )
            rmse = result.get("holdout_rmse", float("inf"))
            if isinstance(rmse, (int, float)) and rmse < best_rmse:
                best_rmse = rmse
                best_config = cfg
        except Exception:
            continue

    if best_config is None:
        return None

    # Retrain best config on full data (already done by _eval_lightgbm)
    final = _eval_lightgbm(
        df,
        eval_df.iloc[:split],
        test_values,
        datetime_col,
        target_col,
        horizon,
        gap,
        group_cols,
        features_config,
        time.time(),
        lgb_params=best_config,
        future_exog_df=future_exog_df,
    )
    final["best_params"] = best_config
    final["model_name"] = "lightgbm_tuned"
    return final


# =============================================================================
# MESSAGE FORMATTING
# =============================================================================


def format_workflow_message(result: WorkflowResult) -> str:
    """Format workflow results as a Cursor-style chat message with step progress."""
    lines = []

    lines.append("🧙 **Forecast Workflow**\n")

    # Plan overview (Cursor-style checklist)
    step_names = {
        "analysis": "Data Analysis",
        "preparation": "Data Preparation",
        "features": "Feature Engineering",
        "training": "Model Training",
        "evaluation": "Model Evaluation & Selection",
        "forecast": "Final Forecast",
    }

    lines.append("**📋 Plan:**")
    for i, (step_id, step_name) in enumerate(step_names.items(), 1):
        step = result.steps.get(step_id, StepResult())
        icon = "✅" if step.status == "done" else "❌" if step.status == "failed" else "⬜"
        dur = f" ({step.duration:.1f}s)" if step.duration > 0.1 else ""
        lines.append(f"  {icon} {i}. {step_name}{dur}")
    lines.append("\n---\n")

    # Detailed step results
    step_icons = {
        "analysis": "📊",
        "preparation": "🔧",
        "features": "⚙️",
        "training": "🏋️",
        "evaluation": "🏆",
        "forecast": "✅",
    }

    for step_id, step_name in step_names.items():
        step = result.steps.get(step_id, StepResult())
        if step.status == "pending":
            continue

        icon = step_icons.get(step_id, "•")
        lines.append(f"{icon} **{step_name}**")

        for detail in step.details:
            lines.append(f"  {detail}")

        lines.append("")

    # Footer
    if result.success:
        lines.append("---")
        lines.append(f"⏱️ Total: **{result.total_duration:.1f}s**")
        lines.append(
            "\n_Check **Dashboard** for charts and **Model Analysis** for full diagnostics._"
        )
    else:
        failed = [k for k, v in result.steps.items() if v.status == "failed"]
        lines.append(f"❌ Workflow failed at: {', '.join(failed)}")

    return "\n".join(lines)
