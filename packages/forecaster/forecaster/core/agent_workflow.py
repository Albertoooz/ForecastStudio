"""
Agent-powered forecast workflow — bridges new multi-agent pipeline
with the battle-tested training infrastructure from workflow_engine.

Flow:
  1. MemoryManagerAgent  → pre-flight resource check
  2. DataAnalyzerAgent   → DataProfile (frequency, trend, seasonality, outliers)
  3. FeatureEngineerAgent → feature specs (lags, rolling, date parts)
  4. ModelSelectorAgent  → recommended model + hyperparameters
  5. Training            → multi-model holdout evaluation (reuses workflow_engine logic)
  6. Final forecast      → uses best model, produces ForecastResult

The ContextWindow is stored so the UI can display the full audit trail.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecaster.core.context import (
    AgentDecision,
    ContextWindow,
    ResourceBudget,
)
from forecaster.core.session import ForecastResult

# ---------------------------------------------------------------------------
#  Data structures (compatible with format_workflow_message)
# ---------------------------------------------------------------------------


@dataclass
class AgentStepResult:
    """Result of a single agent step."""

    agent_name: str = ""
    status: str = "pending"  # pending | running | done | failed
    message: str = ""
    details: list[str] = field(default_factory=list)
    duration: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    decision: AgentDecision | None = None


@dataclass
class AgentWorkflowResult:
    """Complete workflow result (superset of old WorkflowResult)."""

    success: bool = False
    steps: dict[str, AgentStepResult] = field(default_factory=dict)
    best_model_name: str = ""
    best_model: Any = None
    forecast_result: ForecastResult | None = None
    prepared_df: pd.DataFrame | None = None
    all_model_results: dict[str, dict] = field(default_factory=dict)
    total_duration: float = 0.0
    features_config: dict[str, Any] = field(default_factory=dict)
    # New: full ContextWindow with audit trail
    context: ContextWindow | None = None


# ---------------------------------------------------------------------------
#  Feature spec → workflow_engine format converter
# ---------------------------------------------------------------------------


def _specs_to_features_config(
    feature_specs: list,
    gap: int,
    n_rows: int,
) -> dict[str, Any]:
    """Convert FeatureEngineerAgent specs → MLForecastModel format."""
    lags = sorted(s.parameters.get("lag", 1) for s in feature_specs if s.feature_type == "lag")
    rolling_windows = sorted(
        {
            s.parameters.get("window", 7)
            for s in feature_specs
            if s.feature_type in ("rolling_mean", "rolling_std")
        }
    )

    # Only include date features that are valid pandas DatetimeIndex attributes.
    # MLForecast uses getattr(DatetimeIndex, feat) so custom ones like
    # 'is_weekend' would raise AttributeError.
    _VALID_DT_ATTRS = {  # noqa: N806
        "dayofweek",
        "day_of_week",
        "month",
        "dayofyear",
        "day_of_year",
        "day",
        "hour",
        "minute",
        "second",
        "quarter",
        "year",
        "week",
        "weekday",
    }
    date_features = [
        s.parameters.get("part", s.name)
        for s in feature_specs
        if s.feature_type == "date_part" and s.parameters.get("part", s.name) in _VALID_DT_ATTRS
    ]

    # Ensure minimum lags
    if not lags:
        lags = [max(1, gap + 1), max(7, gap + 7), max(14, gap + 14)]
    if not rolling_windows:
        rolling_windows = [7, 14]
    if not date_features:
        date_features = ["dayofweek", "month", "dayofyear"]

    config = {
        "lags": lags,
        "rolling_windows": rolling_windows,
        "date_features": date_features,
        "use_ewm": n_rows >= 500,
    }

    # Apply user overrides from session_state (legacy UI; empty in API-only mode)
    from forecaster.utils.streamlit_optional import get_session_state

    ss = get_session_state()
    override_lags = ss.get("feature_override_lags")
    if override_lags is not None and len(override_lags) > 0:
        config["lags"] = sorted(override_lags)

    override_rolling = ss.get("feature_override_rolling")
    if override_rolling is not None and len(override_rolling) > 0:
        config["rolling_windows"] = sorted(override_rolling)

    override_date = ss.get("feature_override_date")
    if override_date is not None and len(override_date) > 0:
        config["date_features"] = override_date

    override_ewm = ss.get("feature_override_ewm")
    if override_ewm is not None:
        config["use_ewm"] = bool(override_ewm)

    return config


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------


def run_agent_workflow(
    df: pd.DataFrame,
    datetime_column: str,
    target_column: str,
    horizon: int,
    gap: int = 0,
    group_cols: list[str] | None = None,
    frequency: str | None = None,
    model_type: str = "auto",
    progress_callback: Callable | None = None,
    memory_budget_mb: int = 2048,
    compute_budget_seconds: float = 600.0,
) -> AgentWorkflowResult:
    """
    Run the full forecast workflow using the new multi-agent pipeline.

    Uses agents for analysis/features/model selection,
    then the proven training logic for actual model training.
    """
    from forecaster.agents.data_analyzer import DataAnalyzerAgent
    from forecaster.agents.feature_engineer import FeatureEngineerAgent
    from forecaster.agents.memory_manager import MemoryManagerAgent
    from forecaster.agents.model_selector import ModelSelectorAgent

    result = AgentWorkflowResult()
    start_time = time.time()
    group_cols = group_cols or []

    def report(step: str, status: str, msg: str = ""):
        if progress_callback:
            try:
                progress_callback(step, status, msg)
            except Exception:
                pass

    # ===================================================================
    # Create ContextWindow
    # ===================================================================
    ctx = ContextWindow(
        budget=ResourceBudget(
            memory_budget_mb=memory_budget_mb,
            compute_budget_seconds=compute_budget_seconds,
        ),
        target_column=target_column,
        datetime_column=datetime_column,
        group_columns=group_cols,
        horizon=horizon,
        gap=gap,
    )
    ctx.register_data("primary", df)

    # ===================================================================
    # AGENT 1: Memory Manager
    # ===================================================================
    step = AgentStepResult(agent_name="memory_manager", status="running")
    result.steps["memory"] = step
    report("memory", "running", "🧠 Checking resources...")
    t0 = time.time()

    try:
        agent = MemoryManagerAgent()
        ctx, decision = agent.run(ctx)
        step.decision = decision
        step.duration = time.time() - t0

        if decision.decision_type == "error":
            step.status = "failed"
            step.message = f"Resource check failed: {decision.reasoning}"
            step.details = [f"❌ {decision.reasoning}"]
            report("memory", "failed", step.message)
            result.context = ctx
            result.total_duration = time.time() - start_time
            return result

        step.status = "done"
        step.message = "Resources OK"
        step.details = [
            f"✅ {decision.action}",
            f"  Budget: {ctx.budget.memory_budget_mb}MB RAM, "
            f"{ctx.budget.compute_budget_seconds:.0f}s compute",
        ]
        report("memory", "done", step.message)
    except Exception as e:
        step.status = "failed"
        step.message = f"Resource check error: {e}"
        step.details = [f"⚠️ {step.message} — continuing"]
        report("memory", "failed", step.message)

    # ===================================================================
    # AGENT 2: Data Analyzer
    # ===================================================================
    step = AgentStepResult(agent_name="data_analyzer", status="running")
    result.steps["analysis"] = step
    report("analysis", "running", "🔍 Analyzing data with DataAnalyzerAgent...")
    t0 = time.time()

    try:
        agent = DataAnalyzerAgent()
        ctx, decision = agent.run(ctx)
        step.decision = decision
        step.duration = time.time() - t0

        profile = ctx.data_profile
        if profile:
            step.details.append(f"📊 {profile.n_rows:,} rows × {profile.n_columns} columns")
            if profile.frequency:
                step.details.append(f"📅 Frequency: {profile.frequency}")
            if profile.date_range:
                step.details.append(f"📅 Range: {profile.date_range[0]} → {profile.date_range[1]}")
            if profile.missing_pct > 0:
                severity = "⚠️" if profile.missing_pct > 5 else "ℹ️"
                step.details.append(f"{severity} Missing values: {profile.missing_pct:.1f}%")
            if profile.outlier_pct > 0:
                step.details.append(f"ℹ️ Outliers (IQR): {profile.outlier_pct:.1f}%")
            if profile.has_seasonality:
                step.details.append(
                    f"🔄 Seasonality detected (period={profile.seasonality_period})"
                )
            if profile.has_trend:
                step.details.append("📈 Trend detected")
            if profile.cleaning_recommendations:
                for rec in profile.cleaning_recommendations:
                    step.details.append(
                        f"  💡 Recommendation: {rec['type']} ({rec['severity']}) — {rec['detail']}"
                    )

            # Use agent-detected frequency if we didn't have one
            if not frequency and profile.frequency:
                frequency = profile.frequency

            # Detect potential exogenous variables
            reserved = {datetime_column, target_column, "unique_id", "ds", "y"} | set(group_cols)
            potential_exog = [
                c
                for c in df.columns
                if c not in reserved
                and pd.api.types.is_numeric_dtype(df[c])
                and not c.startswith("Unnamed")
                and c != "index"
            ]
            if potential_exog:
                step.details.append(
                    f"ℹ️ {len(potential_exog)} potential exogenous variables "
                    f"(e.g., {', '.join(potential_exog[:3])})"
                )

            step.data = {
                "n_rows": profile.n_rows,
                "frequency": profile.frequency,
                "missing_pct": profile.missing_pct,
                "has_seasonality": profile.has_seasonality,
                "has_trend": profile.has_trend,
            }

        step.status = "done"
        step.message = f"Analyzed {profile.n_rows:,} rows" if profile else "Analysis complete"
        report("analysis", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Analysis failed: {e}"
        step.details = [f"❌ {step.message}"]
        report("analysis", "failed", step.message)
        # Non-fatal: continue with defaults

    # ===================================================================
    # STEP 2b: Data Preparation (uses existing robust logic)
    # ===================================================================
    step = AgentStepResult(agent_name="data_preparation", status="running")
    result.steps["preparation"] = step
    report("preparation", "running", "🔧 Preparing data...")
    t0 = time.time()

    try:
        from forecaster.agents.forecast_wizard import (
            analyze_for_wizard,
            apply_preparation,
            suggest_preparations,
        )

        # Build analysis dict for compatibility
        analysis = analyze_for_wizard(
            df, datetime_column, target_column, group_cols, frequency, gap
        )

        prep_steps = suggest_preparations(analysis)
        prepared_df, prep_log = apply_preparation(
            df, prep_steps, datetime_column, target_column, frequency, group_cols
        )

        # Auto-clip negative values for non-negative metrics
        non_neg_kw = ["count", "quantity", "volume", "sales", "revenue", "price", "amount"]
        is_non_negative = any(k in target_column.lower() for k in non_neg_kw)
        if is_non_negative:
            stats = analysis.get("stats", {})
            if stats.get("negative_pct", 0) > 0 and target_column in prepared_df.columns:
                n_neg = int((prepared_df[target_column] < 0).sum())
                if n_neg > 0:
                    prepared_df = prepared_df.copy()
                    prepared_df[target_column] = prepared_df[target_column].clip(lower=0)
                    prep_log.append(f"✅ Clipped {n_neg:,} negative values to 0")

        step.details = prep_log if prep_log else ["✅ Data is clean — no preparation needed"]
        step.data = {"n_rows_before": len(df), "n_rows_after": len(prepared_df)}
        result.prepared_df = prepared_df
        step.status = "done"
        step.duration = time.time() - t0
        step.message = f"Prepared ({len(prepared_df):,} rows)"
        report("preparation", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Preparation failed: {e}"
        step.details = [f"⚠️ {step.message} — using original data"]
        result.prepared_df = df.copy()
        report("preparation", "failed", step.message)

    # Also update context's primary data with prepared df
    if result.prepared_df is not None:
        ctx.register_data("primary", result.prepared_df)

    # ===================================================================
    # AGENT 3: Feature Engineer
    # ===================================================================
    step = AgentStepResult(agent_name="feature_engineer", status="running")
    result.steps["features"] = step
    report("features", "running", "⚙️ FeatureEngineerAgent configuring features...")
    t0 = time.time()

    try:
        agent = FeatureEngineerAgent()
        ctx, decision = agent.run(ctx)
        step.decision = decision
        step.duration = time.time() - t0

        n_specs = len(ctx.feature_specs)
        # Convert to workflow_engine format for training
        prep_df = result.prepared_df if result.prepared_df is not None else df
        features_config = _specs_to_features_config(ctx.feature_specs, gap, len(prep_df))
        result.features_config = features_config

        step.details.append(f"📊 Generated {n_specs} feature specifications:")
        step.details.append(f"  Lags: {features_config['lags']}")
        step.details.append(f"  Rolling windows: mean/std {features_config['rolling_windows']}")
        step.details.append(f"  Date features: {', '.join(features_config['date_features'])}")
        if features_config.get("use_ewm"):
            step.details.append("  Exponentially weighted mean: enabled")

        from forecaster.utils.streamlit_optional import get_session_state

        ss = get_session_state()
        active = []
        if ss.get("feature_override_lags"):
            active.append("lags")
        if ss.get("feature_override_rolling"):
            active.append("rolling")
        if ss.get("feature_override_date"):
            active.append("date features")
        if ss.get("feature_override_ewm") is not None:
            active.append("ewm")
        if active:
            step.details.append(f"  🔧 User overrides active for: {', '.join(active)}")

        # Show estimated impact for top features
        top_specs = sorted(
            ctx.feature_specs,
            key=lambda s: s.estimated_impact or 0,
            reverse=True,
        )[:5]
        if top_specs and top_specs[0].estimated_impact:
            step.details.append("  🎯 Top features by estimated impact:")
            for s in top_specs:
                impact = s.estimated_impact or 0
                step.details.append(f"    • {s.name}: {impact:.3f}")

        step.status = "done"
        step.message = f"{n_specs} features specified"
        report("features", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Feature engineering failed: {e}"
        step.details = [f"⚠️ {step.message} — using defaults"]
        # Build data-size-aware defaults
        prep_df = result.prepared_df if result.prepared_df is not None else df
        n = len(prep_df)
        max_lag = max(1, n // 3)
        max_w = max(1, n // 4)
        default_lags = [l for l in [1, 7, 14] if l <= max_lag]  # noqa: E741
        default_windows = [w for w in [7, 14] if w <= max_w]
        features_config = {
            "lags": default_lags or [1],
            "rolling_windows": default_windows or [7],
            "date_features": ["dayofweek", "month", "dayofyear"],
            "use_ewm": False,
        }
        result.features_config = features_config
        report("features", "failed", step.message)

    # ===================================================================
    # AGENT 4: Model Selector
    # ===================================================================
    step = AgentStepResult(agent_name="model_selector", status="running")
    result.steps["model_selection"] = step
    report("model_selection", "running", "🤖 ModelSelectorAgent choosing model...")
    t0 = time.time()

    recommended_model = "auto"
    try:
        agent = ModelSelectorAgent()
        ctx, decision = agent.run(ctx)
        step.decision = decision
        step.duration = time.time() - t0

        if ctx.model_spec:
            recommended_model = ctx.model_spec.model_type
            step.details.append(
                f"🏆 Recommended: **{recommended_model}** (confidence: {decision.confidence:.0%})"
            )
            step.details.append(f"  💡 {decision.reasoning}")
            if ctx.model_spec.hyperparameters:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in list(ctx.model_spec.hyperparameters.items())[:4]
                )
                step.details.append(f"  ⚙️ Params: {params_str}")

        step.status = "done"
        step.message = f"Recommended: {recommended_model}"
        step.data = {"recommended_model": recommended_model}
        report("model_selection", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Model selection failed: {e}"
        step.details = [f"⚠️ {step.message} — will try all models"]
        report("model_selection", "failed", step.message)

    # ===================================================================
    # STEP 5: Model Training (multi-model + holdout eval)
    # ===================================================================
    step = AgentStepResult(agent_name="model_training", status="running")
    result.steps["training"] = step
    report("training", "running", "🏋️ Training models...")
    t0 = time.time()

    train_df = result.prepared_df if result.prepared_df is not None else df.copy()

    from forecaster.agents.model_agent import ModelAgent

    model_agent = ModelAgent()
    available = model_agent.get_available_models()

    # Use agent recommendation to prioritize models
    MODEL_NAME_MAP = {  # noqa: N806
        "simple_ewm": "naive",
        "lightgbm_default": "lightgbm",
        "mlforecast_global": "lightgbm",
        "prophet": "prophet",
        "naive": "naive",
        "linear": "linear",
    }

    if group_cols and len(group_cols) > 0:
        # Grouped data: use LightGBM
        if "lightgbm" in available:
            models_to_train = ["lightgbm"]
            step.details.append("ℹ️ Grouped data — using LightGBM (supports multi-group)")
        else:
            models_to_train = ["naive", "linear"]
    elif model_type != "auto":
        mapped = MODEL_NAME_MAP.get(model_type, model_type)
        models_to_train = [mapped] if mapped in available else ["naive", "linear"]
    elif recommended_model != "auto":
        # Agent recommended a specific model — still train competitors
        primary = MODEL_NAME_MAP.get(recommended_model, recommended_model)
        models_to_train = []
        if primary in available:
            models_to_train.append(primary)
        # Add comparison models (always train at least naive + lightgbm)
        for m in ["naive", "linear", "lightgbm"]:
            if m in available and m not in models_to_train:
                models_to_train.append(m)
        step.details.append(
            f"ℹ️ Agent recommended **{recommended_model}** → "
            f"training {', '.join(m.upper() for m in models_to_train)} for comparison"
        )
    else:
        models_to_train = [m for m in ["naive", "linear", "lightgbm"] if m in available]

    # Import training helpers from workflow_engine
    from forecaster.agents.workflow_engine import (
        _evaluate_model_on_holdout,
        _tune_lightgbm_hyperparameters,
    )

    model_results = {}
    trained_models = {}
    features_config = result.features_config

    from forecaster.utils.streamlit_optional import get_session_state

    ss = get_session_state()
    future_exog_df = ss.get("future_exog_df")
    model_hyperparam_overrides = ss.get("model_override_hyperparams") or {}

    # If user provided hyperparameter overrides, inject into features_config
    if model_hyperparam_overrides:
        features_config = {**features_config, "lgb_params": model_hyperparam_overrides}
        step.details.append(
            f"⚙️ User hyperparameter overrides: {', '.join(f'{k}={v}' for k, v in model_hyperparam_overrides.items())}"
        )

    for model_name in models_to_train:
        report("training", "running", f"Training {model_name.upper()}...")
        try:
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
            dur_str = f" ({mr.get('train_time', 0):.1f}s)" if mr.get("train_time") else ""

            report("training", "running", f"✅ {model_name.upper()} — {rmse_str} {mape_str}")
            step.details.append(f"✅ {model_name.upper()} — {rmse_str} {mape_str}{dur_str}")

            # Log to context
            ctx.log_decision(
                AgentDecision(
                    agent_name="model_trainer",
                    decision_type="model_training",
                    action=f"trained: {model_name}",
                    parameters={
                        "holdout_rmse": rmse if isinstance(rmse, (int, float)) else None,
                        "holdout_mape": mape if isinstance(mape, (int, float)) else None,
                    },
                    confidence=0.8,
                    reasoning=f"{model_name.upper()}: {rmse_str} {mape_str}",
                    estimated_compute_seconds=mr.get("train_time", 0),
                )
            )

        except Exception as e:
            report("training", "running", f"❌ {model_name.upper()} failed")
            step.details.append(f"❌ {model_name.upper()} — {e}")
            print(f"[AgentWorkflow] {model_name} failed: {e}", file=sys.stderr)

    # Hyperparameter tuning for LightGBM
    if "lightgbm" in trained_models and len(train_df) >= 500:
        report("training", "running", "Tuning LightGBM hyperparameters...")
        try:
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
            report("training", "running", f"⚠️ Tuning skipped: {e}")
            step.details.append(f"⚠️ Tuning skipped: {e}")

    step.status = "done"
    step.duration = time.time() - t0
    step.message = f"Trained {len(trained_models)} model(s) in {step.duration:.1f}s"
    result.all_model_results = model_results
    report("training", "done", step.message)

    # ===================================================================
    # STEP 6: Evaluation & Selection
    # ===================================================================
    step = AgentStepResult(agent_name="evaluation", status="running")
    result.steps["evaluation"] = step
    report("evaluation", "running", "🏆 Selecting best model...")
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

        # Comparison with agent recommendation
        agent_recommended = MODEL_NAME_MAP.get(recommended_model, recommended_model)
        if agent_recommended != "auto" and best_name != agent_recommended:
            step.details.append(
                f"ℹ️ Agent recommended **{recommended_model}** but "
                f"**{best_name.upper()}** won on holdout RMSE"
            )
        elif agent_recommended != "auto":
            step.details.append(
                f"✅ Agent recommendation confirmed: **{best_name.upper()}** is best"
            )

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
            model_results.items(),
            key=lambda x: x[1].get("holdout_rmse", float("inf")),
        ):
            rmse = res.get("holdout_rmse", None)
            marker = " ⭐" if name == best_name else ""
            if isinstance(rmse, (int, float)) and rmse < 1e10:
                step.details.append(f"  {name.upper()}: RMSE={rmse:.2f}{marker}")

        # Log evaluation decision
        ctx.log_decision(
            AgentDecision(
                agent_name="evaluator",
                decision_type="model_selection",
                action=f"selected: {best_name}",
                parameters={
                    "best_rmse": best_rmse if isinstance(best_rmse, (int, float)) else None,
                    "improvement_vs_naive": improvement,
                    "agent_recommendation": recommended_model,
                    "leaderboard": {n: r.get("holdout_rmse") for n, r in model_results.items()},
                },
                confidence=0.9 if improvement > 10 else 0.7,
                reasoning=(
                    f"Best model: {best_name.upper()} "
                    f"(RMSE={best_rmse:.2f}, +{improvement:.1f}% vs naive)"
                ),
            )
        )

        step.status = "done"
        step.duration = time.time() - t0
        step.message = f"Best: {best_name.upper()}"
        step.data = {
            "best_model": best_name,
            "best_rmse": best_rmse,
            "improvement": improvement,
        }
        report("evaluation", "done", step.message)

    except Exception as e:
        step.status = "failed"
        step.message = f"Selection failed: {e}"
        result.best_model_name = models_to_train[0] if models_to_train else "naive"
        result.best_model = trained_models.get(result.best_model_name)
        report("evaluation", "failed", step.message)

    # ===================================================================
    # STEP 7: Final Forecast
    # ===================================================================
    step = AgentStepResult(agent_name="forecast", status="running")
    result.steps["forecast"] = step
    report("forecast", "running", "✅ Generating final forecast...")
    t0 = time.time()

    try:
        best_model = result.best_model
        best_name = result.best_model_name

        if best_model is None:
            raise ValueError("No model available for forecasting")

        actual_model_type = best_name.replace("_tuned", "")

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
            # Simple models
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

        # Build metrics
        best_holdout = model_results.get(result.best_model_name, {})
        naive_holdout = model_results.get("naive", {})

        rmse_model = best_holdout.get("holdout_rmse", 0)
        rmse_naive = naive_holdout.get("holdout_rmse", 0)
        mape_model = best_holdout.get("holdout_mape", 0)
        mape_naive = naive_holdout.get("holdout_mape", 0)
        rmse_improvement = (rmse_naive - rmse_model) / rmse_naive * 100 if rmse_naive > 0 else 0
        mape_improvement = (mape_naive - mape_model) / mape_naive * 100 if mape_naive > 0 else 0

        real_baseline = {
            "rmse_model": rmse_model,
            "rmse_naive": rmse_naive,
            "mape_model": mape_model,
            "mape_naive": mape_naive,
            "rmse_improvement_pct": rmse_improvement,
            "mape_improvement_pct": mape_improvement,
            "beats_baseline": rmse_improvement > 0,
        }

        # Data quality & sanity checks
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
        residual_analysis = {
            "is_random": True,
            "autocorr_lag1": 0.0,
            "has_trend": False,
        }

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

        # Per-group diagnostics
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
                            f"[AgentWorkflow] Per-group diagnostics error for '{gid}': {ge}",
                            file=sys.stderr,
                        )

                forecast.per_group_metrics = per_group_metrics or None
                forecast.per_group_health_scores = per_group_health or None
                forecast.per_group_warnings = per_group_warns or None
                forecast.per_group_residuals = per_group_residuals or None
                forecast.per_group_residual_dates = per_group_residual_dates or None

            except Exception as pg_err:
                print(
                    f"[AgentWorkflow] Per-group diagnostics failed: {pg_err}",
                    file=sys.stderr,
                )

        result.forecast_result = forecast

        # Log final decision
        ctx.log_decision(
            AgentDecision(
                agent_name="forecaster",
                decision_type="forecast",
                action=f"forecast_complete: {actual_model_type}",
                parameters={
                    "n_predictions": len(predictions),
                    "health_score": health_score,
                    "horizon": horizon,
                },
                confidence=0.9,
                reasoning=(
                    f"Generated {len(predictions)} predictions "
                    f"with health score {health_score:.0f}/100"
                ),
            )
        )

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
        step.message = f"Forecast failed: {e}"
        step.details = [f"❌ {step.message}"]
        step.duration = time.time() - t0
        report("forecast", "failed", step.message)

    # ===================================================================
    # Finalize
    # ===================================================================
    result.success = result.steps.get("forecast", AgentStepResult()).status == "done"
    result.total_duration = time.time() - start_time
    result.context = ctx

    return result


# ---------------------------------------------------------------------------
#  Message formatting (compatible with existing UI)
# ---------------------------------------------------------------------------


def format_agent_workflow_message(result: AgentWorkflowResult) -> str:
    """Format agent workflow results as a chat message."""
    lines = []
    lines.append("🤖 **Multi-Agent Forecast Pipeline**\n")

    # Step overview
    step_names = {
        "memory": ("🧠", "Resource Check"),
        "analysis": ("🔍", "Data Analysis (DataAnalyzerAgent)"),
        "preparation": ("🔧", "Data Preparation"),
        "features": ("⚙️", "Feature Engineering (FeatureEngineerAgent)"),
        "model_selection": ("🤖", "Model Selection (ModelSelectorAgent)"),
        "training": ("🏋️", "Model Training"),
        "evaluation": ("🏆", "Evaluation & Selection"),
        "forecast": ("✅", "Final Forecast"),
    }

    lines.append("**📋 Pipeline:**")
    for i, (step_id, (icon, step_name)) in enumerate(step_names.items(), 1):  # noqa: B007
        step = result.steps.get(step_id, AgentStepResult())
        status = "✅" if step.status == "done" else "❌" if step.status == "failed" else "⬜"
        dur = f" ({step.duration:.1f}s)" if step.duration > 0.1 else ""
        lines.append(f"  {status} {i}. {step_name}{dur}")
    lines.append("\n---\n")

    # Detailed step results
    for step_id, (icon, step_name) in step_names.items():
        step = result.steps.get(step_id, AgentStepResult())
        if step.status == "pending":
            continue

        lines.append(f"{icon} **{step_name}**")
        for detail in step.details:
            lines.append(f"  {detail}")
        lines.append("")

    # Agent audit trail summary
    if result.context:
        ctx = result.context
        n_decisions = len(ctx.decision_log)
        cost = ctx.get_cost_estimate()
        lines.append("---")
        lines.append(f"📋 **Audit Trail:** {n_decisions} agent decisions logged")
        lines.append(
            f"💰 **Cost:** ${cost['total_cost_usd']:.6f} "
            f"(compute: {ctx.budget.consumed_compute_seconds:.1f}s)"
        )

    # Footer
    if result.success:
        lines.append("")
        lines.append(f"⏱️ Total: **{result.total_duration:.1f}s**")
        lines.append(
            "\n_Check **Dashboard** for charts, "
            "**Model Analysis** for diagnostics, "
            "and **🤖 Pipeline** for the full audit trail._"
        )
    else:
        failed = [k for k, v in result.steps.items() if v.status == "failed"]
        lines.append(f"❌ Workflow failed at: {', '.join(failed)}")

    return "\n".join(lines)
