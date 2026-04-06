"""
Graph node functions — thin adapters between LangGraph state and the existing agents.

Rules:
  - Each function signature:  (state: ForecastGraphState) -> dict
  - The dict is a PARTIAL state update (only keys the node produces).
  - Zero business logic lives here — every node delegates to agents/models.
  - LangGraph imports are allowed here; they must NOT appear in forecaster/agents/*.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from langgraph.types import interrupt

from forecaster.graph.state import ForecastGraphState
from forecaster.utils.streamlit_optional import get_session_state

# ── helpers ──────────────────────────────────────────────────────────────────


def _ts() -> float:
    return time.time()


def _prog(text: str, step: str, status: str = "info") -> dict:
    return {"text": text, "step": step, "status": status, "ts": _ts()}


def _get_df(state: ForecastGraphState) -> pd.DataFrame:
    """Return prepared_dataframe if available, else original dataframe."""
    prepared = state.get("prepared_dataframe")
    if prepared is not None:
        return prepared
    return state["dataframe"]


def _step(status: str, message: str, details: list[str], duration: float) -> dict:
    return {"status": status, "message": message, "details": details, "duration": duration}


# ═════════════════════════════════════════════════════════════════════════════
# NODE 1 — analyze_data
# ═════════════════════════════════════════════════════════════════════════════


def analyze_data(state: ForecastGraphState) -> dict:
    """
    Run DataAnalyzerAgent + analyze_for_wizard.

    May interrupt once to ask the user:
      • Whether to clip negative values to 0.
      • Whether to use detected exogenous variables (requires future data upload).
    After the user answers, the node re-runs from the top and interrupt() returns
    the answer immediately (LangGraph semantics).
    """
    from forecaster.agents.data_analyzer import DataAnalyzerAgent
    from forecaster.agents.forecast_wizard import analyze_for_wizard
    from forecaster.core.context import ContextWindow, ResourceBudget

    t0 = _ts()
    df: pd.DataFrame = state["dataframe"]
    datetime_column = state["datetime_column"]
    target_column = state["target_column"]
    group_columns = state["group_columns"]
    frequency = state.get("frequency")
    gap = state["gap"]
    horizon = state["horizon"]

    details: list[str] = []
    decisions: list[dict] = []

    # ── DataAnalyzerAgent ──────────────────────────────────────────────────────
    ctx = ContextWindow(
        budget=ResourceBudget(memory_budget_mb=2048, compute_budget_seconds=600.0),
        target_column=target_column,
        datetime_column=datetime_column,
        group_columns=group_columns,
        horizon=horizon,
        gap=gap,
    )
    ctx.register_data("primary", df)

    data_profile_dict: dict = {}
    try:
        agent = DataAnalyzerAgent()
        ctx, decision = agent.run(ctx)
        decisions.append(decision.model_dump())

        if ctx.data_profile:
            p = ctx.data_profile
            data_profile_dict = {
                "n_rows": p.n_rows,
                "n_columns": p.n_columns,
                "frequency": p.frequency,
                "missing_pct": p.missing_pct,
                "outlier_pct": p.outlier_pct,
                "has_trend": p.has_trend,
                "has_seasonality": p.has_seasonality,
                "seasonality_period": p.seasonality_period,
                "date_range": list(p.date_range) if p.date_range else None,
            }
            details.append(f"{p.n_rows:,} rows × {p.n_columns} columns")
            if p.frequency:
                details.append(f"Frequency: {p.frequency}")
                if not frequency:
                    frequency = p.frequency
            if p.missing_pct > 0:
                details.append(f"Missing values: {p.missing_pct:.1f}%")
            if p.outlier_pct > 0:
                details.append(f"Outliers: {p.outlier_pct:.1f}%")
            if p.has_seasonality:
                details.append(f"Seasonality detected (period={p.seasonality_period})")
            if p.has_trend:
                details.append("Trend detected")

    except Exception as e:
        details.append(f"DataAnalyzerAgent warning (non-fatal): {e}")
        print(f"[graph/analyze_data] DataAnalyzerAgent: {e}", file=sys.stderr)

    # ── analyze_for_wizard (richer stats) ────────────────────────────────────
    try:
        wizard_analysis = analyze_for_wizard(
            df, datetime_column, target_column, group_columns, frequency, gap
        )
    except Exception as e:
        wizard_analysis = {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "stats": {},
            "issues": [],
        }
        details.append(f"Wizard analysis warning (non-fatal): {e}")

    stats = wizard_analysis.get("stats", {})
    if stats.get("zeros_pct", 0) > 10:
        details.append(f"Sparse data: {stats['zeros_pct']:.0f}% zeros")
    if stats.get("negative_pct", 0) > 0:
        details.append(f"Negative values: {stats['negative_pct']:.1f}% of target")
    if stats.get("date_range"):
        details.append(f"Date range: {stats['date_range']}")
    if group_stats := wizard_analysis.get("group_stats", {}):
        details.append(
            f"{group_stats.get('n_groups', 0)} groups — "
            f"sizes {group_stats.get('min_group_size', 0):,}–{group_stats.get('max_group_size', 0):,}"
        )

    # ── Detect potential exogenous columns ────────────────────────────────────
    reserved = {datetime_column, target_column, "unique_id", "ds", "y"} | set(group_columns)
    potential_exog = [
        col
        for col in df.columns
        if col not in reserved
        and pd.api.types.is_numeric_dtype(df[col])
        and not col.startswith("Unnamed")
        and col != "index"
    ]
    wizard_analysis["potential_exog"] = potential_exog
    if potential_exog:
        preview = ", ".join(potential_exog[:3]) + ("..." if len(potential_exog) > 3 else "")
        details.append(f"Detected {len(potential_exog)} potential exogenous variables: {preview}")

    # ── Build HITL question map ────────────────────────────────────────────────
    has_negatives = stats.get("negative_pct", 0) > 0
    has_exog = bool(potential_exog)
    has_future_exog = state.get("future_exog_df") is not None

    questions_needed: dict[str, dict] = {}
    if has_negatives:
        questions_needed["negative_values"] = {
            "question": (
                f"We detected **{stats['negative_pct']:.1f}% negative values** in `{target_column}`. "
                "Should we clip them to 0? "
                "Reply **yes** to clip, **no** to keep them."
            ),
            "options": ["yes", "no"],
        }
    if has_exog and not has_future_exog:
        exog_preview = ", ".join(potential_exog[:5]) + ("..." if len(potential_exog) > 5 else "")
        questions_needed["exog_variables"] = {
            "question": (
                f"We detected **{len(potential_exog)} potential exogenous variables** "
                f"({exog_preview}). "
                "Do you want to use them? If **yes**, upload a CSV with future values "
                "in the sidebar (📊 Future Exogenous Variables section) before answering. "
                "If **no**, we'll use only autoregressive features (lags, rolling stats, dates)."
            ),
            "options": ["yes", "no"],
        }

    # ── HITL interrupt (if needed and not already answered) ───────────────────
    user_responses: dict = state.get("user_responses") or {}
    if questions_needed and not user_responses:
        formatted = "\n\n".join(
            f"**Question {i + 1}:** {info['question']}"
            for i, (_, info) in enumerate(questions_needed.items())
        )
        answer = interrupt(
            {
                "type": "analysis_questions",
                "questions": questions_needed,
                "formatted_message": (
                    "**A few quick questions before we proceed:**\n\n"
                    + formatted
                    + "\n\n_Answer all questions separated by commas, e.g. `yes, no`. "
                    "If there's only one question, a single word is fine._"
                ),
            }
        )
        # interrupt() returns here when the graph is resumed with Command(resume=answer)
        user_responses = _parse_interrupt_answer(answer, list(questions_needed.keys()))

    # ── Resolve user answers ──────────────────────────────────────────────────
    clip_negatives = bool(user_responses.get("negative_values", "no") == "yes")
    use_exog = bool(user_responses.get("exog_variables", "no") == "yes") and has_future_exog

    wizard_analysis["clip_negatives"] = clip_negatives
    wizard_analysis["use_exog"] = use_exog
    if has_exog and has_future_exog:
        wizard_analysis["use_exog"] = True
        details.append("Future exogenous data provided — will be used in model")
    elif has_exog and not has_future_exog:
        details.append("No future exog data provided — using autoregressive features only")

    duration = _ts() - t0
    n_rows = wizard_analysis.get("n_rows", len(df))

    return {
        "data_profile": data_profile_dict,
        "analysis_data": wizard_analysis,
        "frequency": frequency,
        "user_responses": user_responses,
        "step_results": {"analysis": _step("done", f"Analyzed {n_rows:,} rows", details, duration)},
        "decision_log": decisions,
        "progress_messages": [_prog(f"Analyzed {n_rows:,} rows", "analysis", "done")],
        "errors": [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 2 — prepare_data
# ═════════════════════════════════════════════════════════════════════════════


def prepare_data(state: ForecastGraphState) -> dict:
    """Apply data preparation steps from forecast_wizard (fill, clip, dedup, etc.)."""
    from forecaster.agents.forecast_wizard import apply_preparation, suggest_preparations

    t0 = _ts()
    df: pd.DataFrame = state["dataframe"]
    analysis = state.get("analysis_data") or {}
    datetime_column = state["datetime_column"]
    target_column = state["target_column"]
    group_columns = state["group_columns"]
    frequency = state.get("frequency")

    try:
        prep_steps = suggest_preparations(analysis)
        prepared_df, prep_log = apply_preparation(
            df, prep_steps, datetime_column, target_column, frequency, group_columns
        )

        # Clip negatives if user confirmed or analysis flagged it
        if analysis.get("clip_negatives") and target_column in prepared_df.columns:
            n_neg = int((prepared_df[target_column] < 0).sum())
            if n_neg > 0:
                prepared_df = prepared_df.copy()
                prepared_df[target_column] = prepared_df[target_column].clip(lower=0)
                prep_log.append(f"Clipped {n_neg:,} negative values to 0 (user confirmed)")

        details = prep_log if prep_log else ["Data is clean — no preparation needed"]
        msg = f"Prepared ({len(prepared_df):,} rows)"

        return {
            "prepared_dataframe": prepared_df,
            "step_results": {"preparation": _step("done", msg, details, _ts() - t0)},
            "progress_messages": [_prog(msg, "preparation", "done")],
            "errors": [],
        }

    except Exception as e:
        print(f"[graph/prepare_data] {e}", file=sys.stderr)
        return {
            "prepared_dataframe": df.copy(),
            "step_results": {
                "preparation": _step(
                    "failed", f"Preparation failed — using original: {e}", [str(e)], _ts() - t0
                )
            },
            "progress_messages": [_prog(f"Preparation warning: {e}", "preparation", "warning")],
            "errors": [str(e)],
        }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 3 — engineer_features
# ═════════════════════════════════════════════════════════════════════════════


def engineer_features(state: ForecastGraphState) -> dict:
    """Run FeatureEngineerAgent and convert specs to MLForecastModel config."""
    from forecaster.agents.feature_engineer import FeatureEngineerAgent
    from forecaster.core.agent_workflow import _specs_to_features_config
    from forecaster.core.context import ContextWindow, ResourceBudget

    t0 = _ts()
    prep_df = (
        state.get("prepared_dataframe")
        if state.get("prepared_dataframe") is not None
        else state["dataframe"]
    )
    decisions: list[dict] = []
    details: list[str] = []

    # Rebuild a minimal ContextWindow for the agent
    ctx = ContextWindow(
        budget=ResourceBudget(memory_budget_mb=2048, compute_budget_seconds=600.0),
        target_column=state["target_column"],
        datetime_column=state["datetime_column"],
        group_columns=state["group_columns"],
        horizon=state["horizon"],
        gap=state["gap"],
    )
    ctx.register_data("primary", prep_df)

    # Inject data profile so the agent can make frequency-aware decisions
    if dp := state.get("data_profile"):
        from forecaster.core.context import DataProfile

        try:
            ctx.data_profile = DataProfile(**{k: v for k, v in dp.items() if v is not None})
        except Exception:
            pass

    features_config: dict = {}
    try:
        agent = FeatureEngineerAgent()
        ctx, decision = agent.run(ctx)
        decisions.append(decision.model_dump())

        n_specs = len(ctx.feature_specs)
        features_config = _specs_to_features_config(ctx.feature_specs, state["gap"], len(prep_df))

        details.append(f"Generated {n_specs} feature specifications")
        details.append(f"Lags: {features_config['lags']}")
        details.append(f"Rolling windows: {features_config['rolling_windows']}")
        details.append(f"Date features: {', '.join(features_config['date_features'])}")
        if features_config.get("use_ewm"):
            details.append("Exponential weighted mean: enabled")

    except Exception as e:
        print(f"[graph/engineer_features] {e}", file=sys.stderr)
        # Safe defaults proportional to dataset size
        n = len(prep_df)
        max_lag = max(1, n // 3)
        features_config = {
            "lags": [l for l in [1, 7, 14] if l <= max_lag] or [1],  # noqa: E741
            "rolling_windows": [w for w in [7, 14] if w <= max(1, n // 4)] or [7],
            "date_features": ["dayofweek", "month", "dayofyear"],
            "use_ewm": False,
        }
        details.append(f"FeatureEngineerAgent warning — using safe defaults: {e}")

    ss = get_session_state()
    active_overrides: list[str] = []
    if lags := ss.get("feature_override_lags"):
        features_config["lags"] = sorted(lags)
        active_overrides.append("lags")
    if rw := ss.get("feature_override_rolling"):
        features_config["rolling_windows"] = sorted(rw)
        active_overrides.append("rolling")
    if df_feats := ss.get("feature_override_date"):
        features_config["date_features"] = df_feats
        active_overrides.append("date features")
    if ewm := ss.get("feature_override_ewm"):
        features_config["use_ewm"] = bool(ewm)
        active_overrides.append("ewm")
    if active_overrides:
        details.append(f"User overrides active: {', '.join(active_overrides)}")

    duration = _ts() - t0
    return {
        "feature_config": features_config,
        "step_results": {
            "features": _step(
                "done", f"{len(features_config.get('lags', []))} lags configured", details, duration
            )
        },
        "decision_log": decisions,
        "progress_messages": [_prog("Features configured", "features", "done")],
        "errors": [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 4 — select_model
# ═════════════════════════════════════════════════════════════════════════════


def select_model(state: ForecastGraphState) -> dict:
    """Run ModelSelectorAgent to get a recommended model type."""
    from forecaster.agents.model_selector import ModelSelectorAgent
    from forecaster.core.context import ContextWindow, ResourceBudget

    t0 = _ts()
    prep_df = _get_df(state)
    decisions: list[dict] = []
    details: list[str] = []

    ctx = ContextWindow(
        budget=ResourceBudget(memory_budget_mb=2048, compute_budget_seconds=600.0),
        target_column=state["target_column"],
        datetime_column=state["datetime_column"],
        group_columns=state["group_columns"],
        horizon=state["horizon"],
        gap=state["gap"],
    )
    ctx.register_data("primary", prep_df)

    if dp := state.get("data_profile"):
        from forecaster.core.context import DataProfile

        try:
            ctx.data_profile = DataProfile(**{k: v for k, v in dp.items() if v is not None})
        except Exception:
            pass

    recommended_model = "auto"
    try:
        agent = ModelSelectorAgent()
        ctx, decision = agent.run(ctx)
        decisions.append(decision.model_dump())

        if ctx.model_spec:
            recommended_model = ctx.model_spec.model_type
            details.append(
                f"Recommended: {recommended_model} (confidence: {decision.confidence:.0%})"
            )
            details.append(f"Reasoning: {decision.reasoning}")
            if ctx.model_spec.hyperparameters:
                params = ", ".join(
                    f"{k}={v}" for k, v in list(ctx.model_spec.hyperparameters.items())[:4]
                )
                details.append(f"Suggested params: {params}")

    except Exception as e:
        details.append(f"ModelSelectorAgent warning — will try all models: {e}")
        print(f"[graph/select_model] {e}", file=sys.stderr)

    return {
        "recommended_model": recommended_model,
        "step_results": {
            "model_selection": _step(
                "done", f"Recommended: {recommended_model}", details, _ts() - t0
            )
        },
        "decision_log": decisions,
        "progress_messages": [
            _prog(f"Model selected: {recommended_model}", "model_selection", "done")
        ],
        "errors": [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 5 — train_evaluate
# ═════════════════════════════════════════════════════════════════════════════


def train_evaluate(state: ForecastGraphState) -> dict:
    """
    Train multiple models on a holdout split, optionally tune LightGBM,
    then select the best model by holdout RMSE.
    """
    from forecaster.agents.model_agent import ModelAgent
    from forecaster.agents.workflow_engine import (
        _evaluate_model_on_holdout,
        _tune_lightgbm_hyperparameters,
    )

    t0 = _ts()
    train_df = _get_df(state)
    datetime_column = state["datetime_column"]
    target_column = state["target_column"]
    group_columns = state["group_columns"]
    horizon = state["horizon"]
    gap = state["gap"]
    features_config = state.get("feature_config") or {}
    recommended_model = state.get("recommended_model") or "auto"
    future_exog_df = state.get("future_exog_df")

    details: list[str] = []
    decisions: list[dict] = []

    # ── Determine which models to train ──────────────────────────────────────
    model_agent = ModelAgent()
    available = model_agent.get_available_models()

    MODEL_NAME_MAP = {  # noqa: N806
        "simple_ewm": "naive",
        "lightgbm_default": "lightgbm",
        "mlforecast_global": "lightgbm",
        "prophet": "prophet",
        "naive": "naive",
        "linear": "linear",
    }

    if group_columns:
        # Grouped series: only LightGBM handles multi-group properly
        models_to_train = ["lightgbm"] if "lightgbm" in available else ["naive"]
        details.append("Grouped data — using LightGBM (handles multiple series)")
    else:
        primary = MODEL_NAME_MAP.get(recommended_model, recommended_model)
        models_to_train = [m for m in ["naive", "linear", "lightgbm"] if m in available]
        if primary in available and primary not in models_to_train:
            models_to_train.insert(0, primary)
        details.append(
            f"Agent recommended {recommended_model} — training "
            f"{', '.join(m.upper() for m in models_to_train)} for comparison"
        )

    overrides = get_session_state().get("model_override_hyperparams") or {}
    if overrides:
        features_config = {**features_config, "lgb_params": overrides}
        details.append(f"User hyperparameter overrides: {overrides}")

    # ── Train each model on holdout ───────────────────────────────────────────
    model_results: dict = {}
    trained_models: dict = {}

    for model_name in models_to_train:
        try:
            mr = _evaluate_model_on_holdout(
                model_name,
                train_df,
                datetime_column,
                target_column,
                horizon,
                gap,
                group_columns,
                features_config,
                future_exog_df=future_exog_df,
            )
            model_results[model_name] = mr
            if mr.get("model"):
                trained_models[model_name] = mr["model"]

            rmse = mr.get("holdout_rmse")
            mape = mr.get("holdout_mape")
            rmse_s = (
                f"RMSE={rmse:.2f}" if isinstance(rmse, (int, float)) and rmse < 1e10 else "RMSE=N/A"
            )
            mape_s = f"MAPE={mape:.1f}%" if isinstance(mape, (int, float)) and mape < 1e6 else ""
            details.append(
                f"{model_name.upper()}: {rmse_s} {mape_s} ({mr.get('train_time', 0):.1f}s)"
            )

        except Exception as e:
            details.append(f"{model_name.upper()}: failed — {e}")
            print(f"[graph/train_evaluate] {model_name}: {e}", file=sys.stderr)

    # ── Optional HPO for LightGBM ────────────────────────────────────────────
    if "lightgbm" in trained_models and len(train_df) >= 500:
        try:
            tuned = _tune_lightgbm_hyperparameters(
                train_df,
                datetime_column,
                target_column,
                horizon,
                gap,
                group_columns,
                features_config,
                future_exog_df=future_exog_df,
            )
            default_rmse = model_results.get("lightgbm", {}).get("holdout_rmse", float("inf"))
            tuned_rmse = tuned.get("holdout_rmse", float("inf")) if tuned else float("inf")
            if tuned and tuned_rmse < default_rmse:
                model_results["lightgbm_tuned"] = tuned
                if tuned.get("model"):
                    trained_models["lightgbm_tuned"] = tuned["model"]
                params = ", ".join(
                    f"{k}={v}" for k, v in list(tuned.get("best_params", {}).items())[:3]
                )
                details.append(f"LightGBM (tuned): RMSE={tuned_rmse:.2f} ⭐ ({params})")
            else:
                details.append("HPO did not improve over default LightGBM")
        except Exception as e:
            details.append(f"HPO skipped: {e}")

    # ── Select best model ─────────────────────────────────────────────────────
    best_name: str = ""
    best_rmse = float("inf")
    for name, res in model_results.items():
        rmse = res.get("holdout_rmse")
        if isinstance(rmse, (int, float)) and rmse < best_rmse:
            best_rmse = rmse
            best_name = name

    if not best_name:
        best_name = models_to_train[0] if models_to_train else "naive"

    # Leaderboard
    details.append("")
    details.append(f"Best model: {best_name.upper()}")
    naive_rmse = model_results.get("naive", {}).get("holdout_rmse")
    if (
        isinstance(naive_rmse, (int, float))
        and naive_rmse > 0
        and isinstance(best_rmse, (int, float))
    ):
        improvement = (1 - best_rmse / naive_rmse) * 100
        tag = f"+{improvement:.1f}% vs naive" if improvement > 0 else f"{improvement:.1f}% vs naive"
        details.append(tag)

    duration = _ts() - t0
    return {
        "model_results": model_results,
        "best_model_name": best_name,
        "trained_model": trained_models.get(best_name),
        "step_results": {
            "training": _step(
                "done",
                f"{len(trained_models)} model(s) trained in {duration:.1f}s",
                details,
                duration,
            )
        },
        "decision_log": decisions,
        "progress_messages": [_prog(f"Best model: {best_name.upper()}", "training", "done")],
        "errors": [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 6 — generate_forecast
# ═════════════════════════════════════════════════════════════════════════════


def generate_forecast(state: ForecastGraphState) -> dict:
    """Generate final forecast + diagnostics and produce a ForecastResult."""
    from forecaster.analysis.model_diagnostics import (
        analyze_data_quality,
        calculate_health_score,
        calculate_residuals,
        check_forecast_sanity,
        generate_trust_indicators,
        generate_warnings,
    )
    from forecaster.core.session import ForecastResult
    from forecaster.models.mlforecast_models import MLForecastModel

    t0 = _ts()
    df: pd.DataFrame = state["dataframe"]
    train_df = _get_df(state)
    best_model = state.get("trained_model")
    best_name = state.get("best_model_name") or "naive"
    model_results = state.get("model_results") or {}
    horizon = state["horizon"]
    target_column = state["target_column"]
    datetime_column = state["datetime_column"]
    group_columns = state["group_columns"]
    details: list[str] = []

    if best_model is None:
        return {
            "workflow_status": "failed",
            "step_results": {
                "forecast": _step(
                    "failed",
                    "No trained model available",
                    ["Train at least one model first"],
                    _ts() - t0,
                )
            },
            "progress_messages": [_prog("Forecast failed: no model", "forecast", "failed")],
            "errors": ["No trained model available"],
        }

    # ── Generate predictions ──────────────────────────────────────────────────
    actual_model_type = best_name.replace("_tuned", "")
    group_info: dict | None = None

    try:
        if isinstance(best_model, MLForecastModel):
            pred_df = best_model.predict(horizon)
            predictions: list[float] = pred_df["prediction"].tolist()
            if "datetime" in pred_df.columns:
                dates: list[str] = pred_df["datetime"].astype(str).tolist()
            elif "ds" in pred_df.columns:
                dates = pred_df["ds"].astype(str).tolist()
            else:
                dates = [str(i) for i in range(len(predictions))]

            if group_columns and "unique_id" in pred_df.columns:
                group_info = {
                    "groups": pred_df["unique_id"].unique().tolist(),
                    "n_groups": pred_df["unique_id"].nunique(),
                    "group_columns": group_columns,
                }
        else:
            # Simple models (Naive, Linear)
            proc_df = train_df.copy()
            proc_df[datetime_column] = pd.to_datetime(proc_df[datetime_column], errors="coerce")
            proc_df = proc_df.dropna(subset=[datetime_column]).set_index(datetime_column)
            proc_df = proc_df[[target_column]].rename(columns={target_column: "value"})
            proc_df = proc_df.sort_index().dropna()
            best_model.fit(proc_df)
            simple_result = best_model.predict(horizon)
            predictions = simple_result.predictions
            dates = simple_result.dates

        details.append(f"{len(predictions):,} predictions generated")
        if group_info:
            details.append(f"{group_info['n_groups']} separate group forecasts")

    except Exception as e:
        print(f"[graph/generate_forecast] prediction failed: {e}", file=sys.stderr)
        return {
            "workflow_status": "failed",
            "step_results": {
                "forecast": _step("failed", f"Prediction failed: {e}", [str(e)], _ts() - t0)
            },
            "progress_messages": [_prog(f"Forecast failed: {e}", "forecast", "failed")],
            "errors": [str(e)],
        }

    # ── Diagnostics ───────────────────────────────────────────────────────────
    best_holdout = model_results.get(best_name, {})
    naive_holdout = model_results.get("naive", {})

    rmse_model = best_holdout.get("holdout_rmse", 0) or 0
    rmse_naive = naive_holdout.get("holdout_rmse", 0) or 0
    mape_model = best_holdout.get("holdout_mape", 0) or 0
    mape_naive = naive_holdout.get("holdout_mape", 0) or 0
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

    data_quality = analyze_data_quality(df, datetime_column, target_column)
    historical = df[target_column].dropna().tolist() if target_column in df.columns else []
    sanity = check_forecast_sanity(predictions, historical, target_column)
    residual_analysis = {"is_random": True, "autocorr_lag1": 0.0, "has_trend": False}

    warnings_list = generate_warnings(residual_analysis, real_baseline, data_quality, sanity)
    trust_list = generate_trust_indicators(residual_analysis, real_baseline, data_quality, sanity)
    health_score, _ = calculate_health_score(
        residual_analysis, real_baseline, data_quality, warnings_list
    )

    hs_label = (
        "EXCELLENT"
        if health_score >= 85
        else "GOOD"
        if health_score >= 70
        else "FAIR"
        if health_score >= 50
        else "POOR"
    )
    hs_icon = "🟢" if health_score >= 70 else "🟡" if health_score >= 50 else "🔴"
    details.append(f"{hs_icon} Health Score: {health_score:.0f}/100 ({hs_label})")
    if warnings_list:
        details.append(f"{len(warnings_list)} warning(s)")

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

    # ── Per-group diagnostics ─────────────────────────────────────────────────
    if isinstance(best_model, MLForecastModel) and group_columns and best_model.has_groups():
        try:
            per_group_metrics: dict = {}
            per_group_health: dict = {}
            per_group_warns: dict = {}
            per_group_residuals: dict = {}
            per_group_residual_dates: dict = {}

            for gid in group_info.get("groups", []) if group_info else []:
                try:
                    uid_col = "_uid_tmp"
                    gdf = df.copy()
                    gdf[uid_col] = gdf[group_columns].astype(str).agg("_".join, axis=1)
                    gdf = gdf[gdf[uid_col] == gid].sort_values(datetime_column)
                    actual_vals = (
                        pd.to_numeric(gdf[target_column], errors="coerce").dropna().tolist()
                    )
                    if len(actual_vals) < 20:
                        continue

                    split = int(len(actual_vals) * 0.8)
                    test_act = actual_vals[split:]
                    naive_preds = [actual_vals[split - 1]] * len(test_act)
                    res = calculate_residuals(test_act, naive_preds)

                    test_dates = (
                        gdf[datetime_column].iloc[split:].astype(str).tolist()
                        if datetime_column in gdf.columns
                        else []
                    )
                    per_group_residuals[gid] = res.get("residuals", [])
                    per_group_residual_dates[gid] = test_dates[: len(per_group_residuals[gid])]
                    per_group_metrics[gid] = {
                        "n_observations": len(actual_vals),
                        "mean": float(np.mean(actual_vals)),
                        "std": float(np.std(actual_vals)),
                        **real_baseline,
                    }
                    group_dq = {
                        "total_rows": len(actual_vals),
                        "missing_pct": 0.0,
                        "outliers_pct": 0.0,
                    }
                    gw = generate_warnings(res, real_baseline, group_dq, [])
                    per_group_warns[gid] = gw
                    try:
                        gs, _ = calculate_health_score(res, real_baseline, group_dq, gw)
                        per_group_health[gid] = gs
                    except Exception:
                        per_group_health[gid] = 50.0
                except Exception as ge:
                    print(f"[graph/generate_forecast] per-group '{gid}': {ge}", file=sys.stderr)

            forecast.per_group_metrics = per_group_metrics or None
            forecast.per_group_health_scores = per_group_health or None
            forecast.per_group_warnings = per_group_warns or None
            forecast.per_group_residuals = per_group_residuals or None
            forecast.per_group_residual_dates = per_group_residual_dates or None
        except Exception as pg_err:
            print(f"[graph/generate_forecast] per-group diagnostics: {pg_err}", file=sys.stderr)

    duration = _ts() - t0
    return {
        "forecast_result": forecast,
        "workflow_status": "done",
        "step_results": {
            "forecast": _step("done", f"{len(predictions):,} predictions", details, duration)
        },
        "progress_messages": [
            _prog(
                f"{len(predictions):,} predictions — Health {health_score:.0f}/100",
                "forecast",
                "done",
            )
        ],
        "errors": [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _parse_interrupt_answer(raw_answer: Any, question_keys: list[str]) -> dict:
    """
    Convert the raw resume value from interrupt() into a {key: answer} dict.

    Supports:
      - dict with explicit keys: {"negative_values": "yes", "exog_variables": "no"}
      - str "yes, no" mapped positionally to question_keys
      - str single word applied to all questions
    """
    if isinstance(raw_answer, dict):
        return {k: str(v).strip().lower() for k, v in raw_answer.items()}

    if isinstance(raw_answer, str):
        parts = [p.strip().lower() for p in raw_answer.split(",")]
        result: dict = {}
        for i, key in enumerate(question_keys):
            result[key] = parts[i] if i < len(parts) else parts[-1]
        return result

    return {}
