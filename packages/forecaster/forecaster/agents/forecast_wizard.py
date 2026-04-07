"""
Forecast Wizard — Step-by-step guided forecasting workflow.

Instead of blindly training a model, this module:
1. Analyzes data quality
2. Identifies issues and suggests fixes
3. Proposes feature engineering
4. Recommends models with reasoning
5. Trains with full transparency

Every step returns structured recommendations with reasoning.
The UI renders each step and waits for user confirmation before proceeding.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

# =============================================================================
# WIZARD STATE
# =============================================================================

STEPS = ["analysis", "preparation", "features", "model_selection", "training", "complete"]


@dataclass
class WizardStep:
    """A single recommendation within a wizard step."""

    id: str
    description: str
    reasoning: str
    recommendation: str
    options: list[str] = field(default_factory=list)
    selected: str | None = None
    enabled: bool = True  # Whether the user checked/approved this


@dataclass
class WizardState:
    """Full state of the forecast wizard."""

    active: bool = False
    current_step: str = "analysis"

    # Step results
    analysis: dict[str, Any] | None = None
    preparation_steps: list[WizardStep] = field(default_factory=list)
    feature_steps: list[WizardStep] = field(default_factory=list)
    model_recommendation: dict[str, Any] | None = None
    training_result: dict[str, Any] | None = None

    # User choices (persisted across reruns)
    prep_applied: bool = False
    features_applied: bool = False
    pending_question: str | None = None
    question_context: dict[str, Any] = field(default_factory=dict)
    selected_model: str | None = None
    evaluation: dict[str, Any] | None = None


# =============================================================================
# STEP 1: DATA ANALYSIS
# =============================================================================


def analyze_for_wizard(
    df: pl.DataFrame,
    datetime_column: str,
    target_column: str,
    group_cols: list[str],
    frequency: str | None = None,
    gap: int = 0,
) -> dict[str, Any]:
    """
    Comprehensive data analysis for the wizard.

    Returns structured analysis with issues, stats, and recommendations.
    """
    result = {
        "n_rows": df.height,
        "n_columns": df.width,
        "datetime_column": datetime_column,
        "target_column": target_column,
        "group_columns": group_cols,
        "frequency": frequency,
        "gap": gap,
        "issues": [],
        "info": [],
        "stats": {},
        "group_stats": {},
        "column_quality": {},
    }

    # --- Target column analysis ---
    if target_column in df.columns:
        target_series = df[target_column].cast(pl.Float64, strict=False)
        n_missing = int(target_series.null_count())
        n_total = df.height
        missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0
        clean = target_series.drop_nulls()
        n_clean = clean.len()

        result["stats"] = {
            "count": n_clean,
            "missing": n_missing,
            "missing_pct": round(missing_pct, 2),
            "mean": round(float(clean.mean() or 0), 4) if n_clean > 0 else 0,
            "std": round(float(clean.std() or 0), 4) if n_clean > 0 else 0,
            "min": round(float(clean.min() or 0), 4) if n_clean > 0 else 0,
            "max": round(float(clean.max() or 0), 4) if n_clean > 0 else 0,
            "zeros_pct": round(float((clean == 0).sum() / n_clean * 100), 2) if n_clean > 0 else 0,
            "negative_pct": round(float((clean < 0).sum() / n_clean * 100), 2)
            if n_clean > 0
            else 0,
        }

        # Outliers (IQR method)
        if n_clean > 10:
            q1 = float(clean.quantile(0.25) or 0)
            q3 = float(clean.quantile(0.75) or 0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(((clean < lower) | (clean > upper)).sum())
            result["stats"]["outliers"] = n_outliers
            result["stats"]["outliers_pct"] = round(n_outliers / n_clean * 100, 2)
            result["stats"]["outlier_bounds"] = {
                "lower": round(float(lower), 4),
                "upper": round(float(upper), 4),
            }

        # Skewness (manual: m3/m2^1.5)
        if n_clean > 3:
            arr = clean.to_numpy()
            m2 = float(np.mean((arr - arr.mean()) ** 2))
            m3 = float(np.mean((arr - arr.mean()) ** 3))
            sk = m3 / (m2**1.5) if m2 > 0 else 0.0
            result["stats"]["skewness"] = round(sk, 4)

        # Issues from target
        if missing_pct > 0:
            severity = "🔴" if missing_pct > 10 else "⚠️" if missing_pct > 1 else "ℹ️"
            result["issues"].append(
                {
                    "id": "missing_target",
                    "severity": severity,
                    "message": f"Target column has {n_missing:,} missing values ({missing_pct:.1f}%)",
                }
            )

        if result["stats"].get("outliers_pct", 0) > 5:
            result["issues"].append(
                {
                    "id": "outliers",
                    "severity": "⚠️",
                    "message": f"{result['stats']['outliers']:,} outliers detected ({result['stats']['outliers_pct']:.1f}%) using IQR method",
                }
            )

        if result["stats"].get("zeros_pct", 0) > 50:
            result["issues"].append(
                {
                    "id": "high_zeros",
                    "severity": "⚠️",
                    "message": f"Target has {result['stats']['zeros_pct']:.0f}% zero values — sparse/intermittent demand pattern",
                }
            )

        # Negative values (may be invalid for many business metrics)
        if result["stats"].get("negative_pct", 0) > 0:
            result["issues"].append(
                {
                    "id": "negative_values",
                    "severity": "⚠️",
                    "message": f"Target has {result['stats']['negative_pct']:.1f}% negative values",
                }
            )

    # --- Datetime analysis ---
    if datetime_column in df.columns:
        try:
            dates_s = (
                df.sort(datetime_column)[datetime_column]
                .cast(pl.Datetime, strict=False)
                .drop_nulls()
            )
            if dates_s.len() > 1:
                diffs_s = dates_s.diff().drop_nulls().dt.total_seconds()
                if diffs_s.len() > 0:
                    mode_diff = float(diffs_s.mode()[0])
                    n_gaps = int((diffs_s != mode_diff).sum())
                    gap_pct = round(n_gaps / diffs_s.len() * 100, 2)
                    result["stats"]["frequency_gaps"] = n_gaps
                    result["stats"]["frequency_gap_pct"] = gap_pct
                    if gap_pct > 5:
                        result["issues"].append(
                            {
                                "id": "frequency_gaps",
                                "severity": "⚠️",
                                "message": f"{n_gaps:,} irregular time gaps detected ({gap_pct:.1f}% of intervals)",
                            }
                        )

            # Duplicates
            if group_cols:
                cols = [datetime_column] + [c for c in group_cols if c in df.columns]
                n_dup = (
                    df.height
                    - df.select(
                        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cols], separator="_")
                    ).n_unique()
                )
                if n_dup > 0:
                    result["issues"].append(
                        {
                            "id": "duplicate_timestamps",
                            "severity": "⚠️",
                            "message": f"{n_dup:,} duplicate timestamps within groups",
                        }
                    )
            else:
                n_dup = df.height - df.select(pl.col(datetime_column)).n_unique()
                if n_dup > 0:
                    result["issues"].append(
                        {
                            "id": "duplicate_timestamps",
                            "severity": "⚠️",
                            "message": f"{n_dup:,} duplicate timestamps found",
                        }
                    )

            last_date = dates_s.max()
            mn = dates_s.min()
            if last_date is not None and mn is not None:
                if not isinstance(last_date, datetime):
                    last_date = datetime.fromisoformat(str(last_date)[:19])
                if not isinstance(mn, datetime):
                    mn = datetime.fromisoformat(str(mn)[:19])
                data_age = (datetime.now() - last_date).days
                result["stats"]["data_age_days"] = int(data_age)
                result["stats"]["date_range"] = f"{str(mn)[:10]} → {str(last_date)[:10]}"
                if data_age > 30:
                    result["issues"].append(
                        {
                            "id": "stale_data",
                            "severity": "🔴",
                            "message": f"Data is {data_age} days old — model may not reflect current patterns",
                        }
                    )
        except Exception:
            result["issues"].append(
                {
                    "id": "datetime_parse_error",
                    "severity": "🔴",
                    "message": f"Could not parse datetime column '{datetime_column}'",
                }
            )

    # --- Group analysis ---
    if group_cols and all(c in df.columns for c in group_cols):
        uid_expr = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in group_cols], separator="_")
        group_counts_s = (
            df.select(uid_expr.alias("__uid")).group_by("__uid").agg(pl.len().alias("n"))["n"]
        )
        n_groups = int(group_counts_s.len())
        mn_g = int(group_counts_s.min() or 0)
        mx_g = int(group_counts_s.max() or 0)
        med_g = int(float(group_counts_s.median() or 0))
        result["group_stats"] = {
            "n_groups": n_groups,
            "min_group_size": mn_g,
            "max_group_size": mx_g,
            "median_group_size": med_g,
        }
        small_groups = int((group_counts_s < 30).sum())
        if small_groups > 0:
            result["issues"].append(
                {
                    "id": "small_groups",
                    "severity": "⚠️",
                    "message": f"{small_groups} group(s) have fewer than 30 observations — forecasts may be unreliable",
                }
            )

    # --- Sample size check ---
    if df.height < 50:
        result["issues"].append(
            {
                "id": "small_dataset",
                "severity": "🔴",
                "message": f"Only {df.height} rows — insufficient for reliable ML forecasting",
            }
        )

    # --- Column quality for all numeric columns ---
    for col in df.columns:
        if df[col].dtype.is_numeric():
            n_miss = int(df[col].null_count())
            if n_miss > 0:
                result["column_quality"][col] = {
                    "missing": n_miss,
                    "missing_pct": round(n_miss / df.height * 100, 2),
                }

    # --- Info (positive signals) ---
    if not result["issues"]:
        result["info"].append("✅ No data quality issues detected")
    if df.height >= 1000:
        result["info"].append(f"✅ Good sample size ({df.height:,} observations)")
    if frequency:
        result["info"].append(f"✅ Consistent frequency detected: {frequency}")

    return result


# =============================================================================
# STEP 2: DATA PREPARATION SUGGESTIONS
# =============================================================================


def suggest_preparations(analysis: dict[str, Any]) -> list[WizardStep]:
    """
    Based on analysis, suggest data preparation steps.
    Each step can be toggled on/off by the user.
    """
    steps = []
    issues = {i["id"]: i for i in analysis.get("issues", [])}
    stats = analysis.get("stats", {})

    # Missing values
    if "missing_target" in issues:
        missing_pct = stats.get("missing_pct", 0)
        if missing_pct > 20:
            rec = "drop"
            reasoning = f"With {missing_pct:.1f}% missing, imputation would introduce too much noise. Dropping missing rows is safer."
        elif missing_pct > 5:
            rec = "interpolate"
            reasoning = f"With {missing_pct:.1f}% missing, linear interpolation preserves the time series trend."
        else:
            rec = "forward_fill"
            reasoning = f"With only {missing_pct:.1f}% missing, forward fill (last known value) is simple and effective."

        steps.append(
            WizardStep(
                id="fill_missing",
                description=f"Handle {stats.get('missing', 0):,} missing values in target ({missing_pct:.1f}%)",
                reasoning=reasoning,
                recommendation=rec,
                options=["forward_fill", "interpolate", "drop", "skip"],
                selected=rec,
                enabled=True,
            )
        )

    # Outliers
    if "outliers" in issues:
        n_out = stats.get("outliers", 0)
        out_pct = stats.get("outliers_pct", 0)
        bounds = stats.get("outlier_bounds", {})
        zeros_pct = stats.get("zeros_pct", 0)

        # CRITICAL FIX: Don't clip if bounds are degenerate (sparse data)
        lower = bounds.get("lower", 0)
        upper = bounds.get("upper", 0)
        bounds_range = upper - lower

        # If sparse data (>50% zeros) or degenerate bounds, keep outliers
        if zeros_pct > 50 or bounds_range < 0.01 or lower == upper:
            rec = "keep"
            if zeros_pct > 50:
                reasoning = f"Data is sparse ({zeros_pct:.0f}% zeros) — clipping would destroy signal. LightGBM handles sparse data well."
            else:
                reasoning = f"IQR bounds too narrow (range={bounds_range:.2f}) — clipping would destroy variance. Keep outliers."
        elif out_pct > 20:
            rec = "remove"
            reasoning = f"With {out_pct:.1f}% outliers, removing extreme values (below {lower:.2f} or above {upper:.2f}) may help."
        elif out_pct > 10:
            rec = "clip"
            reasoning = f"With {out_pct:.1f}% outliers, clipping to IQR bounds [{lower:.2f}, {upper:.2f}] is safer than removal."
        else:
            rec = "keep"
            reasoning = f"Only {out_pct:.1f}% outliers — the model should handle these, especially LightGBM which is robust to outliers."

        steps.append(
            WizardStep(
                id="handle_outliers",
                description=f"Handle {n_out:,} outliers ({out_pct:.1f}%)",
                reasoning=reasoning,
                recommendation=rec,
                options=["clip", "remove", "keep"],
                selected=rec,
                enabled=(rec != "keep"),
            )
        )

    # Duplicate timestamps
    if "duplicate_timestamps" in issues:
        steps.append(
            WizardStep(
                id="handle_duplicates",
                description="Handle duplicate timestamps",
                reasoning="Duplicate timestamps can confuse the model. Aggregating (mean) is the safest approach.",
                recommendation="aggregate_mean",
                options=["aggregate_mean", "keep_last", "skip"],
                selected="aggregate_mean",
                enabled=True,
            )
        )

    # Frequency gaps
    if "frequency_gaps" in issues:
        gap_pct = stats.get("frequency_gap_pct", 0)
        if gap_pct > 10:
            steps.append(
                WizardStep(
                    id="fix_frequency",
                    description=f"Resample to fill {stats.get('frequency_gaps', 0):,} time gaps",
                    reasoning="Large gaps in the time series can cause the model to learn incorrect patterns. Resampling fills these gaps.",
                    recommendation="resample_ffill",
                    options=["resample_ffill", "resample_interpolate", "skip"],
                    selected="resample_ffill",
                    enabled=True,
                )
            )

    # If nothing to do
    if not steps:
        steps.append(
            WizardStep(
                id="no_prep_needed",
                description="No data preparation needed",
                reasoning="Data looks clean — no missing values, outliers, or frequency issues detected.",
                recommendation="continue",
                options=["continue"],
                selected="continue",
                enabled=False,
            )
        )

    return steps


# =============================================================================
# STEP 3: FEATURE ENGINEERING SUGGESTIONS
# =============================================================================


def suggest_features(
    analysis: dict[str, Any],
    frequency: str | None = None,
    gap: int = 0,
) -> list[WizardStep]:
    """
    Suggest feature engineering steps based on data properties.
    """
    steps = []
    analysis.get("stats", {})
    n_rows = analysis.get("n_rows", 0)
    analysis.get("group_stats", {})

    # --- Lag features ---
    if gap == 0:
        default_lags = [1, 7, 14]
        lag_reasoning = "Lag features capture recent patterns. Lag-1 is the most recent value, lag-7 captures weekly seasonality."
    else:
        default_lags = [gap + 1, gap + 7, gap + 14, gap + 28]
        lag_reasoning = (
            f"With a gap of {gap}, lags 1–{gap} are unavailable at forecast time. "
            f"Starting from lag {gap + 1} (first available value)."
        )

    lag_str = ", ".join(str(l) for l in default_lags)  # noqa: E741
    steps.append(
        WizardStep(
            id="lag_features",
            description=f"Add lag features: [{lag_str}]",
            reasoning=lag_reasoning,
            recommendation="auto",
            options=["auto", "custom", "skip"],
            selected="auto",
            enabled=True,
        )
    )

    # --- Rolling statistics ---
    if n_rows >= 30:
        windows = [7, 14] if gap == 0 else [max(7, gap + 1), max(14, gap + 7)]
        win_str = ", ".join(str(w) for w in windows)
        steps.append(
            WizardStep(
                id="rolling_stats",
                description=f"Add rolling mean/std (windows: [{win_str}])",
                reasoning="Rolling statistics smooth out noise and capture local trends. Mean captures level, std captures volatility.",
                recommendation="auto",
                options=["auto", "skip"],
                selected="auto",
                enabled=True,
            )
        )

    # --- Date features ---
    date_features = []
    if frequency:
        # Sub-daily: hour features are important
        if "min" in frequency or "h" in frequency:
            date_features = ["hour", "dayofweek", "month", "is_weekend"]
        elif frequency in ("D", "1D"):
            date_features = ["dayofweek", "month", "dayofyear", "is_weekend"]
        elif frequency in ("W",):
            date_features = ["month", "weekofyear", "quarter"]
        elif frequency in ("M",):
            date_features = ["month", "quarter"]
        else:
            date_features = ["dayofweek", "month"]
    else:
        date_features = ["dayofweek", "month"]

    feat_str = ", ".join(date_features)
    steps.append(
        WizardStep(
            id="date_features",
            description=f"Add date features: [{feat_str}]",
            reasoning="Calendar features help the model learn seasonal patterns (weekday vs weekend, summer vs winter, etc.)",
            recommendation="auto",
            options=["auto", "skip"],
            selected="auto",
            enabled=True,
        )
    )

    return steps


# =============================================================================
# STEP 4: MODEL SELECTION
# =============================================================================


def recommend_model(
    analysis: dict[str, Any],
    available_models: list[str],
    frequency: str | None = None,
) -> dict[str, Any]:
    """
    Recommend a model based on data properties.

    Returns structured recommendation with reasoning.
    """
    n_rows = analysis.get("n_rows", 0)
    n_groups = analysis.get("group_stats", {}).get("n_groups", 1)
    stats = analysis.get("stats", {})

    candidates = []

    # LightGBM
    if "lightgbm" in available_models:
        score = 0
        reasons = []

        if n_rows >= 100:
            score += 3
            reasons.append(f"sufficient data ({n_rows:,} rows)")
        if n_groups > 1:
            score += 2
            reasons.append(f"handles {n_groups} groups well")
        if stats.get("outliers_pct", 0) > 2:
            score += 1
            reasons.append("robust to outliers")

        score += 2  # General preference for flexibility
        reasons.append("flexible, fast, handles non-linear patterns")

        candidates.append(
            {
                "model": "lightgbm",
                "label": "LightGBM (Gradient Boosting)",
                "score": score,
                "reasons": reasons,
                "description": "Fast gradient boosting. Best for medium-large datasets, handles non-linear patterns and outliers well.",
                "hyperparameter_tuning": n_rows >= 500,
            }
        )

    # Linear
    if "linear" in available_models:
        score = 0
        reasons = []

        if n_rows < 100:
            score += 3
            reasons.append("simple model for small data")
        if stats.get("outliers_pct", 0) < 2:
            score += 1
            reasons.append("clean data suits linear models")

        score += 1
        reasons.append("interpretable baseline")

        candidates.append(
            {
                "model": "linear",
                "label": "Linear Regression",
                "score": score,
                "reasons": reasons,
                "description": "Simple linear model. Good baseline, fast, interpretable. Best for small or clean datasets.",
                "hyperparameter_tuning": False,
            }
        )

    # Prophet
    if "prophet" in available_models:
        score = 0
        reasons = []

        if frequency and frequency in ("D", "W", "M"):
            score += 2
            reasons.append(f"designed for {frequency} frequency")
        if n_groups <= 1:
            score += 1
            reasons.append("best for single series")

        score += 1
        reasons.append("handles seasonality and holidays automatically")

        candidates.append(
            {
                "model": "prophet",
                "label": "Prophet (Meta)",
                "score": score,
                "reasons": reasons,
                "description": "Meta's Prophet. Handles seasonality, holidays, trend changes. Best for daily/weekly business data.",
                "hyperparameter_tuning": False,
            }
        )

    # Naive (always available)
    if "naive" in available_models:
        candidates.append(
            {
                "model": "naive",
                "label": "Naive Baseline",
                "score": 0,
                "reasons": ["simple benchmark — repeats last value"],
                "description": "Repeats the last observed value. Use as a baseline to compare other models against.",
                "hyperparameter_tuning": False,
            }
        )

    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    recommended = candidates[0] if candidates else None

    return {
        "recommended": recommended,
        "candidates": candidates,
        "reasoning": f"Based on {n_rows:,} rows, {n_groups} group(s), and data quality analysis.",
    }


# =============================================================================
# MODEL EVALUATION (LIGHTWEIGHT BACKTEST)
# =============================================================================


def evaluate_models(
    df: pl.DataFrame,
    datetime_column: str,
    target_column: str,
    group_cols: list[str] | None,
    candidate_models: list[str],
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """
    Evaluate multiple models on a simple holdout split (train/test).
    Returns best model and metrics.
    """
    results = {}
    best_model = None
    best_rmse = None

    if datetime_column not in df.columns or target_column not in df.columns:
        return {"best_model": None, "results": {}, "error": "Missing datetime/target column"}

    eval_df = df.clone()
    if group_cols:
        eval_df = eval_df.group_by(datetime_column).agg(pl.col(target_column).sum())

    eval_df = (
        eval_df.sort(datetime_column)
        .with_columns(pl.col(target_column).cast(pl.Float64, strict=False))
        .drop_nulls(target_column)
    )
    n = eval_df.height
    if n < 20:
        return {"best_model": None, "results": {}, "error": "Not enough data for evaluation"}

    split = int(n * (1 - test_fraction))
    train_df = eval_df.slice(0, split)
    test_df = eval_df.slice(split)

    train_processed = train_df.select([datetime_column, pl.col(target_column).alias("value")])
    test_values = test_df[target_column].to_numpy()
    if len(test_values) == 0:
        return {"best_model": None, "results": {}, "error": "Test set has no valid values"}

    horizon = len(test_values)

    for model_name in candidate_models:
        try:
            if model_name == "naive":
                from forecaster.models.simple import NaiveForecaster

                model = NaiveForecaster()
                model.fit(train_processed)
                preds = model.predict(horizon).predictions
            elif model_name == "linear":
                from forecaster.models.simple import LinearForecaster

                model = LinearForecaster()
                model.fit(train_processed)
                preds = model.predict(horizon).predictions
            elif model_name == "lightgbm":
                from forecaster.models.mlforecast_models import MLForecastModel

                model = MLForecastModel()
                model.fit(train_df, datetime_column, target_column, group_by_columns=None)
                forecast = model.predict(horizon)
                preds = (
                    forecast["prediction"].to_list()
                    if isinstance(forecast, pl.DataFrame)
                    else forecast.predictions
                )
            else:
                continue

            if len(preds) != horizon:
                preds = preds[:horizon]

            rmse = float(np.sqrt(np.mean((test_values - np.array(preds)) ** 2)))
            mape = float(
                np.mean(np.abs((test_values - np.array(preds)) / (test_values + 1e-10))) * 100
            )

            results[model_name] = {"rmse": rmse, "mape": mape}
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
        except Exception:
            continue

    return {"best_model": best_model, "results": results, "error": None}


# =============================================================================
# STEP 2 EXECUTION: Apply preparation steps
# =============================================================================


def apply_preparation(
    df: pl.DataFrame,
    steps: list[WizardStep],
    datetime_column: str,
    target_column: str,
    frequency: str | None = None,
    group_cols: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Apply selected preparation steps to the Polars DataFrame.

    Returns (modified_df, log_messages).
    """
    log: list[str] = []

    for step in steps:
        if not step.enabled:
            continue

        choice = step.selected or step.recommendation

        if step.id == "fill_missing" and choice != "skip":
            if target_column not in df.columns:
                continue
            before_missing = int(df[target_column].null_count())
            if choice == "forward_fill":
                df = df.with_columns(pl.col(target_column).forward_fill())
                method = "forward fill"
            elif choice == "interpolate":
                df = df.with_columns(pl.col(target_column).interpolate())
                method = "linear interpolation"
            elif choice == "drop":
                df = df.drop_nulls(target_column)
                method = "drop rows"
            else:
                continue
            after_missing = int(df[target_column].null_count())
            log.append(
                f"✅ Filled {before_missing - after_missing:,} missing values using {method}"
            )

        elif step.id == "handle_outliers" and choice != "keep":
            if target_column in df.columns:
                clean = df[target_column].cast(pl.Float64, strict=False).drop_nulls()
                if clean.len() > 10:
                    q1 = float(clean.quantile(0.25) or 0)
                    q3 = float(clean.quantile(0.75) or 0)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    if lower == upper or abs(upper - lower) < 0.01:
                        log.append(
                            f"⚠️ Skipped outlier handling — IQR bounds too narrow ({lower:.2f}, {upper:.2f})"
                        )
                        continue
                    if choice == "clip":
                        n_clipped = int(
                            ((df[target_column] < lower) | (df[target_column] > upper)).sum()
                        )
                        if n_clipped > 0:
                            df = df.with_columns(
                                pl.col(target_column).clip(lower_bound=lower, upper_bound=upper)
                            )
                            log.append(
                                f"✅ Clipped {n_clipped:,} outliers to [{lower:.2f}, {upper:.2f}]"
                            )
                        else:
                            log.append(
                                f"ℹ️ No outliers to clip (bounds: [{lower:.2f}, {upper:.2f}])"
                            )
                    elif choice == "remove":
                        before = df.height
                        df = df.filter(
                            (pl.col(target_column) >= lower) & (pl.col(target_column) <= upper)
                        )
                        removed = before - df.height
                        if removed > 0:
                            log.append(f"✅ Removed {removed:,} outlier rows")
                        else:
                            log.append("ℹ️ No outlier rows to remove")

        elif step.id == "handle_duplicates" and choice != "skip":
            if datetime_column in df.columns:
                before = df.height
                grp = group_cols or []
                if choice == "aggregate_mean":
                    agg_cols = [datetime_column] + [c for c in grp if c in df.columns]
                    num_cols = [
                        c for c in df.columns if c not in agg_cols and df[c].dtype.is_numeric()
                    ]
                    df = (
                        df.group_by(agg_cols)
                        .agg([pl.col(c).mean() for c in num_cols])
                        .sort(datetime_column)
                    )
                elif choice == "keep_last":
                    subset_cols = [datetime_column] + [c for c in grp if c in df.columns]
                    df = df.unique(subset=subset_cols, keep="last")
                log.append(f"✅ Handled duplicates: {before:,} → {df.height:,} rows")

    return df, log
