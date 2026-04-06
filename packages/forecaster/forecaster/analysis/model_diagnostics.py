"""
Model diagnostics and health analysis for forecasting models.

Implements state-of-the-art time series model analysis:
- Residual analysis
- Baseline comparison
- Data quality checks
- Automatic warnings
- Health score calculation
"""

from typing import Any

import numpy as np
import pandas as pd


def calculate_residuals(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    """
    Calculate residuals and analyze them for patterns.

    Returns:
        Dict with residuals, statistics, and warnings
    """
    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)
    residuals = actual_arr - predicted_arr

    # Basic statistics
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))

    # Check for autocorrelation (simple lag-1)
    if len(residuals) > 1:
        autocorr_lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        autocorr_lag1 = 0.0

    # Detect trend in residuals (should be flat)
    has_trend = False
    if len(residuals) >= 10:
        x = np.arange(len(residuals))
        slope = float(np.polyfit(x, residuals, 1)[0])
        # Significant if slope > 0.1 * std per period
        if abs(slope) > 0.1 * std_residual:
            has_trend = True

    # Check if residuals look random (Ljung-Box approximation)
    is_random = abs(autocorr_lag1) < 0.2 and not has_trend

    return {
        "residuals": residuals.tolist(),
        "mean": mean_residual,
        "std": std_residual,
        "autocorr_lag1": autocorr_lag1,
        "has_trend": has_trend,
        "is_random": is_random,
    }


def calculate_baseline_metrics(
    actual: list[float], predictions: list[float], naive_predictions: list[float]
) -> dict[str, float]:
    """
    Compare model performance against naive baseline.

    Args:
        actual: Actual values
        predictions: Model predictions
        naive_predictions: Naive forecast (last value repeated)

    Returns:
        Dict with model metrics, baseline metrics, and improvement
    """
    actual_arr = np.array(actual)
    pred_arr = np.array(predictions)
    naive_arr = np.array(naive_predictions)

    # Model metrics
    mae_model = float(np.mean(np.abs(actual_arr - pred_arr)))
    rmse_model = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))
    mape_model = float(np.mean(np.abs((actual_arr - pred_arr) / (actual_arr + 1e-10))) * 100)

    # Baseline metrics
    mae_naive = float(np.mean(np.abs(actual_arr - naive_arr)))
    rmse_naive = float(np.sqrt(np.mean((actual_arr - naive_arr) ** 2)))
    mape_naive = float(np.mean(np.abs((actual_arr - naive_arr) / (actual_arr + 1e-10))) * 100)

    # Improvement
    rmse_improvement = (
        float(((rmse_naive - rmse_model) / rmse_naive) * 100) if rmse_naive > 0 else 0.0
    )
    mape_improvement = (
        float(((mape_naive - mape_model) / mape_naive) * 100) if mape_naive > 0 else 0.0
    )

    return {
        "mae_model": mae_model,
        "rmse_model": rmse_model,
        "mape_model": mape_model,
        "mae_naive": mae_naive,
        "rmse_naive": rmse_naive,
        "mape_naive": mape_naive,
        "rmse_improvement_pct": rmse_improvement,
        "mape_improvement_pct": mape_improvement,
        "beats_baseline": rmse_improvement > 0,
    }


def analyze_data_quality(
    df: pd.DataFrame, datetime_column: str, target_column: str
) -> dict[str, Any]:
    """
    Analyze training data quality.

    Returns:
        Dict with quality metrics and issues
    """
    quality = {}

    # Missing values
    total_rows = len(df)
    missing_target = df[target_column].isna().sum()
    missing_pct = float((missing_target / total_rows) * 100) if total_rows > 0 else 0.0

    quality["total_rows"] = total_rows
    quality["missing_count"] = int(missing_target)
    quality["missing_pct"] = missing_pct

    # Outliers (Z-score > 3)
    if target_column in df.columns:
        values = df[target_column].dropna()
        if len(values) > 0:
            mean_val = float(values.mean())
            std_val = float(values.std())
            if std_val > 0:
                z_scores = np.abs((values - mean_val) / std_val)
                outliers = int((z_scores > 3).sum())
                quality["outliers_count"] = outliers
                quality["outliers_pct"] = float((outliers / len(values)) * 100)
            else:
                quality["outliers_count"] = 0
                quality["outliers_pct"] = 0.0

    # Data recency
    if datetime_column in df.columns:
        try:
            df_sorted = df.sort_values(datetime_column)
            last_date = pd.to_datetime(df_sorted[datetime_column].iloc[-1])
            now = pd.Timestamp.now()
            days_old = (now - last_date).days
            quality["data_age_days"] = int(days_old)
            quality["last_data_date"] = str(last_date.date())
        except Exception:
            quality["data_age_days"] = None

    # Basic statistics
    if target_column in df.columns:
        values = df[target_column].dropna()
        if len(values) > 0:
            quality["mean"] = float(values.mean())
            quality["std"] = float(values.std())
            quality["min"] = float(values.min())
            quality["max"] = float(values.max())

    return quality


def check_forecast_sanity(
    predictions: list[float], historical_values: list[float], target_column: str
) -> list[str]:
    """
    Sanity checks for forecast values.

    Returns:
        List of warnings
    """
    warnings = []

    pred_arr = np.array(predictions)
    hist_arr = np.array(historical_values)

    # Check for negative values in non-negative metrics
    non_negative_keywords = ["count", "quantity", "volume", "sales", "revenue", "price"]
    is_non_negative = any(kw in target_column.lower() for kw in non_negative_keywords)

    if is_non_negative and np.any(pred_arr < 0):
        neg_count = int(np.sum(pred_arr < 0))
        warnings.append(f"⚠️ Forecast contains {neg_count} negative values for non-negative metric")

    # Check if forecast exceeds historical range significantly
    hist_min, hist_max = float(hist_arr.min()), float(hist_arr.max())
    pred_min, pred_max = float(pred_arr.min()), float(pred_arr.max())
    hist_range = hist_max - hist_min

    if pred_max > hist_max + 0.5 * hist_range:
        exceed_pct = float(((pred_max - hist_max) / hist_range) * 100)
        warnings.append(f"⚠️ Forecast maximum exceeds historical range by {exceed_pct:.1f}%")

    if pred_min < hist_min - 0.5 * hist_range:
        exceed_pct = float(((hist_min - pred_min) / hist_range) * 100)
        warnings.append(f"⚠️ Forecast minimum below historical range by {exceed_pct:.1f}%")

    # Check for extreme changes
    if len(hist_arr) > 0:
        last_actual = float(hist_arr[-1])
        first_forecast = float(pred_arr[0])
        if last_actual != 0:
            change_pct = float(((first_forecast - last_actual) / abs(last_actual)) * 100)
            if abs(change_pct) > 50:
                warnings.append(f"⚠️ Forecast shows {change_pct:.1f}% change from current level")

    return warnings


def generate_trust_indicators(
    residual_analysis: dict[str, Any],
    baseline_metrics: dict[str, float],
    data_quality: dict[str, Any],
    forecast_warnings: list[str],
) -> list[str]:
    """
    Generate positive trust indicators.

    Returns:
        List of trust signals
    """
    indicators = []

    # Model performance
    if baseline_metrics.get("beats_baseline", False):
        improvement = baseline_metrics.get("rmse_improvement_pct", 0)
        indicators.append(f"✅ Model beats naive baseline by {improvement:.1f}%")

    # Residuals
    if residual_analysis.get("is_random", False):
        indicators.append("✅ Residuals appear random (good fit)")

    # Data quality
    missing_pct = data_quality.get("missing_pct", 0)
    if missing_pct < 5:
        indicators.append(f"✅ High data quality ({100 - missing_pct:.1f}% complete)")

    data_age = data_quality.get("data_age_days")
    if data_age is not None and data_age < 7:
        indicators.append(f"✅ Training data is fresh ({data_age} days old)")

    # Forecast sanity
    if len(forecast_warnings) == 0:
        indicators.append("✅ Forecast passes all sanity checks")

    return indicators


def calculate_health_score(
    residual_analysis: dict[str, Any],
    baseline_metrics: dict[str, float],
    data_quality: dict[str, Any],
    warnings: list[str],
) -> tuple[float, dict[str, float]]:
    """
    Calculate overall model health score (0-100).

    Returns:
        (score, component_scores)
    """
    components = {}

    # Residuals (25 points)
    if residual_analysis.get("is_random", False):
        components["residuals"] = 25.0
    elif abs(residual_analysis.get("autocorr_lag1", 0)) < 0.4:
        components["residuals"] = 15.0
    else:
        components["residuals"] = 5.0

    # Baseline comparison (25 points)
    if baseline_metrics.get("beats_baseline", False):
        improvement = baseline_metrics.get("rmse_improvement_pct", 0)
        components["baseline"] = min(25.0, 10.0 + improvement / 2)
    else:
        components["baseline"] = 0.0

    # Data quality (25 points)
    missing_pct = data_quality.get("missing_pct", 100)
    outliers_pct = data_quality.get("outliers_pct", 10)
    data_age = data_quality.get("data_age_days", 365)

    quality_score = 0.0
    if missing_pct < 5:
        quality_score += 10.0
    elif missing_pct < 15:
        quality_score += 5.0

    if outliers_pct < 5:
        quality_score += 5.0
    elif outliers_pct < 10:
        quality_score += 2.0

    if data_age < 7:
        quality_score += 10.0
    elif data_age < 30:
        quality_score += 5.0

    components["data_quality"] = quality_score

    # Warnings (25 points)
    warning_penalty = min(25.0, len(warnings) * 5.0)
    components["warnings"] = max(0.0, 25.0 - warning_penalty)

    total_score = sum(components.values())

    return float(total_score), components


def generate_warnings(
    residual_analysis: dict[str, Any],
    baseline_metrics: dict[str, float],
    data_quality: dict[str, Any],
    forecast_warnings: list[str],
) -> list[str]:
    """
    Generate all automatic warnings.

    Returns:
        List of warnings
    """
    warnings = []

    # Model warnings
    if not baseline_metrics.get("beats_baseline", False):
        warnings.append("🔴 Model does NOT beat naive baseline")

    if not residual_analysis.get("is_random", False):
        if abs(residual_analysis.get("autocorr_lag1", 0)) > 0.4:
            warnings.append("⚠️ Residuals show autocorrelation (model may be underfitting)")
        if residual_analysis.get("has_trend", False):
            warnings.append("⚠️ Residuals show trend (model is missing pattern)")

    # Data warnings
    missing_pct = data_quality.get("missing_pct", 0)
    if missing_pct > 10:
        warnings.append(f"⚠️ {missing_pct:.1f}% of training data was missing")

    outliers_pct = data_quality.get("outliers_pct", 0)
    if outliers_pct > 5:
        warnings.append(f"⚠️ {outliers_pct:.1f}% of data are outliers")

    data_age = data_quality.get("data_age_days")
    if data_age is not None:
        if data_age > 30:
            warnings.append(f"🔴 Training data is {data_age} days old - consider refreshing")
        elif data_age > 7:
            warnings.append(f"⚠️ Training data is {data_age} days old")

    total_rows = data_quality.get("total_rows", 0)
    if total_rows < 50:
        warnings.append("⚠️ Insufficient data for reliable forecast (need more historical data)")

    # Forecast warnings
    warnings.extend(forecast_warnings)

    return warnings
