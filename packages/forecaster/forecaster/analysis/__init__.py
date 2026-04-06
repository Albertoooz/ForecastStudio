"""Analysis utilities for model diagnostics."""

from forecaster.analysis.model_diagnostics import (
    analyze_data_quality,
    calculate_baseline_metrics,
    calculate_health_score,
    calculate_residuals,
    check_forecast_sanity,
    generate_trust_indicators,
    generate_warnings,
)

__all__ = [
    "calculate_residuals",
    "calculate_baseline_metrics",
    "analyze_data_quality",
    "check_forecast_sanity",
    "generate_trust_indicators",
    "calculate_health_score",
    "generate_warnings",
]
