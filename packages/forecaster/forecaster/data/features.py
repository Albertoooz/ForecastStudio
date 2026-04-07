"""Feature engineering for time series (declarative, reproducible)."""

import numpy as np
import pandas as pd


def create_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_trend: bool = True,
    include_seasonal: bool = True,
) -> pd.DataFrame:
    """
    Create features from time series data.

    All transformations are explicit and reproducible.

    Args:
        df: DataFrame with datetime index and 'value' column
        lags: List of lag periods to include (e.g., [1, 7, 30])
        rolling_windows: List of rolling window sizes (e.g., [7, 30])
        include_trend: Whether to include linear trend
        include_seasonal: Whether to include seasonal features

    Returns:
        DataFrame with original value plus features
    """
    if lags is None:
        lags = [1, 7, 30]
    if rolling_windows is None:
        rolling_windows = [7, 30]

    result = df.copy()

    # Lag features
    for lag in lags:
        result[f"lag_{lag}"] = result["value"].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        result[f"rolling_mean_{window}"] = result["value"].rolling(window=window).mean()
        result[f"rolling_std_{window}"] = result["value"].rolling(window=window).std()

    # Trend
    if include_trend:
        result["trend"] = np.arange(len(result))

    # Seasonal features (day of week, month, etc.)
    if include_seasonal and isinstance(result.index, pd.DatetimeIndex):
        result["day_of_week"] = result.index.dayofweek
        result["month"] = result.index.month
        result["day_of_month"] = result.index.day

    return result
