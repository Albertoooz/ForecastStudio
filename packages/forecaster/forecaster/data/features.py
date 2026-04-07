"""Feature engineering for time series (declarative, reproducible) — Polars."""

import polars as pl


def create_features(
    df: pl.DataFrame,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    include_trend: bool = True,
    include_seasonal: bool = True,
) -> pl.DataFrame:
    """
    Create features from time series data.

    All transformations are explicit and reproducible.

    Args:
        df: Polars DataFrame with a temporal column and 'value' column
        lags: List of lag periods to include (e.g., [1, 7, 30])
        rolling_windows: List of rolling window sizes (e.g., [7, 30])
        include_trend: Whether to include a linear trend counter
        include_seasonal: Whether to include seasonal (date-part) features

    Returns:
        DataFrame with original columns plus engineered features
    """
    if lags is None:
        lags = [1, 7, 30]
    if rolling_windows is None:
        rolling_windows = [7, 30]

    exprs: list[pl.Expr] = []

    # Lag features
    for lag in lags:
        exprs.append(pl.col("value").shift(lag).alias(f"lag_{lag}"))

    # Rolling statistics
    for window in rolling_windows:
        exprs.append(
            pl.col("value")
            .rolling_mean(window_size=window, min_periods=1)
            .alias(f"rolling_mean_{window}")
        )
        exprs.append(
            pl.col("value")
            .rolling_std(window_size=window, min_periods=1)
            .alias(f"rolling_std_{window}")
        )

    if include_trend:
        exprs.append(pl.int_range(0, df.height, eager=False).alias("trend"))

    result = df.with_columns(exprs)

    if include_seasonal:
        dt_col = next(
            (c for c in df.columns if df[c].dtype.is_temporal()),
            None,
        )
        if dt_col is not None:
            result = result.with_columns(
                [
                    pl.col(dt_col).cast(pl.Datetime).dt.weekday().alias("day_of_week"),
                    pl.col(dt_col).cast(pl.Datetime).dt.month().alias("month"),
                    pl.col(dt_col).cast(pl.Datetime).dt.day().alias("day_of_month"),
                ]
            )

    return result
