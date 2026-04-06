"""Dagster definitions — extend with assets and jobs as the platform grows."""

from dagster import Definitions, asset


@asset(group_name="forecast_studio")
def health_check() -> str:
    """Placeholder asset; replace with real data and training pipelines."""
    return "ok"


defs = Definitions(assets=[health_check])
