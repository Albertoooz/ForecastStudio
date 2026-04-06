# Dagster webserver — code location fs_orch (Forecast Studio orchestration)

FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/dagster

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY packages/forecaster/pyproject.toml packages/forecaster/
COPY packages/forecaster/forecaster packages/forecaster/forecaster
COPY orchestration/pyproject.toml orchestration/
COPY orchestration/fs_orch orchestration/fs_orch

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package forecast-studio-orchestration

RUN mkdir -p /dagster

ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv

EXPOSE 3000

CMD ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000", "-m", "fs_orch.definitions"]
