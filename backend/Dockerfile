# ── Forecast Studio backend — multi-stage build with uv ───────────────────

FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies + install ────────────────────────────────────────────────
FROM base AS deps

COPY pyproject.toml uv.lock ./
COPY packages/forecaster/pyproject.toml packages/forecaster/
COPY packages/forecaster/forecaster packages/forecaster/forecaster
COPY backend/pyproject.toml backend/
COPY orchestration/pyproject.toml orchestration/
COPY orchestration/fs_orch orchestration/fs_orch
COPY backend/app backend/app
COPY backend/alembic backend/alembic
COPY backend/alembic.ini backend/alembic.ini
COPY backend/scripts backend/scripts

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package forecast-studio-backend

# ── Runtime ───────────────────────────────────────────────────────────────
FROM base AS runtime

COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /app/pyproject.toml /app/uv.lock ./
COPY --from=deps /app/packages /app/packages
COPY --from=deps /app/backend /app/backend

ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
