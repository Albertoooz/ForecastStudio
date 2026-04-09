# Setup Instructions

## Tooling: uv

This repo uses **[uv](https://docs.astral.sh/uv/)** for dependencies and virtualenvs. The root `pyproject.toml` defines a **workspace**; lockfile is `uv.lock`.

```bash
# Install uv (one-time): see https://docs.astral.sh/uv/getting-started/installation/

# Install all workspace members + dev tools
uv sync --all-packages --group dev
```

## Quick start (CLI forecaster)

```bash
uv run python -m forecaster.main example_data.csv "Forecast next 7 days"
```

Run from repo root so the `forecaster` package resolves.

## Web UI (Next.js + FastAPI)

```bash
cp .env.example .env
# Set LLM_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY) — see CONFIG.md

# API
cd backend && uv run uvicorn app.main:app --reload --port 8000

# UI (other terminal)
cd frontend && npm install && npm run dev
```

App: http://localhost:3000 · API docs: http://localhost:8000/docs

Or use Docker: `docker compose up -d` or `make local-up` (with Langfuse — see `README.md`).

### Training stuck in “queued” on the Pipeline page

Training is executed by **Celery**, not the FastAPI process. If you start only `backend`, `postgres`, and `redis`, runs stay **`queued`** until a worker picks them up.

- **Docker:** ensure the `celery-worker` service is running:

  ```bash
  docker compose ps celery-worker
  docker compose up -d celery-worker
  ```

- **Check:** `GET http://localhost:8000/health` — `celery_training_ready` should be `true` and `celery_workers` non-empty when workers are connected to Redis.

- **Local API without Docker:** start a worker in another terminal (same broker URL as in `.env`):

  ```bash
  cd backend && uv run celery -A app.tasks worker --loglevel=info -Q training,forecast,etl
  ```

## Backend only (FastAPI)

```bash
cd backend && uv run uvicorn app.main:app --reload --port 8000
```

## Adding dependencies

```bash
# Add to the relevant member package, e.g. packages/forecaster
uv add --package forecaster some-package

uv lock
```

## Pre-commit

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
