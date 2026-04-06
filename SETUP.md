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

Or use Docker: `make local-up` or `make local-platform` (see `README.md`).

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
