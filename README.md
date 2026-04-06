<div align="center">
  <img src="docs/assets/logo.png" alt="Forecast Studio Logo" width="420" />
</div>

# Forecast Studio

Deployable forecasting and analytics platform for teams: multi-agent orchestration (LangGraph), FastAPI backend and Next.js UI, with optional Dagster pipelines, MLflow tracking, and Great Expectations for data quality.

## Quick start (Docker)

```bash
cp .env.example .env
# Set LLM_API_KEY (or legacy DEEPSEEK_API_KEY) and optional LLM_BASE_URL / LLM_MODEL

make local-up          # stack with Langfuse (observability profile)
# or
make local-platform    # app + MLflow + Dagster UI (no Langfuse)
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000 (OpenAPI: `/docs`)
- Langfuse (with `local-up`): http://localhost:3001
- MLflow (`local-dataops` or `local-platform`): http://localhost:5000
- Dagster (`local-platform`): http://localhost:3005

## Quick start (local dev, no Docker for app code)

```bash
uv sync --all-packages --group dev
cp .env.example .env

# Terminal 1 — API
cd backend && uv run uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
cd frontend && npm install && npm run dev
```

Open http://localhost:3000. The frontend expects the API at `http://localhost:8000` (see `NEXT_PUBLIC_API_URL` in compose / frontend env).

## Features

- **File upload** — CSV or Excel time series
- **Chat** — OpenAI-compatible LLM (configure via `LLM_*` env; legacy `DEEPSEEK_*` still supported)
- **Auto-detection** — Date and value columns
- **Forecasting** — Multiple models (naive, linear, LightGBM, Prophet, …)
- **Configuration** — Column selection and forecast horizon

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NEXT.JS FRONTEND                         │
│  Chat · uploads · charts · auth                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              FASTAPI + CELERY (backend)                     │
│  Sessions · jobs · forecaster library                       │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  CHAT AGENT   │  │  DATA AGENT   │  │  MODEL AGENT  │
│ (LLM tools)   │  │ (analysis)    │  │ (training)    │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Usage (UI)

1. **Upload** a CSV with a date column and numeric targets
2. **Configure** date and target columns
3. **Chat** to run forecasts or inspect data
4. **View** charts and tables in the app

### Example chat commands

- "Zrób prognozę na 14 dni"
- "Prognozuj kolumnę sales"
- "Ustaw horyzont na 30 dni"
- "Jakie modele są dostępne?"

## Data format

- A date column (e.g. `date`, `timestamp`)
- One or more numeric columns to forecast

Example:

```csv
date,sales,expenses
2024-01-01,100,50
2024-01-02,120,60
```

## Project structure

```
forecast-studio/
├── packages/forecaster/      # Core ML + LangGraph library
├── backend/                  # FastAPI + Celery
├── frontend/                 # Next.js
├── orchestration/            # Dagster (fs_orch)
├── infra/                    # Azure Bicep
├── pyproject.toml            # uv workspace root
└── uv.lock
```

## Development

### CLI forecaster (no server)

```bash
uv run python -m forecaster.main example_data.csv "Forecast next 7 days"
```

### Tests and formatting

```bash
make test-backend
make lint
cd backend && uv run mypy app/ --ignore-missing-imports
```

### Dagster (local process)

```bash
uv run dagster dev -m fs_orch.definitions
```

### MLflow (Docker)

```bash
make local-dataops
# Tracking URI: http://localhost:5000
```

## Configuration

See [CONFIG.md](./CONFIG.md) for `LLM_*` variables, legacy provider keys, Langfuse, and database settings.

## Agents

The platform uses a multi-agent architecture:

- **Chat agent** — Interprets intent and tool calls
- **Data agent** — Analyzes uploads and columns
- **Model agent** — Selects models and runs forecasts

## Roadmap

- [ ] More models (ARIMA, …)
- [ ] Confidence intervals
- [ ] Export forecasts
- [ ] Session history
- [ ] Multiple time series
- [ ] Feature engineering

## License

MIT
