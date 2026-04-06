# Configuration Guide

## Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

### LLM (OpenAI-compatible SDK)

The stack uses a **single OpenAI-compatible client**. Configure it with generic variables; legacy names still work.

**Preferred (provider-agnostic):**

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.example.com/v1   # omit for official OpenAI
LLM_MODEL=gpt-4o-mini
LLM_PROVIDER=openai-compatible
```

**Precedence:** `LLM_API_KEY` → `DEEPSEEK_API_KEY` → `OPENAI_API_KEY`.

**Base URL:** `LLM_BASE_URL` if set; otherwise, if only `DEEPSEEK_API_KEY` is set (and not `LLM_API_KEY`), `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com`.

**Model default:** if only legacy DeepSeek key is used, default model is `deepseek-chat`; otherwise `gpt-4o-mini`.

#### DeepSeek (legacy env names)

```env
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

#### OpenAI

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
# Leave LLM_BASE_URL unset for api.openai.com
```

### Configuration reference

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | Primary API key (recommended) |
| `LLM_BASE_URL` | OpenAI-compatible base URL (optional) |
| `LLM_MODEL` | Model id for the provider |
| `LLM_PROVIDER` | Label for logs/settings (e.g. `openai-compatible`) |
| `DEEPSEEK_API_KEY` | Legacy; used if `LLM_API_KEY` unset |
| `DEEPSEEK_BASE_URL` | Legacy DeepSeek endpoint |
| `OPENAI_API_KEY` | Legacy; used if neither of the above keys is set |

## Programmatic configuration

```python
from forecaster.agents.planner import ForecastingPlanner

planner = ForecastingPlanner(
    api_key="your-key",
    model="gpt-4o-mini",
    provider="openai",
    base_url=None,
)
```

## Verification

```python
from forecaster.agents.planner import ForecastingPlanner

planner = ForecastingPlanner()
result = planner.process({
    "user_request": "Forecast next 7 days",
    "data_summary": {"valid": True, "n_points": 30},
    "n_points": 30,
})
print(result.success)
```

## Troubleshooting

### "API key not found"

- Ensure `.env` exists in the **project root** (or export vars in the shell).
- Set `LLM_API_KEY` or a legacy key (`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`).

### "Connection error" or "Invalid API key"

- Confirm the key and that `LLM_BASE_URL` matches the provider.

### "Model not found"

- Use a model id valid for that provider and endpoint.

## Frontend login (Next.js)

### Local admin (after seed)

- **Email:** `admin@local.dev`
- **Password:** `admin`

```bash
cd backend && uv run python scripts/seed_admin.py
```

(Docker: `docker compose run --rm backend python scripts/seed_admin.py`)

### Self-registration

1. Open http://localhost:3000/login and use **Sign up**.
2. Later, use **Sign in** with the same email and password.

## Security Notes

- Do not commit `.env`.
- Rotate keys if exposed.
- Use different keys for dev and production.

## Langfuse Observability

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

**Backend on host, Langfuse in Docker (host port 3001):** `LANGFUSE_BASE_URL=http://localhost:3001`

**Backend inside Compose:** use the internal URL from `docker-compose.yml` (`http://langfuse-web:3000`), not `localhost:3001` from inside the backend container.

If these variables are missing, tracing falls back to local-only behavior.
