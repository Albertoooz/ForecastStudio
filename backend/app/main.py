"""
Forecaster Platform — FastAPI Application.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import get_settings, sync_langfuse_env_from_settings, sync_llm_env_from_settings
from app.db.session import engine, Base
from app.api import auth, connections, data, models, agents, monitoring
from forecaster.utils.observability import flush_langfuse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown events."""
    # Startup: create tables (dev only — use Alembic in prod)
    settings = get_settings()
    sync_llm_env_from_settings(settings)
    sync_langfuse_env_from_settings(settings)
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        logger.info("Langfuse tracing: enabled")
    else:
        logger.info("Langfuse tracing: disabled (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)")
    if settings.debug:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown: dispose engine
    await engine.dispose()
    flush_langfuse()


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url=f"{settings.api_prefix}/docs",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["auth"])
    application.include_router(data.router, prefix=f"{settings.api_prefix}/data", tags=["data"])
    application.include_router(
        connections.router,
        prefix=f"{settings.api_prefix}/data/connections",
        tags=["data-connections"],
    )
    application.include_router(
        models.router, prefix=f"{settings.api_prefix}/models", tags=["models"]
    )
    application.include_router(
        agents.router, prefix=f"{settings.api_prefix}/agents", tags=["agents"]
    )
    application.include_router(
        monitoring.router, prefix=f"{settings.api_prefix}/monitoring", tags=["monitoring"]
    )

    @application.get("/")
    async def root():
        """Redirect to API docs — API routes live under /api."""
        return RedirectResponse(url=f"{settings.api_prefix}/docs", status_code=302)

    @application.get("/health")
    async def health():
        """
        Liveness probe. Includes Celery worker ping so you can see if training tasks
        will stay stuck in `queued` (no workers consuming the `training` queue).
        """
        celery_workers: list[str] = []
        try:
            from app.tasks import celery_app

            ping = celery_app.control.inspect(timeout=1.0).ping()
            if ping:
                celery_workers = list(ping.keys())
        except Exception:
            pass

        return {
            "status": "ok",
            "version": "0.1.0",
            "celery_workers": celery_workers,
            "celery_training_ready": len(celery_workers) > 0,
        }

    return application


app = create_app()
