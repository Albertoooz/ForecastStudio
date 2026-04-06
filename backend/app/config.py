"""
Forecaster Platform — Backend Configuration.

All settings loaded from environment variables (.env / Azure Key Vault).
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore extra env vars (e.g. azure_display_name) so local .env works
    )

    # ── App ──────────────────────────────────────────────
    app_name: str = "Forecaster Platform"
    debug: bool = False
    environment: str = "local"  # local | dev | test | prod
    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/forecaster"
    database_echo: bool = False

    # ── Redis ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Azure Blob Storage ───────────────────────────────
    azure_storage_connection_string: Optional[str] = None
    azure_storage_account_name: Optional[str] = "forecasterblob"
    blob_container_datasets: str = "datasets"
    blob_container_models: str = "models"
    blob_container_forecasts: str = "forecasts"

    # ── Auth (JWT) ───────────────────────────────────────
    secret_key: str = "CHANGE-ME-in-production-use-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours

    # ── LLM (OpenAI-compatible: OpenAI, DeepSeek, OpenRouter, Azure OpenAI, etc.) ──
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai-compatible"
    # Legacy names (still read from env for existing deployments)
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"

    # ── Langfuse (optional tracing; also read via os.environ in forecaster) ──
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None

    # ── Local Storage (dev fallback) ─────────────────────
    local_storage_path: str = ".storage"

    # ── Azure Service Principal ──────────────────────────
    azure_tenant_id: Optional[str] = None
    azure_app_id: Optional[str] = None
    azure_password: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def sync_llm_env_from_settings(settings: Optional[Settings] = None) -> None:
    """Expose LLM config to os.environ for libraries that read env directly (forecaster)."""
    s = settings or get_settings()
    key = s.llm_api_key or s.deepseek_api_key
    if key:
        os.environ.setdefault("LLM_API_KEY", key)
    base = s.llm_base_url
    if not base and s.deepseek_api_key and not s.llm_api_key:
        base = s.deepseek_base_url
    if base:
        os.environ.setdefault("LLM_BASE_URL", base)
    os.environ.setdefault("LLM_MODEL", s.llm_model)
    if s.llm_provider:
        os.environ.setdefault("LLM_PROVIDER", s.llm_provider)


def sync_langfuse_env_from_settings(settings: Optional[Settings] = None) -> None:
    """Ensure Langfuse Python SDK sees keys (Pydantic .env load does not always set os.environ)."""
    s = settings or get_settings()
    if s.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = s.langfuse_public_key
    if s.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = s.langfuse_secret_key
    if s.langfuse_base_url:
        os.environ["LANGFUSE_BASE_URL"] = s.langfuse_base_url
