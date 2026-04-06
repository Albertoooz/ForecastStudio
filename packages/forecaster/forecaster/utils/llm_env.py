"""Resolve OpenAI-compatible LLM settings from environment (provider-agnostic)."""

from __future__ import annotations

import os


def get_openai_compatible_settings() -> tuple[str | None, str | None, str]:
    """
    Returns (api_key, base_url, model) for OpenAI SDK.

    Precedence for API key: LLM_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY.
    Base URL: LLM_BASE_URL, else DEEPSEEK_BASE_URL when using legacy DeepSeek key only.
    Model: LLM_MODEL, with sensible defaults.
    """
    api_key = (
        os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    )

    base_url = os.getenv("LLM_BASE_URL")
    if not base_url and os.getenv("DEEPSEEK_API_KEY") and not os.getenv("LLM_API_KEY"):
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    model = os.getenv("LLM_MODEL")
    if not model:
        if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("LLM_API_KEY"):
            model = "deepseek-chat"
        else:
            model = "gpt-4o-mini"

    if base_url == "":
        base_url = None

    return api_key, base_url, model
