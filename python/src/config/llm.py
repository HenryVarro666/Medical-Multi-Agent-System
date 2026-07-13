"""
Shared LLM factory.

All LLM-backed agents obtain their ChatOpenAI client here, so the entire
pipeline can be pointed at either the real OpenAI API or any OpenAI-compatible
server (e.g. a local vLLM instance) purely via environment variables:

    OPENAI_BASE_URL=http://localhost:8000/v1
    OPENAI_API_KEY=local
    OPENAI_MODEL=Qwen/Qwen3-14B
"""

from __future__ import annotations
from langchain_openai import ChatOpenAI

from .settings import get_settings


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Build a ChatOpenAI client honoring base-URL / local-model settings."""
    settings = get_settings()
    kwargs: dict = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key or "local",
        "temperature": temperature,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 1,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
        if settings.llm_disable_thinking:
            # Qwen3-style hybrid models: skip the <think> phase for fast,
            # deterministic structured output.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
    return ChatOpenAI(**kwargs)
