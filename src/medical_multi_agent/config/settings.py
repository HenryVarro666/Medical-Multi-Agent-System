"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Optional OpenAI-compatible endpoint (e.g. local vLLM: http://localhost:8001/v1).
    # Empty string = use the real OpenAI API.
    openai_base_url: str = ""
    # For Qwen3-style hybrid reasoning models served locally: skip the <think>
    # phase so agents get fast, clean structured output. Only applied when
    # openai_base_url is set (the real OpenAI API rejects this parameter).
    llm_disable_thinking: bool = False
    llm_timeout_seconds: float = Field(default=180.0, gt=0, allow_inf_nan=False)

    # Pipeline
    # Max number of diagnosis->intake loop-backs before proceeding to
    # treatment on a low-confidence path (guarantees loop termination).
    max_diagnosis_retries: int = Field(default=2, ge=0)

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # FHIR
    fhir_server_url: str = "http://localhost:8080/fhir"


@lru_cache
def get_settings() -> Settings:
    return Settings()
