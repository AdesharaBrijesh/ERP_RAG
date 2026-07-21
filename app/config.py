"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg://postgres:Brijesh%40292@localhost:5432/erp_db"
    )
    # Every LLM-generated statement runs through this connection only.
    readonly_database_url: str = Field(default="")
    db_schema: str = Field(default="public")
    rag_schema: str = Field(default="rag")

    # --- LLM ---
    llm_provider: Literal["groq", "bedrock", "fake"] = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    bedrock_region: str = "ap-south-1"
    bedrock_model_id: str = "us.meta.llama3-3-70b-instruct-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    llm_timeout_seconds: float = 45.0

    # --- Embeddings ---
    embedding_provider: Literal["bedrock", "fastembed", "hashing"] = "hashing"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    hashing_dim: int = 512

    # --- Retrieval ---
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.15
    retrieval_fk_expansion: bool = True
    retrieval_max_tables: int = 8

    # --- SQL guardrails ---
    sql_statement_timeout_ms: int = 10_000
    sql_max_rows: int = 200
    sql_max_rows_to_llm: int = 50

    # --- Sessions ---
    session_backend: Literal["postgres", "redis"] = "postgres"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 86_400
    history_max_turns: int = 8
    max_message_chars: int = 1_000

    # --- API security ---
    # Comma-separated in the environment; parsed via the `api_keys` property.
    # Kept as a plain string so pydantic-settings does not try to JSON-decode it.
    api_keys_raw: str = Field(default="", alias="API_KEYS")
    rate_limit_per_minute: int = 30

    # --- Cost ---
    usd_to_inr: float = 88.0

    # --- Misc ---
    log_level: str = "INFO"
    env: str = "local"

    @property
    def api_keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]

    @property
    def effective_readonly_url(self) -> str:
        """Fall back to the main URL so local dev works before bootstrap_db runs.

        The SELECT-only guard still applies either way; the read-only role is
        defence in depth, not the only defence.
        """
        return self.readonly_database_url or self.database_url

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()
