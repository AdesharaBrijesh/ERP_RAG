"""Request/response contract for the backend team.

Single-response JSON, not streaming. The answer is a synthesised summary of a
result set that does not exist until the query returns, so token-by-token
streaming would show a blank screen for most of the latency and then dump the
whole answer anyway. If a typing effect is wanted in the UI, animate it client
side. Say the word and an SSE variant is straightforward to add.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Client-generated, or omit and the server issues one on first call.",
    )
    message: str = Field(min_length=1, description="The user's question in plain English.")
    user_id: str | None = Field(
        default=None, max_length=128, description="Optional, for audit logging."
    )
    company_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Scope every answer to one company. Enforced by row-level security "
            "on the database, not by the model. Omit for group-wide totals. "
            "Send the caller's own company id - never let an end user choose it."
        ),
    )

    @field_validator("company_id")
    @classmethod
    def _check_company_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if not stripped.isdigit():
            # It is interpolated into a Postgres setting and cast to bigint by
            # the RLS predicate; anything non-numeric is a caller bug.
            raise ValueError("company_id must be numeric")
        return stripped

    @field_validator("message")
    @classmethod
    def _check_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        limit = get_settings().max_message_chars
        if len(stripped) > limit:
            raise ValueError(f"message must be at most {limit} characters")
        return stripped


class TokenUsageModel(BaseModel):
    input: int = 0
    output: int = 0


class ChatResponse(BaseModel):
    session_id: str
    type: Literal["answer", "clarification_needed", "error"]
    message: str = Field(description="Conversational Markdown to show the user.")
    sql_generated: str | None = Field(
        default=None, description="Log-only. Not required to display."
    )
    tables_used: list[str] = Field(default_factory=list)
    tokens_used: TokenUsageModel = Field(default_factory=TokenUsageModel)
    cost_estimate_inr: float = 0.0


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    env: str
    database: bool
    vector_index: int
    llm_provider: str
    embedding_provider: str
    vector_backend: str
    session_backend: str


class ReindexResponse(BaseModel):
    total_tables: int
    embedded: int
    skipped: int
    pruned: int
    backend: str
    model: str
    duration_ms: int


class ErrorResponse(BaseModel):
    detail: str
