from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text

from app.api.deps import enforce_rate_limit, require_api_key
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReindexResponse,
    TokenUsageModel,
)
from app.config import get_settings
from app.core.logging import get_logger, new_request_id, set_session_id
from app.db.engine import get_engine
from app.pipeline.orchestrator import get_pipeline
from app.retrieval.indexer import index_schema
from app.retrieval.retriever import get_retriever
from app.retrieval.store import get_vector_store
from app.session.factory import get_session_store

log = get_logger(__name__)

router = APIRouter()


@router.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    dependencies=[Depends(enforce_rate_limit)],
    summary="Ask a question about the ERP in plain English",
)
async def chat(
    payload: ChatRequest,
    api_key: str = Depends(require_api_key),  # noqa: ARG001 - auth side effect
) -> ChatResponse:
    new_request_id()
    session_id = payload.session_id or f"s_{uuid.uuid4().hex[:24]}"
    set_session_id(session_id)

    outcome = get_pipeline().handle(
        session_id=session_id,
        message=payload.message,
        user_id=payload.user_id,
        company_id=payload.company_id,
    )

    return ChatResponse(
        session_id=outcome.session_id,
        type=outcome.type,
        message=outcome.message,
        sql_generated=outcome.sql_generated,
        tables_used=outcome.tables_used,
        tokens_used=TokenUsageModel(
            input=outcome.tokens.input, output=outcome.tokens.output
        ),
        cost_estimate_inr=outcome.tokens.cost_inr,
    )


@router.get("/healthz", response_model=HealthResponse, summary="Liveness and wiring check")
async def healthz() -> HealthResponse:
    settings = get_settings()
    engine = get_engine()

    database_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        database_ok = False
        log.error("health check: database unreachable", extra={"error": str(exc)})

    store = get_vector_store(engine)
    indexed = store.count() if database_ok else 0

    return HealthResponse(
        status="ok" if database_ok and indexed > 0 else "degraded",
        env=settings.env,
        database=database_ok,
        vector_index=indexed,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        vector_backend=store.backend,
        session_backend=get_session_store().backend,
    )


@router.post(
    "/api/v1/admin/reindex",
    response_model=ReindexResponse,
    summary="Rebuild the table-description index after a schema change",
)
async def reindex(
    force: bool = False,
    api_key: str = Depends(require_api_key),  # noqa: ARG001 - auth side effect
) -> ReindexResponse:
    if not get_settings().auth_enabled:
        # Reindexing walks the whole schema and calls the embedding provider;
        # leaving it open on an unauthenticated deployment is not acceptable.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="reindex requires API_KEYS to be configured",
        )

    result = index_schema(force=force)
    get_retriever().refresh()
    return ReindexResponse(
        total_tables=result.total_tables,
        embedded=result.embedded,
        skipped=result.skipped,
        pruned=result.pruned,
        backend=result.backend,
        model=result.model,
        duration_ms=result.duration_ms,
    )
