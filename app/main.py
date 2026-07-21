from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.core.logging import configure_logging, current_request_id, get_logger, new_request_id
from app.db.engine import dispose_engines, get_engine
from app.retrieval.retriever import get_retriever
from app.retrieval.store import get_vector_store
from app.session.factory import get_session_store

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    # Warm the expensive singletons at boot rather than on the first user
    # request: schema snapshot, embedding model, session tables.
    engine = get_engine()
    store = get_vector_store(engine)
    get_session_store().ensure_schema()

    indexed = store.count()
    if indexed == 0:
        log.warning(
            "vector index is empty - run `python -m scripts.index_schema` "
            "or POST /api/v1/admin/reindex before serving traffic"
        )
    else:
        get_retriever()

    log.info(
        "service started",
        extra={
            "env": settings.env,
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
            "vector_backend": store.backend,
            "indexed_tables": indexed,
            "auth_enabled": settings.auth_enabled,
        },
    )
    yield
    dispose_engines()
    log.info("service stopped")


app = FastAPI(
    title="ERP RAG Chatbot",
    version="0.1.0",
    description=(
        "Ask questions about the ERP in plain English. Retrieval prunes the "
        "87-table schema to the few tables that matter, an LLM writes a "
        "read-only SELECT, and a second call turns the rows into a "
        "conversational answer."
    ),
    lifespan=lifespan,
)
app.include_router(router)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    started = time.perf_counter()
    request_id = new_request_id()
    try:
        response = await call_next(request)
    except Exception:
        log.exception(
            "unhandled error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error"},
            headers={"X-Request-ID": request_id},
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "http request",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    response.headers["X-Request-ID"] = current_request_id() or request_id
    return response
