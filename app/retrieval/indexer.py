"""Offline schema indexing.

Runs on deploy and on schema change - never per request. Introspects the ERP
schema, builds a natural-language description per table, embeds it, and
upserts into the vector store. Checksums mean a re-run only re-embeds tables
whose schema or glossary entry actually changed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import Engine

from app.config import get_settings
from app.core.logging import get_logger
from app.db.introspect import TableInfo, introspect_schema
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.retrieval.descriptions import (
    build_description,
    description_checksum,
    keywords_for,
)
from app.retrieval.store import TableEmbedding, VectorStore, get_vector_store

log = get_logger(__name__)


@dataclass
class IndexResult:
    total_tables: int
    embedded: int
    skipped: int
    pruned: int
    backend: str
    model: str
    duration_ms: int


def index_schema(
    engine: Engine | None = None,
    provider: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
    force: bool = False,
) -> IndexResult:
    started = time.perf_counter()
    settings = get_settings()

    if engine is None:
        from app.db.engine import get_engine

        engine = get_engine()
    provider = provider or get_embedding_provider()
    store = store or get_vector_store(engine)

    tables: list[TableInfo] = introspect_schema(
        engine, schema=settings.db_schema, sample_enum_values=True
    )
    store.ensure_schema(provider.dim)

    existing = {} if force else store.existing_checksums()
    signature = provider.index_signature

    pending: list[tuple[TableInfo, str, str]] = []
    skipped = 0
    for table in tables:
        description = build_description(table)
        checksum = description_checksum(table, description)
        if existing.get(table.name) == f"{checksum}::{signature}":
            skipped += 1
            continue
        pending.append((table, description, checksum))

    if pending:
        vectors = provider.embed_documents([d for _, d, _ in pending])
        records = [
            TableEmbedding(
                table_name=table.name,
                description=description,
                keywords=" ".join(sorted(keywords_for(table))),
                checksum=checksum,
                embedding=vector,
            )
            for (table, description, checksum), vector in zip(pending, vectors, strict=True)
        ]
        store.upsert(records, signature)

    pruned = store.prune({t.name for t in tables})

    result = IndexResult(
        total_tables=len(tables),
        embedded=len(pending),
        skipped=skipped,
        pruned=pruned,
        backend=store.backend,
        model=provider.model_id,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    log.info(
        "schema index complete",
        extra={
            "total_tables": result.total_tables,
            "embedded": result.embedded,
            "skipped": result.skipped,
            "pruned": result.pruned,
            "vector_backend": result.backend,
            "embedding_model": result.model,
            "duration_ms": result.duration_ms,
        },
    )
    return result
