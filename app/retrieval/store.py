"""Vector store for table-description embeddings.

Two interchangeable backends behind one interface:

* ``PgVectorStore``  - uses the pgvector extension and an HNSW index. This is
  what production (RDS / Aurora) runs.
* ``ArrayVectorStore`` - stores the vector as ``double precision[]`` and does
  the cosine scan in-process with numpy. Local Postgres 18.4 on Windows has no
  pgvector available, and at 87 rows a brute-force scan is sub-millisecond, so
  nothing is lost in development.

``get_vector_store()`` picks the backend by probing for the extension, so the
same code path runs in both places with no config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sqlalchemy import Engine, text

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class TableEmbedding:
    table_name: str
    description: str
    keywords: str
    checksum: str
    embedding: list[float]


@dataclass(frozen=True)
class ScoredTable:
    table_name: str
    description: str
    keywords: str
    score: float


class VectorStore(ABC):
    backend: str = "base"

    @abstractmethod
    def ensure_schema(self, dim: int) -> None: ...

    @abstractmethod
    def upsert(self, records: list[TableEmbedding], index_signature: str) -> None: ...

    @abstractmethod
    def search(self, query_vec: list[float], k: int) -> list[ScoredTable]: ...

    @abstractmethod
    def existing_checksums(self) -> dict[str, str]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def prune(self, keep_tables: set[str]) -> int: ...


def _rag_schema() -> str:
    return get_settings().rag_schema


class PgVectorStore(VectorStore):
    backend = "pgvector"

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.schema = _rag_schema()

    def ensure_schema(self, dim: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.table_embeddings (
                        table_name      text PRIMARY KEY,
                        description     text NOT NULL,
                        keywords        text NOT NULL DEFAULT '',
                        checksum        text NOT NULL,
                        index_signature text NOT NULL,
                        embedding       vector({dim}) NOT NULL,
                        updated_at      timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS table_embeddings_hnsw
                    ON {self.schema}.table_embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )

    def upsert(self, records: list[TableEmbedding], index_signature: str) -> None:
        rows = [
            {
                "table_name": r.table_name,
                "description": r.description,
                "keywords": r.keywords,
                "checksum": r.checksum,
                "index_signature": index_signature,
                "embedding": "[" + ",".join(f"{v:.6f}" for v in r.embedding) + "]",
            }
            for r in records
        ]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.schema}.table_embeddings
                        (table_name, description, keywords, checksum,
                         index_signature, embedding, updated_at)
                    VALUES (:table_name, :description, :keywords, :checksum,
                            :index_signature, CAST(:embedding AS vector), now())
                    ON CONFLICT (table_name) DO UPDATE SET
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        checksum = EXCLUDED.checksum,
                        index_signature = EXCLUDED.index_signature,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """
                ),
                rows,
            )

    def search(self, query_vec: list[float], k: int) -> list[ScoredTable]:
        literal = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT table_name, description, keywords,
                               1 - (embedding <=> CAST(:vec AS vector)) AS score
                        FROM {self.schema}.table_embeddings
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT :k
                        """
                    ),
                    {"vec": literal, "k": k},
                )
                .mappings()
                .all()
            )
        return [
            ScoredTable(r["table_name"], r["description"], r["keywords"], float(r["score"]))
            for r in rows
        ]

    def existing_checksums(self) -> dict[str, str]:
        return _existing_checksums(self.engine, self.schema)

    def count(self) -> int:
        return _count(self.engine, self.schema)

    def prune(self, keep_tables: set[str]) -> int:
        return _prune(self.engine, self.schema, keep_tables)


class ArrayVectorStore(VectorStore):
    """pgvector-free fallback: float8[] column + in-process numpy cosine scan."""

    backend = "array"

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.schema = _rag_schema()
        self._cache: tuple[list[ScoredTable], np.ndarray] | None = None

    def ensure_schema(self, dim: int) -> None:  # noqa: ARG002 - dim unused here
        with self.engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.table_embeddings (
                        table_name      text PRIMARY KEY,
                        description     text NOT NULL,
                        keywords        text NOT NULL DEFAULT '',
                        checksum        text NOT NULL,
                        index_signature text NOT NULL,
                        embedding       double precision[] NOT NULL,
                        updated_at      timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )

    def upsert(self, records: list[TableEmbedding], index_signature: str) -> None:
        rows = [
            {
                "table_name": r.table_name,
                "description": r.description,
                "keywords": r.keywords,
                "checksum": r.checksum,
                "index_signature": index_signature,
                "embedding": r.embedding,
            }
            for r in records
        ]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.schema}.table_embeddings
                        (table_name, description, keywords, checksum,
                         index_signature, embedding, updated_at)
                    VALUES (:table_name, :description, :keywords, :checksum,
                            :index_signature, :embedding, now())
                    ON CONFLICT (table_name) DO UPDATE SET
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        checksum = EXCLUDED.checksum,
                        index_signature = EXCLUDED.index_signature,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """
                ),
                rows,
            )
        self._cache = None

    def _load(self) -> tuple[list[ScoredTable], np.ndarray]:
        if self._cache is not None:
            return self._cache
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        f"SELECT table_name, description, keywords, embedding "
                        f"FROM {self.schema}.table_embeddings ORDER BY table_name"
                    )
                )
                .mappings()
                .all()
            )
        meta = [ScoredTable(r["table_name"], r["description"], r["keywords"], 0.0) for r in rows]
        if rows:
            matrix = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-9)
        else:
            matrix = np.zeros((0, 1), dtype=np.float32)
        self._cache = (meta, matrix)
        return self._cache

    def search(self, query_vec: list[float], k: int) -> list[ScoredTable]:
        meta, matrix = self._load()
        if not meta:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = matrix @ q
        top = np.argsort(-scores)[:k]
        return [
            ScoredTable(meta[i].table_name, meta[i].description, meta[i].keywords, float(scores[i]))
            for i in top
        ]

    def invalidate_cache(self) -> None:
        self._cache = None

    def existing_checksums(self) -> dict[str, str]:
        return _existing_checksums(self.engine, self.schema)

    def count(self) -> int:
        return _count(self.engine, self.schema)

    def prune(self, keep_tables: set[str]) -> int:
        removed = _prune(self.engine, self.schema, keep_tables)
        self._cache = None
        return removed


# --- shared helpers -------------------------------------------------------


def _table_exists(engine: Engine, schema: str) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT to_regclass(:name)"
                ),
                {"name": f"{schema}.table_embeddings"},
            ).scalar()
        )


def _existing_checksums(engine: Engine, schema: str) -> dict[str, str]:
    if not _table_exists(engine, schema):
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT table_name, checksum, index_signature "
                f"FROM {schema}.table_embeddings"
            )
        ).mappings().all()
    return {r["table_name"]: f"{r['checksum']}::{r['index_signature']}" for r in rows}


def _count(engine: Engine, schema: str) -> int:
    if not _table_exists(engine, schema):
        return 0
    with engine.connect() as conn:
        return int(
            conn.execute(text(f"SELECT count(*) FROM {schema}.table_embeddings")).scalar() or 0
        )


def _prune(engine: Engine, schema: str, keep_tables: set[str]) -> int:
    if not _table_exists(engine, schema) or not keep_tables:
        return 0
    with engine.begin() as conn:
        result = conn.execute(
            text(
                f"DELETE FROM {schema}.table_embeddings "
                f"WHERE table_name <> ALL(:keep)"
            ),
            {"keep": list(keep_tables)},
        )
    return result.rowcount or 0


def pgvector_available(engine: Engine) -> bool:
    with engine.connect() as conn:
        installed = conn.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        if installed:
            return True
        return bool(
            conn.execute(
                text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            ).scalar()
        )


_store: VectorStore | None = None


def get_vector_store(engine: Engine, force_backend: str | None = None) -> VectorStore:
    """Pick the backend by probing for pgvector. Cached per process."""
    global _store
    if _store is not None and force_backend is None:
        return _store

    backend = force_backend
    if backend is None:
        backend = "pgvector" if pgvector_available(engine) else "array"
        if backend == "array":
            log.info(
                "pgvector extension not available; using float8[] store with "
                "in-process cosine scan",
                extra={"vector_backend": backend},
            )

    store: VectorStore = (
        PgVectorStore(engine) if backend == "pgvector" else ArrayVectorStore(engine)
    )
    if force_backend is None:
        _store = store
    return store


def reset_vector_store() -> None:
    global _store
    _store = None
