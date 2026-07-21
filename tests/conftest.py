from __future__ import annotations

import os

import pytest
from sqlalchemy import text

# Tests must never hit a paid provider.
os.environ.setdefault("LLM_PROVIDER", "fake")


def _database_reachable() -> bool:
    try:
        from app.db.engine import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


requires_db = pytest.mark.skipif(
    not _database_reachable(), reason="erp_db is not reachable"
)


@pytest.fixture(scope="session")
def engine():
    from app.db.engine import get_engine

    return get_engine()


@pytest.fixture(scope="session")
def retriever(engine):
    """Shared retriever - loading the schema snapshot and embedding model is
    slow enough that per-test construction would dominate the run."""
    from app.retrieval.retriever import TableRetriever
    from app.retrieval.store import get_vector_store

    store = get_vector_store(engine)
    if store.count() == 0:
        pytest.skip("vector index is empty - run `python -m scripts.index_schema`")
    return TableRetriever(engine)


@pytest.fixture
def fake_llm():
    from app.llm.fake_provider import FakeProvider

    return FakeProvider()
