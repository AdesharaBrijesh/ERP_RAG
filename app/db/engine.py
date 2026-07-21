"""SQLAlchemy engines.

Two engines, deliberately separate:

* ``get_engine()``    - app/admin connection. Schema introspection, the rag.*
                        schema (embeddings, sessions). Never runs generated SQL.
* ``get_ro_engine()`` - read-only connection used *exclusively* to execute
                        LLM-generated SQL. Pinned to a read-only transaction
                        with a statement timeout at connection level, so even a
                        bug upstream in the guard cannot write.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine, event

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )


@lru_cache
def get_ro_engine() -> Engine:
    settings = get_settings()
    engine = create_engine(
        settings.effective_readonly_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
        connect_args={
            "options": (
                f"-c statement_timeout={settings.sql_statement_timeout_ms} "
                f"-c idle_in_transaction_session_timeout=30000 "
                f"-c default_transaction_read_only=on"
            )
        },
    )

    @event.listens_for(engine, "connect")
    def _enforce_readonly(dbapi_conn, _record):  # pragma: no cover - driver hook
        # Belt and braces: also set it per session in case connect_args options
        # are stripped by a pooler (pgbouncer) in front of Postgres.
        with dbapi_conn.cursor() as cur:
            cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            cur.execute(
                f"SET statement_timeout = {get_settings().sql_statement_timeout_ms}"
            )
        dbapi_conn.commit()

    return engine


def dispose_engines() -> None:
    for factory in (get_engine, get_ro_engine):
        if factory.cache_info().currsize:
            factory().dispose()
        factory.cache_clear()
