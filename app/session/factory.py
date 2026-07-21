from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.core.logging import get_logger
from app.session.base import SessionStore

log = get_logger(__name__)


@lru_cache
def get_session_store() -> SessionStore:
    settings = get_settings()
    if settings.session_backend == "redis":
        try:
            from app.session.redis_store import RedisSessionStore

            store = RedisSessionStore(settings.redis_url)
            store.ensure_schema()
            return store
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "redis session store unavailable, falling back to postgres",
                extra={"error": str(exc)},
            )

    from app.db.engine import get_engine
    from app.session.postgres_store import PostgresSessionStore

    store = PostgresSessionStore(get_engine())
    store.ensure_schema()
    return store


def reset_session_store() -> None:
    get_session_store.cache_clear()
