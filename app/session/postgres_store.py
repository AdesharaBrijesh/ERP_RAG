"""Postgres-backed session store.

Default backend: the ERP already runs Postgres, so this adds no infrastructure
to operate, back up or secure. Lives in the `rag` schema alongside the
embeddings, written through the admin engine (never the read-only one).
"""

from __future__ import annotations

from sqlalchemy import Engine, text

from app.config import get_settings
from app.core.logging import get_logger
from app.session.base import Message, SessionState, SessionStore, build_topic_digest

log = get_logger(__name__)


class PostgresSessionStore(SessionStore):
    backend = "postgres"

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.schema = get_settings().rag_schema

    def ensure_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.chat_sessions (
                        session_id            text PRIMARY KEY,
                        user_id               text,
                        topic_digest          text NOT NULL DEFAULT '',
                        pending_clarification text,
                        pending_intent        text,
                        turn_count            integer NOT NULL DEFAULT 0,
                        created_at            timestamptz NOT NULL DEFAULT now(),
                        last_active_at        timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.chat_messages (
                        id         bigserial PRIMARY KEY,
                        session_id text NOT NULL
                                   REFERENCES {self.schema}.chat_sessions(session_id)
                                   ON DELETE CASCADE,
                        role       text NOT NULL CHECK (role IN ('user', 'assistant')),
                        content    text NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS chat_messages_session_idx "
                    f"ON {self.schema}.chat_messages (session_id, id)"
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS chat_sessions_last_active_idx "
                    f"ON {self.schema}.chat_sessions (last_active_at)"
                )
            )

    def get(self, session_id: str, user_id: str | None = None) -> SessionState:
        settings = get_settings()
        window = settings.history_max_turns * 2

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.schema}.chat_sessions (session_id, user_id)
                    VALUES (:sid, :uid)
                    ON CONFLICT (session_id) DO UPDATE
                        SET last_active_at = now(),
                            user_id = COALESCE({self.schema}.chat_sessions.user_id,
                                               EXCLUDED.user_id)
                    """
                ),
                {"sid": session_id, "uid": user_id},
            )
            row = conn.execute(
                text(
                    f"""
                    SELECT user_id, topic_digest, pending_clarification,
                           pending_intent, turn_count
                    FROM {self.schema}.chat_sessions WHERE session_id = :sid
                    """
                ),
                {"sid": session_id},
            ).mappings().one()

            messages = conn.execute(
                text(
                    f"""
                    SELECT role, content, created_at FROM (
                        SELECT role, content, created_at, id
                        FROM {self.schema}.chat_messages
                        WHERE session_id = :sid
                        ORDER BY id DESC
                        LIMIT :window
                    ) recent
                    ORDER BY id ASC
                    """
                ),
                {"sid": session_id, "window": window},
            ).mappings().all()

        return SessionState(
            session_id=session_id,
            user_id=row["user_id"],
            messages=[
                Message(role=m["role"], content=m["content"], created_at=m["created_at"])
                for m in messages
            ],
            topic_digest=row["topic_digest"] or "",
            pending_clarification=row["pending_clarification"],
            pending_intent=row["pending_intent"],
            turn_count=row["turn_count"] or 0,
        )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        settings = get_settings()
        window = settings.history_max_turns * 2

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.schema}.chat_messages (session_id, role, content)
                    VALUES (:sid, :role, :content)
                    """
                ),
                {"sid": session_id, "role": role, "content": content},
            )
            conn.execute(
                text(
                    f"""
                    UPDATE {self.schema}.chat_sessions
                    SET last_active_at = now(),
                        turn_count = turn_count + CASE WHEN :role = 'user' THEN 1 ELSE 0 END
                    WHERE session_id = :sid
                    """
                ),
                {"sid": session_id, "role": role},
            )

            # Roll anything outside the window into the topic digest, then drop it.
            evicted = conn.execute(
                text(
                    f"""
                    SELECT role, content FROM {self.schema}.chat_messages
                    WHERE session_id = :sid
                      AND id <= (
                        SELECT id FROM {self.schema}.chat_messages
                        WHERE session_id = :sid ORDER BY id DESC
                        OFFSET :window LIMIT 1
                      )
                    ORDER BY id ASC
                    """
                ),
                {"sid": session_id, "window": window},
            ).mappings().all()

            if evicted:
                current = conn.execute(
                    text(
                        f"SELECT topic_digest FROM {self.schema}.chat_sessions "
                        f"WHERE session_id = :sid"
                    ),
                    {"sid": session_id},
                ).scalar() or ""
                digest = build_topic_digest(
                    current, [Message(role=e["role"], content=e["content"]) for e in evicted]
                )
                conn.execute(
                    text(
                        f"UPDATE {self.schema}.chat_sessions SET topic_digest = :digest "
                        f"WHERE session_id = :sid"
                    ),
                    {"sid": session_id, "digest": digest},
                )
                conn.execute(
                    text(
                        f"""
                        DELETE FROM {self.schema}.chat_messages
                        WHERE session_id = :sid AND id <= (
                            SELECT id FROM {self.schema}.chat_messages
                            WHERE session_id = :sid ORDER BY id DESC
                            OFFSET :window LIMIT 1
                        )
                        """
                    ),
                    {"sid": session_id, "window": window},
                )

    def set_pending_clarification(
        self, session_id: str, question: str, original_message: str
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    UPDATE {self.schema}.chat_sessions
                    SET pending_clarification = :q, pending_intent = :intent
                    WHERE session_id = :sid
                    """
                ),
                {"sid": session_id, "q": question, "intent": original_message},
            )

    def clear_pending_clarification(self, session_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    UPDATE {self.schema}.chat_sessions
                    SET pending_clarification = NULL, pending_intent = NULL
                    WHERE session_id = :sid
                    """
                ),
                {"sid": session_id},
            )

    def purge_expired(self) -> int:
        ttl = get_settings().session_ttl_seconds
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    DELETE FROM {self.schema}.chat_sessions
                    WHERE last_active_at < now() - make_interval(secs => :ttl)
                    """
                ),
                {"ttl": ttl},
            )
        removed = result.rowcount or 0
        if removed:
            log.info("purged expired sessions", extra={"purged": removed})
        return removed
