"""Redis-backed session store.

Preferred once the service runs on more than one instance: TTL expiry is free,
and there is no write amplification on the ERP's primary database. Selected
with SESSION_BACKEND=redis.
"""

from __future__ import annotations

import json

from app.config import get_settings
from app.session.base import Message, SessionState, SessionStore, build_topic_digest


class RedisSessionStore(SessionStore):
    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self._redis = redis.Redis.from_url(url, decode_responses=True)

    def ensure_schema(self) -> None:
        self._redis.ping()

    def _meta_key(self, session_id: str) -> str:
        return f"erprag:session:{session_id}:meta"

    def _messages_key(self, session_id: str) -> str:
        return f"erprag:session:{session_id}:messages"

    def _touch(self, session_id: str) -> None:
        ttl = get_settings().session_ttl_seconds
        self._redis.expire(self._meta_key(session_id), ttl)
        self._redis.expire(self._messages_key(session_id), ttl)

    def get(self, session_id: str, user_id: str | None = None) -> SessionState:
        settings = get_settings()
        meta_key = self._meta_key(session_id)

        meta = self._redis.hgetall(meta_key)
        if not meta:
            meta = {"user_id": user_id or "", "topic_digest": "", "turn_count": "0"}
            self._redis.hset(meta_key, mapping=meta)
        elif user_id and not meta.get("user_id"):
            self._redis.hset(meta_key, "user_id", user_id)
            meta["user_id"] = user_id

        window = settings.history_max_turns * 2
        raw = self._redis.lrange(self._messages_key(session_id), -window, -1)
        self._touch(session_id)

        messages = []
        for item in raw:
            try:
                payload = json.loads(item)
            except json.JSONDecodeError:
                continue
            messages.append(Message(role=payload["role"], content=payload["content"]))

        return SessionState(
            session_id=session_id,
            user_id=meta.get("user_id") or None,
            messages=messages,
            topic_digest=meta.get("topic_digest", ""),
            pending_clarification=meta.get("pending_clarification") or None,
            pending_intent=meta.get("pending_intent") or None,
            turn_count=int(meta.get("turn_count") or 0),
        )

    def append_message(self, session_id: str, role: str, content: str) -> None:
        settings = get_settings()
        window = settings.history_max_turns * 2
        messages_key = self._messages_key(session_id)
        meta_key = self._meta_key(session_id)

        self._redis.rpush(messages_key, json.dumps({"role": role, "content": content}))
        if role == "user":
            self._redis.hincrby(meta_key, "turn_count", 1)

        overflow = self._redis.llen(messages_key) - window
        if overflow > 0:
            evicted_raw = self._redis.lrange(messages_key, 0, overflow - 1)
            evicted = []
            for item in evicted_raw:
                try:
                    payload = json.loads(item)
                except json.JSONDecodeError:
                    continue
                evicted.append(Message(role=payload["role"], content=payload["content"]))
            digest = build_topic_digest(
                self._redis.hget(meta_key, "topic_digest") or "", evicted
            )
            self._redis.hset(meta_key, "topic_digest", digest)
            self._redis.ltrim(messages_key, overflow, -1)

        self._touch(session_id)

    def set_pending_clarification(
        self, session_id: str, question: str, original_message: str
    ) -> None:
        self._redis.hset(
            self._meta_key(session_id),
            mapping={"pending_clarification": question, "pending_intent": original_message},
        )
        self._touch(session_id)

    def clear_pending_clarification(self, session_id: str) -> None:
        self._redis.hdel(
            self._meta_key(session_id), "pending_clarification", "pending_intent"
        )

    def purge_expired(self) -> int:
        # Redis expires keys itself; nothing to sweep.
        return 0
