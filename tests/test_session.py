from __future__ import annotations

import uuid

import pytest

from app.session.base import Message, SessionState, build_topic_digest
from tests.conftest import requires_db


@pytest.fixture
def store(engine):
    from app.session.postgres_store import PostgresSessionStore

    store = PostgresSessionStore(engine)
    store.ensure_schema()
    return store


@pytest.fixture
def session_id() -> str:
    return f"test_{uuid.uuid4().hex[:12]}"


@requires_db
def test_creates_session_on_first_get(store, session_id) -> None:
    state = store.get(session_id, user_id="u1")
    assert state.session_id == session_id
    assert state.messages == []
    assert not state.awaiting_clarification


@requires_db
def test_round_trips_messages_in_order(store, session_id) -> None:
    store.get(session_id)
    store.append_message(session_id, "user", "how many employees?")
    store.append_message(session_id, "assistant", "You have 89.")

    state = store.get(session_id)
    assert [m.role for m in state.messages] == ["user", "assistant"]
    assert state.messages[0].content == "how many employees?"


@requires_db
def test_pending_clarification_round_trip(store, session_id) -> None:
    store.get(session_id)
    store.set_pending_clarification(
        session_id, "Which warehouse did you mean?", "how much stock?"
    )

    state = store.get(session_id)
    assert state.awaiting_clarification
    assert state.pending_clarification == "Which warehouse did you mean?"
    assert state.pending_intent == "how much stock?"

    store.clear_pending_clarification(session_id)
    assert not store.get(session_id).awaiting_clarification


@requires_db
def test_history_is_trimmed_and_older_turns_become_a_digest(store, session_id) -> None:
    from app.config import get_settings

    max_turns = get_settings().history_max_turns
    store.get(session_id)
    for i in range(max_turns * 2 + 4):
        store.append_message(session_id, "user", f"question number {i}")
        store.append_message(session_id, "assistant", f"answer number {i}")

    state = store.get(session_id)
    assert len(state.messages) <= max_turns * 2
    assert state.topic_digest, "evicted turns should survive as a topic digest"
    # The most recent turn must still be in the live window.
    assert any("question number" in m.content for m in state.messages)


@requires_db
def test_sessions_are_isolated(store) -> None:
    a, b = f"a_{uuid.uuid4().hex[:8]}", f"b_{uuid.uuid4().hex[:8]}"
    store.get(a)
    store.get(b)
    store.append_message(a, "user", "only in a")

    assert store.get(b).messages == []
    assert len(store.get(a).messages) == 1


# --- pure logic, no database ---------------------------------------------


def test_history_block_renders_oldest_first() -> None:
    state = SessionState(
        session_id="s",
        messages=[
            Message(role="user", content="first"),
            Message(role="assistant", content="second"),
        ],
    )
    block = state.history_block()
    assert block.index("User: first") < block.index("Assistant: second")


def test_history_block_includes_digest() -> None:
    state = SessionState(session_id="s", topic_digest="stock levels; payroll")
    assert "Earlier in this conversation" in state.history_block()


def test_topic_digest_keeps_recent_topics_within_limit() -> None:
    evicted = [Message(role="user", content=f"topic {i} " + "x" * 40) for i in range(20)]
    digest = build_topic_digest("", evicted, limit=200)
    assert len(digest) <= 200
    assert "topic 19" in digest, "most recent topic must survive truncation"


def test_topic_digest_ignores_assistant_turns() -> None:
    evicted = [
        Message(role="assistant", content="a long assistant answer"),
        Message(role="user", content="what about stock"),
    ]
    digest = build_topic_digest("", evicted)
    assert "stock" in digest
    assert "assistant answer" not in digest
