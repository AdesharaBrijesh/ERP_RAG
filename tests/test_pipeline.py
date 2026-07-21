"""Orchestrator path tests, driven by scripted fake-LLM responses."""

from __future__ import annotations

import json
import uuid

import pytest

from app.llm.fake_provider import FakeProvider
from app.pipeline.orchestrator import ChatPipeline
from tests.conftest import requires_db


def _sql(query: str) -> str:
    return json.dumps({"decision": "sql", "sql": query, "tables_used": ["items"]})


def _clarify(question: str) -> str:
    return json.dumps(
        {"decision": "clarify", "clarifying_question": question, "tables_used": []}
    )


@pytest.fixture
def session_id() -> str:
    return f"pipe_{uuid.uuid4().hex[:10]}"


@requires_db
def test_direct_answer_path(retriever, session_id) -> None:
    llm = FakeProvider(
        scripted=[
            _sql("SELECT count(*) AS total FROM items WHERE is_deleted = false"),
            "You have 89 items.",
        ]
    )
    outcome = ChatPipeline(llm=llm, retriever=retriever).handle(
        session_id, "how many items do we have?"
    )

    assert outcome.type == "answer"
    assert outcome.path == "direct_answer"
    assert outcome.row_count == 1
    assert outcome.tokens.input > 0


@requires_db
def test_failed_sql_is_repaired_and_answered(retriever, session_id) -> None:
    """A query that fails at the database gets one corrective pass.

    This is the path that turned 'ORDER BY an ungrouped column' from a dead
    end into a correct answer during live testing.
    """
    llm = FakeProvider(
        scripted=[
            _sql("SELECT no_such_column FROM items"),  # fails at the database
            _sql("SELECT count(*) AS total FROM items WHERE is_deleted = false"),
            "You have 89 items.",
        ]
    )
    outcome = ChatPipeline(llm=llm, retriever=retriever).handle(
        session_id, "how many items do we have?"
    )

    assert outcome.type == "answer"
    assert outcome.path == "repaired_answer"
    assert "no_such_column" not in (outcome.sql_generated or "")


@requires_db
def test_repair_is_attempted_only_once(retriever, session_id) -> None:
    """Two consecutive failures end the request rather than looping."""
    llm = FakeProvider(
        scripted=[
            _sql("SELECT no_such_column FROM items"),
            _sql("SELECT still_no_such_column FROM items"),
        ]
    )
    outcome = ChatPipeline(llm=llm, retriever=retriever).handle(
        session_id, "how many items do we have?"
    )

    assert outcome.type == "error"
    assert outcome.path == "sql_error"
    assert len(llm.calls) == 2, "should route once and repair once, then stop"


@requires_db
def test_write_attempt_is_rejected_before_execution(retriever, session_id) -> None:
    """SECURITY: a hallucinated DELETE never reaches the database, and the
    user gets a friendly refusal rather than a stack trace."""
    llm = FakeProvider(scripted=[_sql("DELETE FROM items")])
    outcome = ChatPipeline(llm=llm, retriever=retriever).handle(
        session_id, "delete all the items"
    )

    assert outcome.type == "error"
    assert outcome.path == "guard_rejected"
    assert "only look information up" in outcome.message


@requires_db
def test_clarification_sets_pending_state(retriever, session_id) -> None:
    llm = FakeProvider(scripted=[_clarify("Which warehouse did you mean?")])
    pipeline = ChatPipeline(llm=llm, retriever=retriever)
    outcome = pipeline.handle(session_id, "how much stock?")

    assert outcome.type == "clarification_needed"
    assert outcome.path == "clarification"

    state = pipeline.sessions.get(session_id)
    assert state.awaiting_clarification
    assert state.pending_intent == "how much stock?"


@requires_db
def test_answering_a_clarification_resumes_the_original_intent(
    retriever, session_id
) -> None:
    llm = FakeProvider(
        scripted=[
            _clarify("Which warehouse did you mean?"),
            _sql("SELECT count(*) AS total FROM items WHERE is_deleted = false"),
            "Here is the stock for that warehouse.",
        ]
    )
    pipeline = ChatPipeline(llm=llm, retriever=retriever)
    pipeline.handle(session_id, "how much stock?")
    outcome = pipeline.handle(session_id, "the Sanand one")

    assert outcome.type == "answer"
    # The follow-up prompt must carry the original question, not just "the
    # Sanand one", which is meaningless on its own.
    _, repeat_prompt = llm.calls[1]
    assert "how much stock?" in repeat_prompt

    assert not pipeline.sessions.get(session_id).awaiting_clarification


@requires_db
def test_malformed_model_output_degrades_to_clarification(retriever, session_id) -> None:
    """An unparseable response must never become an unguarded query."""
    llm = FakeProvider(scripted=["I'm not sure what you mean by that, sorry!"])
    outcome = ChatPipeline(llm=llm, retriever=retriever).handle(session_id, "hello?")

    assert outcome.type == "clarification_needed"
    assert outcome.sql_generated is None
