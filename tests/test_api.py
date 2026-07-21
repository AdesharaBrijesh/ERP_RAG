"""Endpoint contract tests.

Run against the fake LLM provider, so they exercise the real retrieval,
guard, execution and session code paths without calling a paid API.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_db

API_KEY = "local-dev-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def client():
    from app.config import get_settings
    from app.main import app

    settings = get_settings()
    if API_KEY not in settings.api_keys:
        pytest.skip("API_KEYS must contain 'local-dev-key' for these tests")

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id() -> str:
    return f"apitest_{uuid.uuid4().hex[:12]}"


@requires_db
def test_health_reports_wiring(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] is True
    assert body["vector_index"] > 0
    assert body["vector_backend"] in ("pgvector", "array")


@requires_db
def test_chat_returns_the_documented_contract(client, session_id) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "how many items do we have?"},
        headers=HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body) == {
        "session_id",
        "type",
        "message",
        "sql_generated",
        "tables_used",
        "tokens_used",
        "cost_estimate_inr",
    }
    assert body["session_id"] == session_id
    assert body["type"] in ("answer", "clarification_needed", "error")
    assert isinstance(body["message"], str) and body["message"]
    assert isinstance(body["tables_used"], list)
    assert set(body["tokens_used"]) == {"input", "output"}
    assert body["tokens_used"]["input"] > 0
    assert isinstance(body["cost_estimate_inr"], float)


@requires_db
def test_server_issues_session_id_when_omitted(client) -> None:
    response = client.post(
        "/api/v1/chat", json={"message": "how many items?"}, headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json()["session_id"].startswith("s_")


@requires_db
def test_conversation_state_persists_across_calls(client, session_id) -> None:
    client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "how many items do we have?"},
        headers=HEADERS,
    )
    client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "message": "and how about warehouses?"},
        headers=HEADERS,
    )

    from app.session.factory import get_session_store

    state = get_session_store().get(session_id)
    assert len(state.messages) == 4  # two user turns, two assistant turns


# --- auth ----------------------------------------------------------------


def test_rejects_missing_api_key(client) -> None:
    response = client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 401


def test_rejects_wrong_api_key(client) -> None:
    response = client.post(
        "/api/v1/chat", json={"message": "hello"}, headers={"X-API-Key": "nope"}
    )
    assert response.status_code == 401


# --- validation ----------------------------------------------------------


def test_rejects_blank_message(client) -> None:
    response = client.post(
        "/api/v1/chat", json={"session_id": "x", "message": "   "}, headers=HEADERS
    )
    assert response.status_code == 422


def test_rejects_oversized_message(client) -> None:
    from app.config import get_settings

    oversized = "a" * (get_settings().max_message_chars + 50)
    response = client.post(
        "/api/v1/chat", json={"message": oversized}, headers=HEADERS
    )
    assert response.status_code == 422


def test_rejects_missing_message_field(client) -> None:
    response = client.post("/api/v1/chat", json={"session_id": "x"}, headers=HEADERS)
    assert response.status_code == 422


# --- rate limiting -------------------------------------------------------


def test_rate_limiter_blocks_over_the_limit() -> None:
    from app.api.deps import SlidingWindowRateLimiter

    limiter = SlidingWindowRateLimiter(limit_per_minute=3)
    assert all(limiter.check("key")[0] for _ in range(3))

    allowed, retry_after = limiter.check("key")
    assert not allowed
    assert retry_after > 0


def test_rate_limiter_is_per_identity() -> None:
    from app.api.deps import SlidingWindowRateLimiter

    limiter = SlidingWindowRateLimiter(limit_per_minute=1)
    assert limiter.check("a")[0]
    assert limiter.check("b")[0], "a second caller must not be blocked by the first"
    assert not limiter.check("a")[0]
