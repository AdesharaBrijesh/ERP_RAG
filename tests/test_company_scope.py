"""SECURITY: multi-company isolation.

This ERP holds more than one company. Isolation is enforced by row-level
security keyed on a transaction-local setting the service controls - never by
asking the model to remember a WHERE clause. These tests assert that the model
cannot reach another company's rows even when its SQL explicitly asks for them.

Skipped unless the RLS policies are installed
(`python -m scripts.enable_company_rls --enable`).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db.executor import execute_query
from app.db.guard import validate_sql
from tests.conftest import requires_db


def _rls_enabled(engine) -> bool:
    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT 1 FROM pg_policies "
                    "WHERE policyname = 'rag_company_scope' LIMIT 1"
                )
            ).scalar()
        )


@pytest.fixture(scope="module")
def rls(engine):
    if not _rls_enabled(engine):
        pytest.skip("company RLS not installed; run scripts.enable_company_rls --enable")
    return True


def _stock_rows(company_id: str | None) -> int:
    guarded = validate_sql("SELECT count(*) AS n FROM item_stocks")
    return execute_query(guarded, company_id=company_id).rows[0]["n"]


@requires_db
def test_unscoped_request_sees_every_company(rls) -> None:
    assert _stock_rows(None) == _stock_rows("1") + _stock_rows("2")


@requires_db
def test_scoping_narrows_the_result(rls) -> None:
    total = _stock_rows(None)
    assert 0 < _stock_rows("1") < total
    assert 0 < _stock_rows("2") < total


@requires_db
def test_model_cannot_escape_its_company_scope(rls) -> None:
    """The attack that matters: SQL that explicitly names the other company.

    RLS filters first, so the predicate matches nothing rather than reaching
    across the tenant boundary.
    """
    guarded = validate_sql("SELECT count(*) AS n FROM item_stocks WHERE company_id = 2")
    assert execute_query(guarded, company_id="1").rows[0]["n"] == 0
    assert execute_query(guarded, company_id="2").rows[0]["n"] > 0


@requires_db
def test_scope_does_not_leak_between_pooled_connections(rls) -> None:
    """The setting is transaction-local, so the next request that borrows this
    pooled connection must not inherit the previous caller's company."""
    scoped = _stock_rows("1")
    unscoped_after = _stock_rows(None)
    assert unscoped_after > scoped


@requires_db
def test_company_id_must_be_numeric() -> None:
    from pydantic import ValidationError

    from app.api.schemas import ChatRequest

    assert ChatRequest(message="hi", company_id="1").company_id == "1"
    assert ChatRequest(message="hi", company_id="  ").company_id is None
    with pytest.raises(ValidationError):
        # It is interpolated into a Postgres setting and cast to bigint.
        ChatRequest(message="hi", company_id="1 OR true")
