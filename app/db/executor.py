"""Read-only execution of guarded SQL."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import date, datetime, time as dt_time
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from app.config import get_settings
from app.core.logging import get_logger
from app.db.engine import get_ro_engine
from app.db.guard import GuardedSql

log = get_logger(__name__)


class SqlExecutionError(Exception):
    """A guarded query failed at the database. Message is safe to log, not to
    show verbatim to an end user."""


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    duration_ms: int = 0

    @property
    def is_empty(self) -> bool:
        return self.row_count == 0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        # Keep numbers as numbers so the formatter can reason about magnitude.
        return float(value)
    if isinstance(value, (datetime, date, dt_time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    return str(value)


def execute_query(guarded: GuardedSql) -> QueryResult:
    """Run a validated SELECT on the read-only connection.

    Only ever called with the output of ``validate_sql``; the type signature
    makes it awkward to call with an unvalidated string by accident.
    """
    settings = get_settings()
    started = time.perf_counter()

    try:
        with get_ro_engine().connect() as conn:
            # Explicit read-only transaction, belt-and-braces with the engine
            # level setting, in case a pooler reset the session.
            conn.execute(text("SET TRANSACTION READ ONLY"))
            cursor = conn.execute(text(guarded.sql))
            columns = list(cursor.keys())
            # Fetch one extra row to detect truncation without a second query.
            fetched = cursor.fetchmany(settings.sql_max_rows + 1)
    except DBAPIError as exc:
        raise SqlExecutionError(_clean_db_error(exc)) from exc
    except SQLAlchemyError as exc:  # noqa: BLE001
        raise SqlExecutionError(str(exc).splitlines()[0][:400]) from exc

    truncated = len(fetched) > settings.sql_max_rows
    rows = fetched[: settings.sql_max_rows]

    result = QueryResult(
        columns=columns,
        rows=[{col: _jsonable(val) for col, val in zip(columns, row, strict=False)} for row in rows],
        row_count=len(rows),
        truncated=truncated or guarded.limit_applied and len(rows) == settings.sql_max_rows,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    log.info(
        "sql executed",
        extra={
            "row_count": result.row_count,
            "truncated": result.truncated,
            "sql_duration_ms": result.duration_ms,
        },
    )
    return result


def _clean_db_error(exc: DBAPIError) -> str:
    """Postgres errors are verbose; keep the first useful line."""
    message = str(exc.orig) if exc.orig else str(exc)
    first = message.strip().splitlines()[0] if message.strip() else "database error"
    if "canceling statement due to statement timeout" in message:
        return "query exceeded the statement timeout"
    if "permission denied" in message.lower():
        return "permission denied for the read-only role"
    return first[:400]
