"""SQL guardrails.

SECURITY-CRITICAL. Every statement produced by the LLM passes through
``validate_sql`` before it reaches a database connection. The model is treated
as an untrusted source: it may hallucinate, and a user may try to talk it into
writing DML. Neither is allowed to reach the ERP.

Defence in depth, three independent layers:

  1. this parser-based validator (SELECT / WITH...SELECT only, single statement)
  2. a database role with no INSERT/UPDATE/DELETE/DDL grants at all
     (see ``scripts/bootstrap_db.py``)
  3. a read-only transaction with a statement timeout on the connection itself
     (see ``app/db/engine.py``)

Layer 1 failing open would still leave 2 and 3. Layer 1 exists because a clear
rejection is a far better user experience than a permission-denied traceback,
and because it is the only layer that can log the attempt with intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlparse
from sqlparse import tokens as T

from app.config import get_settings


class SqlGuardError(Exception):
    """Raised when generated SQL is not a safe, single, read-only SELECT."""

    def __init__(self, reason: str, sql: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.sql = sql


# Statement keywords that must never appear. `INTO` is included because
# `SELECT ... INTO new_table` creates a table - it reads like a SELECT but is
# effectively DDL.
FORBIDDEN_KEYWORDS = frozenset(
    {
        "insert", "update", "delete", "merge", "upsert",
        "drop", "create", "alter", "truncate", "rename", "comment",
        "grant", "revoke", "copy", "call", "do", "execute", "prepare",
        "vacuum", "analyze", "reindex", "cluster", "refresh", "listen",
        "notify", "unlisten", "lock", "begin", "commit", "rollback",
        "savepoint", "set", "reset", "discard", "checkpoint", "load",
        "into", "returning", "attach", "detach",
    }
)

# Functions that read the filesystem, execute SQL, sleep, or touch the server
# process. Harmless-looking inside a SELECT, which is exactly the problem.
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
        "pg_sleep", "pg_sleep_for", "pg_sleep_until",
        "lo_import", "lo_export", "lo_get", "lo_put",
        "dblink", "dblink_exec", "dblink_connect",
        "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
        "pg_rotate_logfile", "set_config", "current_setting",
        "query_to_xml", "query_to_xmlschema", "database_to_xml",
        "pg_read_server_files", "pg_execute_server_program",
    }
)

_FENCE_RE = re.compile(r"^\s*```(?:sql|postgresql|postgres)?\s*|\s*```\s*$", re.I)
_STRING_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", re.S)
_LIMIT_TAIL_RE = re.compile(
    r"\blimit\s+(\d+)\s*(offset\s+\d+\s*)?$", re.I
)


@dataclass(frozen=True)
class GuardedSql:
    sql: str
    limit_applied: bool
    original: str


def _strip_wrapper(sql: str) -> str:
    """Remove markdown fences and stray prose the model may have added."""
    text = sql.strip()
    text = _FENCE_RE.sub("", text)
    text = text.strip()
    # Models sometimes prefix a label despite instructions.
    text = re.sub(r"^(sql|query)\s*[:=]\s*", "", text, flags=re.I).strip()
    return text.rstrip(";").strip()


def _strip_comments_and_strings(sql: str) -> str:
    """Keyword scanning must not see literals or comments.

    Without this, `WHERE status = 'DELETED'` would trip the DELETE check, and
    `-- drop everything` in a comment would too.
    """
    no_comments = sqlparse.format(sql, strip_comments=True)
    return _STRING_RE.sub(" '' ", no_comments)


def _single_statement(sql: str) -> str:
    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) > 1:
        raise SqlGuardError(
            f"expected a single statement, got {len(statements)}", sql
        )
    if not statements:
        raise SqlGuardError("empty statement", sql)
    return statements[0].strip().rstrip(";").strip()


def _assert_starts_with_select(parsed: sqlparse.sql.Statement, sql: str) -> None:
    first = None
    for token in parsed.tokens:
        if token.ttype in (T.Whitespace, T.Newline, T.Comment) or token.is_whitespace:
            continue
        if token.ttype in T.Comment:
            continue
        first = token
        break
    if first is None:
        raise SqlGuardError("no executable tokens", sql)

    keyword = first.value.strip().lower()
    if keyword not in ("select", "with", "("):
        raise SqlGuardError(
            f"only SELECT queries are permitted, statement starts with '{keyword.upper()}'",
            sql,
        )


def _assert_no_forbidden_tokens(sql_for_scan: str, sql: str) -> None:
    parsed = sqlparse.parse(sql_for_scan)
    if not parsed:
        raise SqlGuardError("could not parse statement", sql)

    for token in parsed[0].flatten():
        value = token.value.strip().lower()
        if not value:
            continue
        if token.ttype in (T.Keyword.DML, T.Keyword.DDL):
            if value != "select":
                raise SqlGuardError(
                    f"statement contains a non-SELECT operation: '{value.upper()}'", sql
                )
        elif token.ttype in T.Keyword and value in FORBIDDEN_KEYWORDS:
            raise SqlGuardError(
                f"statement contains forbidden keyword: '{value.upper()}'", sql
            )

    # sqlparse's token classification is not exhaustive across dialects, so a
    # plain word-boundary sweep backs it up. Literals are already stripped.
    lowered = sql_for_scan.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            raise SqlGuardError(
                f"statement contains forbidden keyword: '{keyword.upper()}'", sql
            )
    for function in FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{re.escape(function)}\s*\(", lowered):
            raise SqlGuardError(
                f"statement calls a forbidden function: '{function}'", sql
            )


def _apply_row_limit(sql: str, max_rows: int) -> tuple[str, bool]:
    """Cap the result set. A missing LIMIT on `stock_ledger` is 411k rows."""
    scan_target = _strip_comments_and_strings(sql).strip().rstrip(";").strip()
    match = _LIMIT_TAIL_RE.search(scan_target)
    if match:
        existing = int(match.group(1))
        if existing <= max_rows:
            return sql, False
        # Clamp an over-large LIMIT down to the cap.
        start = _LIMIT_TAIL_RE.search(sql.rstrip().rstrip(";"))
        if start:
            capped = sql.rstrip().rstrip(";")
            capped = capped[: start.start()] + f"LIMIT {max_rows}"
            if match.group(2):
                capped += f" {match.group(2).strip()}"
            return capped, True
    return f"{sql.rstrip().rstrip(';')}\nLIMIT {max_rows}", True


def validate_sql(raw_sql: str, max_rows: int | None = None) -> GuardedSql:
    """Validate and normalise LLM-generated SQL.

    Raises SqlGuardError on anything that is not a single read-only SELECT.
    """
    if max_rows is None:
        max_rows = get_settings().sql_max_rows

    if not raw_sql or not raw_sql.strip():
        raise SqlGuardError("empty SQL", raw_sql)

    cleaned = _strip_wrapper(raw_sql)
    cleaned = _single_statement(cleaned)

    scan_target = _strip_comments_and_strings(cleaned)
    if not scan_target.strip():
        raise SqlGuardError("statement contains no SQL after stripping comments", raw_sql)

    parsed = sqlparse.parse(scan_target)
    if not parsed:
        raise SqlGuardError("could not parse statement", raw_sql)

    _assert_starts_with_select(parsed[0], raw_sql)
    _assert_no_forbidden_tokens(scan_target, raw_sql)

    final_sql, limited = _apply_row_limit(cleaned, max_rows)
    return GuardedSql(sql=final_sql, limit_applied=limited, original=raw_sql)
