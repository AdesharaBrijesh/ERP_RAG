"""SECURITY tests for the SELECT-only guard.

Each rejection case is something an LLM could plausibly emit - by
hallucination, by following a user's "ignore that and delete..." instruction,
or by writing a legitimate-looking SELECT that reads the filesystem.
"""

from __future__ import annotations

import pytest

from app.db.guard import SqlGuardError, validate_sql

# --- statements that must be allowed through ------------------------------

ALLOWED = [
    "SELECT count(*) FROM items",
    "select code, name from items where is_deleted = false order by name",
    """
    SELECT w.name, SUM(s.current_qty) AS total_qty
    FROM item_stocks s
    JOIN warehouses w ON w.id = s.warehouse_id
    WHERE s.is_deleted = false
    GROUP BY w.name
    """,
    "WITH recent AS (SELECT * FROM grn WHERE is_deleted = false) SELECT count(*) FROM recent",
    # A literal containing a forbidden word must not trip the scanner.
    "SELECT id FROM items WHERE name ILIKE '%delete me%'",
    "SELECT value_name FROM entity_values WHERE value_code = 'SET'",
    # Comments are stripped, not treated as instructions.
    "SELECT 1 -- drop table items\n",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only_selects(sql: str) -> None:
    guarded = validate_sql(sql)
    assert guarded.sql.lower().lstrip().startswith(("select", "with"))


# --- statements that must be rejected -------------------------------------

REJECTED = [
    pytest.param("DELETE FROM items", id="delete"),
    pytest.param("UPDATE items SET name = 'x'", id="update"),
    pytest.param("INSERT INTO items (code) VALUES ('x')", id="insert"),
    pytest.param("DROP TABLE items", id="drop"),
    pytest.param("TRUNCATE items", id="truncate"),
    pytest.param("ALTER TABLE items ADD COLUMN x int", id="alter"),
    pytest.param("CREATE TABLE hack (x int)", id="create"),
    pytest.param("GRANT ALL ON items TO public", id="grant"),
    pytest.param("REVOKE SELECT ON items FROM erp_rag_ro", id="revoke"),
    # Stacked statements: the classic way a write rides along with a read.
    pytest.param("SELECT 1; DROP TABLE items", id="stacked-drop"),
    pytest.param("SELECT 1; DELETE FROM items;", id="stacked-delete"),
    # SELECT ... INTO creates a table despite looking like a read.
    pytest.param("SELECT * INTO backup_items FROM items", id="select-into"),
    # Data-modifying CTE.
    pytest.param(
        "WITH d AS (DELETE FROM items RETURNING id) SELECT count(*) FROM d",
        id="cte-delete",
    ),
    # Filesystem and process access from inside a SELECT.
    pytest.param("SELECT pg_read_file('/etc/passwd')", id="read-file"),
    pytest.param("SELECT pg_ls_dir('/')", id="ls-dir"),
    pytest.param("SELECT pg_sleep(60)", id="sleep"),
    pytest.param("SELECT lo_import('/etc/passwd')", id="lo-import"),
    pytest.param("SELECT dblink('...', 'select 1')", id="dblink"),
    pytest.param("SELECT pg_terminate_backend(1)", id="terminate-backend"),
    # Session tampering that would disable the read-only transaction.
    pytest.param("SET default_transaction_read_only = off", id="set"),
    pytest.param("SELECT set_config('transaction_read_only','off',false)", id="set-config"),
    # Transaction control.
    pytest.param("BEGIN; DELETE FROM items; COMMIT;", id="explicit-transaction"),
    pytest.param("COPY items TO '/tmp/out.csv'", id="copy"),
    pytest.param("CALL some_procedure()", id="call"),
    pytest.param("DO $$ BEGIN PERFORM 1; END $$", id="do-block"),
    pytest.param("VACUUM FULL items", id="vacuum"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="blank"),
    pytest.param("I could not write a query for that", id="prose"),
]


@pytest.mark.parametrize("sql", REJECTED)
def test_rejects_non_select(sql: str) -> None:
    with pytest.raises(SqlGuardError):
        validate_sql(sql)


def test_rejects_write_hidden_behind_markdown_fence() -> None:
    with pytest.raises(SqlGuardError):
        validate_sql("```sql\nDELETE FROM items;\n```")


def test_strips_markdown_fence_from_valid_query() -> None:
    guarded = validate_sql("```sql\nSELECT count(*) FROM items\n```")
    assert guarded.sql.lower().startswith("select")
    assert "```" not in guarded.sql


def test_strips_sql_label_prefix() -> None:
    guarded = validate_sql("SQL: SELECT count(*) FROM items")
    assert guarded.sql.lower().startswith("select")


# --- row limiting ---------------------------------------------------------


def test_appends_limit_when_missing() -> None:
    guarded = validate_sql("SELECT id FROM stock_ledger", max_rows=50)
    assert guarded.limit_applied
    assert "LIMIT 50" in guarded.sql


def test_keeps_smaller_existing_limit() -> None:
    guarded = validate_sql("SELECT id FROM items LIMIT 10", max_rows=200)
    assert not guarded.limit_applied
    assert guarded.sql.count("LIMIT") == 1
    assert "LIMIT 10" in guarded.sql


def test_clamps_oversized_limit() -> None:
    guarded = validate_sql("SELECT id FROM stock_ledger LIMIT 100000", max_rows=200)
    assert guarded.limit_applied
    assert "LIMIT 200" in guarded.sql
    assert "100000" not in guarded.sql


def test_preserves_offset_when_clamping() -> None:
    guarded = validate_sql("SELECT id FROM items LIMIT 5000 OFFSET 20", max_rows=200)
    assert "LIMIT 200" in guarded.sql
    assert "OFFSET 20" in guarded.sql


def test_trailing_semicolon_is_stripped() -> None:
    guarded = validate_sql("SELECT count(*) FROM items;")
    assert ";" not in guarded.sql
