"""Provision row-level security so company scoping cannot be bypassed.

SECURITY-RELEVANT. This ERP holds more than one company. Asking the LLM to
remember `AND company_id = X` is not isolation - it is a suggestion, and the
one time the model forgets, one company reads another's payroll.

Instead, every table carrying `company_id` gets an RLS policy that filters on
a transaction-local setting (`app.company_id`) which only the service sets, in
``app/db/executor.py``. No query the model can write escapes it.

Two policies per table:

  rag_permissive_all   PERMISSIVE, TO PUBLIC, USING (true)
      Keeps every other role - including the ERP application's own - working
      exactly as before. Without it, enabling RLS would deny all rows to
      non-owner roles.

  rag_company_scope    RESTRICTIVE, TO erp_rag_ro only
      ANDed with the above, and applies to nobody but the chatbot's read-only
      role. When app.company_id is unset the policy passes everything, so
      group-wide questions still work when the caller sends no company_id.

The table owner (postgres) bypasses RLS by default, so the ERP application is
unaffected either way. Reversible with --disable.

    python -m scripts.enable_company_rls --dry-run
    python -m scripts.enable_company_rls --enable
    python -m scripts.enable_company_rls --disable
"""

from __future__ import annotations

import argparse

from sqlalchemy import text

from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.engine import get_engine

log = get_logger(__name__)

PERMISSIVE_POLICY = "rag_permissive_all"
RESTRICTIVE_POLICY = "rag_company_scope"

# When the setting is absent or blank the policy passes every row, so an
# unscoped caller still gets group-wide answers.
_SCOPE_PREDICATE = (
    "nullif(current_setting('app.company_id', true), '') IS NULL "
    "OR company_id = nullif(current_setting('app.company_id', true), '')::bigint"
)


def tables_with_company_id(conn, schema: str) -> list[str]:
    return list(
        conn.execute(
            text(
                """
                SELECT c.table_name
                FROM information_schema.columns c
                JOIN information_schema.tables t
                  ON t.table_name = c.table_name AND t.table_schema = c.table_schema
                WHERE c.table_schema = :schema
                  AND c.column_name = 'company_id'
                  AND t.table_type = 'BASE TABLE'
                ORDER BY c.table_name
                """
            ),
            {"schema": schema},
        ).scalars()
    )


def enable(role: str, dry_run: bool) -> None:
    settings = get_settings()
    schema = settings.db_schema
    engine = get_engine()

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        tables = tables_with_company_id(conn, schema)
        print(f"{len(tables)} table(s) carry company_id in schema {schema}\n")

        for table in tables:
            statements = [
                f'ALTER TABLE "{schema}"."{table}" ENABLE ROW LEVEL SECURITY',
                f'DROP POLICY IF EXISTS {PERMISSIVE_POLICY} ON "{schema}"."{table}"',
                f'CREATE POLICY {PERMISSIVE_POLICY} ON "{schema}"."{table}" '
                f"AS PERMISSIVE FOR SELECT TO PUBLIC USING (true)",
                f'DROP POLICY IF EXISTS {RESTRICTIVE_POLICY} ON "{schema}"."{table}"',
                f'CREATE POLICY {RESTRICTIVE_POLICY} ON "{schema}"."{table}" '
                f'AS RESTRICTIVE FOR SELECT TO "{role}" USING ({_SCOPE_PREDICATE})',
            ]
            if dry_run:
                print(f"-- {table}")
                for statement in statements:
                    print(f"   {statement};")
            else:
                for statement in statements:
                    conn.execute(text(statement))
                print(f"scoped {table}")

    if dry_run:
        print("\n(dry run - nothing was changed)")
    else:
        print(f"\nrow-level company scoping enabled for role {role}")
        print("The service sets app.company_id per transaction; callers that send")
        print("no company_id still receive group-wide answers.")


def disable(dry_run: bool) -> None:
    settings = get_settings()
    schema = settings.db_schema
    engine = get_engine()

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        tables = tables_with_company_id(conn, schema)
        for table in tables:
            statements = [
                f'DROP POLICY IF EXISTS {RESTRICTIVE_POLICY} ON "{schema}"."{table}"',
                f'DROP POLICY IF EXISTS {PERMISSIVE_POLICY} ON "{schema}"."{table}"',
                f'ALTER TABLE "{schema}"."{table}" DISABLE ROW LEVEL SECURITY',
            ]
            if dry_run:
                print(f"-- {table}")
                for statement in statements:
                    print(f"   {statement};")
            else:
                for statement in statements:
                    conn.execute(text(statement))
                print(f"unscoped {table}")

    if not dry_run:
        print("\nrow-level company scoping removed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true")
    group.add_argument("--disable", action="store_true")
    parser.add_argument("--role", default="erp_rag_ro")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING")
    if args.enable:
        enable(args.role, args.dry_run)
    else:
        disable(args.dry_run)


if __name__ == "__main__":
    main()
