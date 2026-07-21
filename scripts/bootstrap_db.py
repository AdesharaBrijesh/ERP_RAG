"""Provision the read-only database role used to execute generated SQL.

SECURITY-RELEVANT. This is the layer that holds even if the SELECT-only guard
in ``app/db/guard.py`` is bypassed: the role simply has no grants to write
with. Run once per environment, as a superuser.

    python -m scripts.bootstrap_db --password '<strong-password>'

Grants: CONNECT on the database, USAGE + SELECT on public.* (existing and
future tables), and nothing else. Explicitly revoked: CREATE on every schema,
all privileges on the rag schema (embeddings and chat history are not ERP data
the answering role has any business reading), and the PUBLIC pseudo-role's
default grants.
"""

from __future__ import annotations

import argparse
import getpass
import re
from urllib.parse import quote

from sqlalchemy import text

from app.config import get_settings
from app.db.engine import get_engine

ROLE = "erp_rag_ro"

# CREATE/ALTER ROLE do not accept bind parameters, so the password has to be
# inlined. Restrict it to characters that cannot terminate a SQL string or
# start a statement, then escape, rather than relying on escaping alone.
_SAFE_PASSWORD_RE = re.compile(r"^[A-Za-z0-9!@#$%^&*()_+\-=\[\]{}:,.<>?~|/]{8,128}$")


def _quote_literal(value: str) -> str:
    if not _SAFE_PASSWORD_RE.match(value):
        raise SystemExit(
            "password must be 8-128 characters and must not contain quotes, "
            "backslashes, semicolons or whitespace"
        )
    return "'" + value.replace("'", "''") + "'"


def _database_name(url: str) -> str:
    match = re.search(r"/([^/?]+)(\?|$)", url)
    if not match:
        raise SystemExit(f"could not determine database name from {url!r}")
    return match.group(1)


def bootstrap(password: str, role: str = ROLE) -> str:
    settings = get_settings()
    engine = get_engine()
    database = _database_name(settings.database_url)
    rag_schema = settings.rag_schema

    # AUTOCOMMIT: CREATE ROLE and GRANT cannot run inside the implicit
    # transaction SQLAlchemy would otherwise open.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}
        ).scalar()

        secret = _quote_literal(password)
        if exists:
            conn.execute(text(f'ALTER ROLE "{role}" WITH LOGIN PASSWORD {secret}'))
            print(f"role {role} already existed - password reset")
        else:
            conn.execute(text(f'CREATE ROLE "{role}" WITH LOGIN PASSWORD {secret}'))
            print(f"created role {role}")

        # No inherited write access, no ability to create objects anywhere.
        conn.execute(text(f'ALTER ROLE "{role}" WITH NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT'))
        conn.execute(text(f'ALTER ROLE "{role}" SET default_transaction_read_only = on'))
        conn.execute(text(f'ALTER ROLE "{role}" SET statement_timeout = '
                          f'{settings.sql_statement_timeout_ms}'))
        conn.execute(text(f'ALTER ROLE "{role}" SET idle_in_transaction_session_timeout = 30000'))

        conn.execute(text(f'GRANT CONNECT ON DATABASE "{database}" TO "{role}"'))
        conn.execute(text(f'GRANT USAGE ON SCHEMA public TO "{role}"'))
        conn.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO "{role}"'))
        conn.execute(
            text(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f'GRANT SELECT ON TABLES TO "{role}"'
            )
        )

        # Explicit denials.
        conn.execute(text(f'REVOKE CREATE ON SCHEMA public FROM "{role}"'))
        conn.execute(text('REVOKE CREATE ON SCHEMA public FROM PUBLIC'))
        conn.execute(text(f'REVOKE ALL ON SCHEMA {rag_schema} FROM "{role}"'))
        conn.execute(
            text(f'REVOKE ALL ON ALL TABLES IN SCHEMA {rag_schema} FROM "{role}"')
        )

        print(f"granted SELECT-only on {database}.public to {role}")
        print(f"revoked all access to schema {rag_schema} (chat history, embeddings)")

    host_port = settings.database_url.split("@")[-1].split("/")[0]
    return f"postgresql+psycopg://{role}:{quote(password)}@{host_port}/{database}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--password", help="password for the read-only role")
    parser.add_argument("--role", default=ROLE)
    args = parser.parse_args()

    password = args.password or getpass.getpass(f"password for {args.role}: ")
    if not password:
        raise SystemExit("a password is required")

    url = bootstrap(password, role=args.role)
    print("\nAdd this to your .env:\n")
    print(f"READONLY_DATABASE_URL={url}")


if __name__ == "__main__":
    main()
