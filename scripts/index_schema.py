"""Build (or refresh) the table-description vector index.

    python -m scripts.index_schema           # incremental, only changed tables
    python -m scripts.index_schema --force   # re-embed everything
    python -m scripts.index_schema --show    # print what got indexed

Run on deploy and whenever the ERP schema or the glossary changes.
"""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.core.logging import configure_logging
from app.db.engine import get_engine
from app.retrieval.indexer import index_schema
from app.retrieval.store import get_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-embed every table")
    parser.add_argument("--show", action="store_true", help="print stored descriptions")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    result = index_schema(force=args.force)

    print(
        f"\nIndexed {result.embedded} table(s), skipped {result.skipped} unchanged, "
        f"pruned {result.pruned}."
    )
    print(f"Backend: {result.backend} | model: {result.model} | {result.duration_ms} ms")
    print(f"Total tables in index: {get_vector_store(get_engine()).count()}")

    if args.show:
        from sqlalchemy import text

        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT table_name, description FROM {settings.rag_schema}"
                    ".table_embeddings ORDER BY table_name"
                )
            ).all()
        print()
        for name, description in rows:
            print(f"--- {name}\n{description}\n")


if __name__ == "__main__":
    main()
