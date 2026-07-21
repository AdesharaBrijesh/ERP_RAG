"""Per-query table retrieval - the step that keeps the prompt at ~1-3k tokens
instead of the ~30k a full 87-table schema dump would cost.

Three signals, combined:
  1. dense vector similarity over the embedded table descriptions
  2. BM25-style lexical overlap + exact glossary phrase hits
  3. foreign-key expansion, so selecting `sales_orders` also pulls in
     `sales_order_items` and the JOIN the question actually needs is possible
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import Engine

from app.config import get_settings
from app.core.logging import get_logger
from app.db.introspect import TableInfo, build_pruned_schema, introspect_schema
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.retrieval.descriptions import weighted_keywords_for
from app.retrieval.lexical import LexicalIndex
from app.retrieval.store import VectorStore, get_vector_store

log = get_logger(__name__)

VECTOR_WEIGHT = 0.5
LEXICAL_WEIGHT = 0.5


@dataclass
class RetrievedTable:
    name: str
    score: float
    description: str
    vector_score: float = 0.0
    lexical_score: float = 0.0
    via: str = "hybrid"  # hybrid | fk-expansion


@dataclass
class RetrievalResult:
    tables: list[RetrievedTable]
    pruned_schema: str
    duration_ms: int
    top_score: float
    # The introspected schema for the selected tables. The router uses this to
    # include only the domain conventions these tables can actually need.
    table_infos: list[TableInfo] = field(default_factory=list)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]


class TableRetriever:
    def __init__(
        self,
        engine: Engine,
        provider: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.engine = engine
        self.provider = provider or get_embedding_provider()
        self.store = store or get_vector_store(engine)
        self._schema: dict[str, TableInfo] = {}
        self._lexical = LexicalIndex()
        self.refresh()

    def refresh(self) -> None:
        """Load the schema snapshot into memory. Startup / reindex only."""
        settings = get_settings()
        tables = introspect_schema(
            self.engine, schema=settings.db_schema, sample_enum_values=True
        )
        self._schema = {t.name: t for t in tables}
        self._lexical = LexicalIndex.build(
            {t.name: weighted_keywords_for(t) for t in tables}
        )
        log.info("schema snapshot loaded", extra={"table_count": len(self._schema)})

    @property
    def schema(self) -> dict[str, TableInfo]:
        return self._schema

    def build_query_text(self, message: str, history: list[str] | None = None) -> str:
        """Retrieval reads the conversation, not just the last message.

        "and how about last month?" carries no retrievable nouns on its own;
        the preceding turns are what make it resolvable. The current message is
        repeated so it still dominates the older context.
        """
        parts = [message, message]
        if history:
            parts.extend(history[-3:])
        return " ".join(parts)

    def retrieve(
        self,
        message: str,
        history: list[str] | None = None,
        k: int | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        settings = get_settings()
        k = k or settings.retrieval_top_k

        query_text = self.build_query_text(message, history)

        # Over-fetch from the vector store so the lexical signal can promote a
        # table that dense similarity ranked just outside the cut.
        candidates = self.store.search(self.provider.embed_query(query_text), k=k * 4)
        vector_scores = {c.table_name: max(c.score, 0.0) for c in candidates}
        descriptions = {c.table_name: c.description for c in candidates}

        lexical_scores = self._lexical.score(query_text)

        best_vector = max(vector_scores.values(), default=0.0) or 1.0
        combined: dict[str, RetrievedTable] = {}
        for name in set(vector_scores) | set(lexical_scores):
            vec = vector_scores.get(name, 0.0) / best_vector
            lex = lexical_scores.get(name, 0.0)
            combined[name] = RetrievedTable(
                name=name,
                score=VECTOR_WEIGHT * vec + LEXICAL_WEIGHT * lex,
                description=descriptions.get(name, ""),
                vector_score=vector_scores.get(name, 0.0),
                lexical_score=lex,
            )

        ranked = sorted(combined.values(), key=lambda t: -t.score)
        selected = [t for t in ranked[:k] if t.score >= settings.retrieval_min_score]
        if not selected and ranked:
            # Never return nothing: hand the router the single best guess and
            # let it decide whether to ask for clarification instead.
            selected = ranked[:1]

        if settings.retrieval_fk_expansion:
            selected = self._expand_via_foreign_keys(selected, combined)

        selected = selected[: settings.retrieval_max_tables]
        pruned = self._build_pruned_schema(selected)

        return RetrievalResult(
            tables=selected,
            pruned_schema=pruned,
            duration_ms=int((time.perf_counter() - started) * 1000),
            top_score=selected[0].score if selected else 0.0,
            table_infos=[
                self._schema[t.name] for t in selected if t.name in self._schema
            ],
        )

    def _expand_via_foreign_keys(
        self, selected: list[RetrievedTable], scored: dict[str, RetrievedTable]
    ) -> list[RetrievedTable]:
        """Pull in the detail tables a JOIN would need.

        A question about order value needs `sales_order_items`, but the words
        in it only ever match `sales_orders`. Header/detail pairs are the most
        common shape in this ERP, so they are expanded first.
        """
        settings = get_settings()
        chosen = {t.name for t in selected}
        budget = settings.retrieval_max_tables - len(chosen)
        if budget <= 0:
            return selected

        additions: list[RetrievedTable] = []
        for table in list(selected):
            info = self._schema.get(table.name)
            if info is None:
                continue

            neighbours = set(info.related_tables)
            # Children: tables holding an FK back to this one (detail lines).
            neighbours |= {
                other.name
                for other in self._schema.values()
                if table.name in other.related_tables
            }

            for neighbour in sorted(neighbours):
                if neighbour in chosen or len(additions) >= budget:
                    continue
                if not self._is_useful_neighbour(table.name, neighbour, scored):
                    continue
                chosen.add(neighbour)
                additions.append(
                    RetrievedTable(
                        name=neighbour,
                        score=table.score * 0.5,
                        description=scored[neighbour].description
                        if neighbour in scored
                        else "",
                        via="fk-expansion",
                    )
                )
        return selected + additions

    _LOOKUP_TABLES = frozenset({"entity_values", "entity_types", "companies", "tenants"})

    def _is_useful_neighbour(
        self, parent: str, neighbour: str, scored: dict[str, RetrievedTable]
    ) -> bool:
        # Generic lookup tables link to everything; adding them by FK alone
        # would flood the prompt. They come in only on a real query match.
        if neighbour in self._LOOKUP_TABLES:
            return False
        stem = parent[:-1] if parent.endswith("s") else parent
        is_detail_child = neighbour.startswith(stem) and neighbour != parent
        is_named_parent = parent.startswith(neighbour[:-1] if neighbour.endswith("s") else neighbour)
        already_relevant = scored.get(neighbour, RetrievedTable("", 0.0, "")).score >= 0.25
        return is_detail_child or is_named_parent or already_relevant

    def _build_pruned_schema(self, selected: list[RetrievedTable]) -> str:
        infos = [
            self._schema[t.name] for t in selected if t.name in self._schema
        ]
        return build_pruned_schema(infos)


_retriever: TableRetriever | None = None


def get_retriever(engine: Engine | None = None) -> TableRetriever:
    global _retriever
    if _retriever is None:
        if engine is None:
            from app.db.engine import get_engine

            engine = get_engine()
        _retriever = TableRetriever(engine)
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None
