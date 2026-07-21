"""Routing / SQL-generation step.

One LLM call decides between three outcomes described in the spec:

  * clear match        -> a single-table SELECT
  * multi-concept      -> a JOIN/UNION across the retrieved tables
  * ambiguous / no fit -> no query at all, a clarifying question instead

The first two are both `decision: "sql"`; the model does not need to
distinguish them, it just needs to write the right query. The third is the one
that must never silently become a guess against production data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from app.core.logging import get_logger
from app.llm.base import LLMProvider, LLMResult
from app.pipeline.prompts import ROUTER_SYSTEM, build_router_user_prompt
from app.retrieval.retriever import RetrievalResult

log = get_logger(__name__)

Decision = Literal["sql", "clarify"]

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_FENCE_RE = re.compile(r"```(?:json|sql)?", re.I)


@dataclass
class RouterDecision:
    decision: Decision
    sql: str | None = None
    clarifying_question: str | None = None
    tables_used: list[str] = field(default_factory=list)
    llm: LLMResult | None = None
    raw: str = ""

    @property
    def is_sql(self) -> bool:
        return self.decision == "sql" and bool(self.sql)


DEFAULT_CLARIFICATION = (
    "I want to make sure I get this right - could you tell me a little more "
    "about what you are looking for?"
)


def route(
    llm: LLMProvider,
    message: str,
    retrieval: RetrievalResult,
    history_block: str = "",
    pending_clarification: str | None = None,
    max_tokens: int = 700,
) -> RouterDecision:
    user_prompt = build_router_user_prompt(
        message=message,
        pruned_schema=retrieval.pruned_schema,
        history_block=history_block,
        pending_clarification=pending_clarification,
    )
    result = llm.complete(
        system=ROUTER_SYSTEM, user=user_prompt, max_tokens=max_tokens, temperature=0.0
    )
    decision = _parse(result.text, retrieval)
    decision.llm = result
    decision.raw = result.text

    log.info(
        "router decision",
        extra={
            "decision": decision.decision,
            "tables_retrieved": retrieval.table_names,
            "tables_used": decision.tables_used,
            "router_input_tokens": result.input_tokens,
            "router_output_tokens": result.output_tokens,
            "router_latency_ms": result.latency_ms,
        },
    )
    return decision


def _parse(text: str, retrieval: RetrievalResult) -> RouterDecision:
    """Tolerant parsing. A malformed response must degrade to a clarification,
    never to an unguarded query."""
    cleaned = _FENCE_RE.sub("", text or "").strip()

    payload: dict | None = None
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        match = _JSON_BLOCK_RE.search(cleaned)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        decision = str(payload.get("decision", "")).lower().strip()
        sql = payload.get("sql")
        question = payload.get("clarifying_question")
        tables = payload.get("tables_used") or []
        if not isinstance(tables, list):
            tables = []
        tables = [str(t) for t in tables]

        if decision == "sql" and isinstance(sql, str) and sql.strip():
            return RouterDecision(
                decision="sql",
                sql=sql.strip(),
                tables_used=tables or retrieval.table_names[:3],
            )
        if decision == "clarify":
            return RouterDecision(
                decision="clarify",
                clarifying_question=(question or DEFAULT_CLARIFICATION).strip(),
                tables_used=tables,
            )

    # The model ignored the JSON contract but produced a bare query.
    if re.match(r"^\s*(select|with)\b", cleaned, re.I):
        return RouterDecision(
            decision="sql", sql=cleaned, tables_used=retrieval.table_names[:3]
        )

    log.warning(
        "router returned unparseable output; falling back to clarification",
        extra={"raw_preview": cleaned[:300]},
    )
    return RouterDecision(decision="clarify", clarifying_question=DEFAULT_CLARIFICATION)
