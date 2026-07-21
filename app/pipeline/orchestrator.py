"""The per-query pipeline.

    retrieve top-k tables
      -> route (SQL | clarification), using pruned schema + history
      -> validate SQL (SELECT-only guard)
      -> execute on the read-only role
      -> format into a conversational answer

Every path returns a ChatOutcome carrying the observability payload the spec
asks for: tables retrieved, tokens in/out, cost, latency, and which branch the
query took.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from app.config import get_settings
from app.core.logging import get_logger
from app.db.executor import QueryResult, SqlExecutionError, execute_query
from app.db.guard import SqlGuardError, validate_sql
from app.llm.base import LLMError, LLMProvider
from app.llm.factory import get_llm
from app.pipeline.formatter import format_answer
from app.pipeline.router import repair, route
from app.retrieval.retriever import TableRetriever, get_retriever
from app.session.base import SessionStore
from app.session.factory import get_session_store

log = get_logger(__name__)

OutcomeType = Literal["answer", "clarification_needed", "error"]
Path = Literal[
    "direct_answer",
    "repaired_answer",
    "clarification",
    "guard_rejected",
    "sql_error",
    "llm_error",
]


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    cost_inr: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.input += input_tokens
        self.output += output_tokens
        self.cost_inr = round(self.cost_inr + cost, 6)


@dataclass
class ChatOutcome:
    session_id: str
    type: OutcomeType
    message: str
    sql_generated: str | None = None
    tables_used: list[str] = field(default_factory=list)
    tables_retrieved: list[str] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    path: Path = "direct_answer"
    row_count: int = 0
    latency_ms: int = 0
    retrieval_ms: int = 0
    sql_ms: int = 0


# Shown instead of a stack trace when the model writes SQL the database
# rejects. The detail goes to the logs, not to the user.
_SQL_ERROR_MESSAGE = (
    "I was not able to pull that one together. Could you try asking it a "
    "slightly different way?"
)
_GUARD_ERROR_MESSAGE = (
    "I can only look information up, not change anything, so I could not run "
    "that. Try asking me to show or count something instead."
)
_LLM_ERROR_MESSAGE = (
    "I am having trouble reaching my language model right now. Please try again "
    "in a moment."
)


class ChatPipeline:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        retriever: TableRetriever | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._sessions = sessions

    @property
    def llm(self) -> LLMProvider:
        return self._llm or get_llm()

    @property
    def retriever(self) -> TableRetriever:
        return self._retriever or get_retriever()

    @property
    def sessions(self) -> SessionStore:
        return self._sessions or get_session_store()

    def handle(
        self, session_id: str, message: str, user_id: str | None = None
    ) -> ChatOutcome:
        started = time.perf_counter()
        settings = get_settings()
        tokens = TokenUsage()

        state = self.sessions.get(session_id, user_id=user_id)
        history_block = state.history_block(settings.history_max_turns)
        pending = state.pending_clarification

        # The retrieval query is the answer to a pending question plus the
        # question it was answering - "the second one" retrieves nothing alone.
        retrieval_message = message
        if pending and state.pending_intent:
            retrieval_message = f"{state.pending_intent} {message}"

        retrieval = self.retriever.retrieve(
            retrieval_message, history=state.recent_user_messages()
        )

        self.sessions.append_message(session_id, "user", message)

        outcome = self._run(
            session_id=session_id,
            message=message,
            retrieval=retrieval,
            history_block=history_block,
            pending=pending,
            pending_intent=state.pending_intent,
            tokens=tokens,
        )

        outcome.tables_retrieved = retrieval.table_names
        outcome.retrieval_ms = retrieval.duration_ms
        outcome.latency_ms = int((time.perf_counter() - started) * 1000)

        self.sessions.append_message(session_id, "assistant", outcome.message)

        log.info(
            "chat handled",
            extra={
                "path": outcome.path,
                "outcome_type": outcome.type,
                "tables_retrieved": outcome.tables_retrieved,
                "tables_used": outcome.tables_used,
                "row_count": outcome.row_count,
                "input_tokens": outcome.tokens.input,
                "output_tokens": outcome.tokens.output,
                "cost_estimate_inr": outcome.tokens.cost_inr,
                "retrieval_ms": outcome.retrieval_ms,
                "sql_ms": outcome.sql_ms,
                "latency_ms": outcome.latency_ms,
                "user_id": user_id,
            },
        )
        return outcome

    def _run(
        self,
        session_id: str,
        message: str,
        retrieval,
        history_block: str,
        pending: str | None,
        pending_intent: str | None,
        tokens: TokenUsage,
    ) -> ChatOutcome:
        try:
            decision = route(
                llm=self.llm,
                message=message,
                retrieval=retrieval,
                history_block=history_block,
                pending_clarification=pending,
            )
        except LLMError as exc:
            log.error("router llm call failed", extra={"error": str(exc)})
            return ChatOutcome(
                session_id=session_id,
                type="error",
                message=_LLM_ERROR_MESSAGE,
                path="llm_error",
                tokens=tokens,
            )

        if decision.llm:
            tokens.add(
                decision.llm.input_tokens, decision.llm.output_tokens, decision.llm.cost_inr
            )

        if not decision.is_sql:
            question = decision.clarifying_question or ""
            # Remember what the user originally wanted, so their reply resumes
            # this intent rather than being routed as a brand new question.
            self.sessions.set_pending_clarification(
                session_id, question, pending_intent or message
            )
            return ChatOutcome(
                session_id=session_id,
                type="clarification_needed",
                message=question,
                tables_used=decision.tables_used,
                tokens=tokens,
                path="clarification",
            )

        # A query was produced, so any pending clarification is resolved.
        self.sessions.clear_pending_clarification(session_id)

        try:
            guarded = validate_sql(decision.sql or "")
        except SqlGuardError as exc:
            log.warning(
                "generated SQL rejected by guard",
                extra={"guard_reason": exc.reason, "rejected_sql": (decision.sql or "")[:500]},
            )
            return ChatOutcome(
                session_id=session_id,
                type="error",
                message=_GUARD_ERROR_MESSAGE,
                sql_generated=decision.sql,
                tables_used=decision.tables_used,
                tokens=tokens,
                path="guard_rejected",
            )

        repaired = False
        try:
            result: QueryResult = execute_query(guarded)
        except SqlExecutionError as exc:
            log.warning(
                "sql execution failed",
                extra={"db_error": str(exc), "failed_sql": guarded.sql[:500]},
            )
            outcome = self._attempt_repair(
                session_id=session_id,
                message=message,
                retrieval=retrieval,
                failed_sql=guarded.sql,
                error=str(exc),
                decision=decision,
                tokens=tokens,
            )
            if isinstance(outcome, ChatOutcome):
                return outcome
            guarded, result = outcome
            repaired = True

        answer = format_answer(
            llm=self.llm, message=message, result=result, history_block=history_block
        )
        if answer.llm:
            tokens.add(
                answer.llm.input_tokens, answer.llm.output_tokens, answer.llm.cost_inr
            )

        return ChatOutcome(
            session_id=session_id,
            type="answer",
            message=answer.message,
            sql_generated=guarded.sql,
            tables_used=decision.tables_used,
            tokens=tokens,
            path="repaired_answer" if repaired else "direct_answer",
            row_count=result.row_count,
            sql_ms=result.duration_ms,
        )

    def _attempt_repair(
        self,
        session_id: str,
        message: str,
        retrieval,
        failed_sql: str,
        error: str,
        decision,
        tokens: TokenUsage,
    ) -> "ChatOutcome | tuple[object, QueryResult]":
        """One corrective LLM pass. Returns the retry's (guarded, result) on
        success, or a terminal ChatOutcome if the repair also fails."""
        failure = ChatOutcome(
            session_id=session_id,
            type="error",
            message=_SQL_ERROR_MESSAGE,
            sql_generated=failed_sql,
            tables_used=decision.tables_used,
            tokens=tokens,
            path="sql_error",
        )

        try:
            fixed = repair(
                llm=self.llm,
                message=message,
                retrieval=retrieval,
                failed_sql=failed_sql,
                error=error,
            )
        except LLMError as exc:
            log.error("repair llm call failed", extra={"error": str(exc)})
            return failure

        if fixed.llm:
            tokens.add(fixed.llm.input_tokens, fixed.llm.output_tokens, fixed.llm.cost_inr)

        if not fixed.is_sql:
            # The model decided on reflection that it cannot answer this.
            question = fixed.clarifying_question or ""
            self.sessions.set_pending_clarification(session_id, question, message)
            return ChatOutcome(
                session_id=session_id,
                type="clarification_needed",
                message=question,
                tables_used=fixed.tables_used,
                tokens=tokens,
                path="clarification",
            )

        try:
            guarded_retry = validate_sql(fixed.sql or "")
            result = execute_query(guarded_retry)
        except SqlGuardError as exc:
            log.warning("repaired SQL rejected by guard", extra={"guard_reason": exc.reason})
            return failure
        except SqlExecutionError as exc:
            log.warning("repaired sql also failed", extra={"db_error": str(exc)})
            return failure

        log.info("sql repair succeeded", extra={"row_count": result.row_count})
        return guarded_retry, result


_pipeline: ChatPipeline | None = None


def get_pipeline() -> ChatPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ChatPipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None
