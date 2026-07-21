"""Conversational answer formatting.

A hard requirement, not polish: raw rows are an explicit non-goal. This is the
second LLM call and it deliberately never sees the schema or the SQL - only
the question and the rows - which is what keeps it cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.core.logging import get_logger
from app.db.executor import QueryResult
from app.llm.base import LLMError, LLMProvider, LLMResult
from app.pipeline.prompts import FORMATTER_SYSTEM, build_formatter_user_prompt

log = get_logger(__name__)


@dataclass
class FormattedAnswer:
    message: str
    llm: LLMResult | None = None
    fell_back: bool = False


def format_answer(
    llm: LLMProvider,
    message: str,
    result: QueryResult,
    history_block: str = "",
    max_tokens: int = 600,
) -> FormattedAnswer:
    settings = get_settings()
    user_prompt = build_formatter_user_prompt(
        message=message,
        result=result,
        max_rows=settings.sql_max_rows_to_llm,
        history_block=history_block,
    )
    try:
        llm_result = llm.complete(
            system=FORMATTER_SYSTEM,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=0.2,
        )
    except LLMError as exc:
        # Never drop the answer on the floor: degrade to a deterministic
        # rendering rather than failing the whole request.
        log.warning("formatter call failed, using fallback", extra={"error": str(exc)})
        return FormattedAnswer(message=_fallback(result), fell_back=True)

    text = llm_result.text.strip()
    if not text:
        return FormattedAnswer(
            message=_fallback(result), llm=llm_result, fell_back=True
        )
    return FormattedAnswer(message=text, llm=llm_result)


def _fallback(result: QueryResult) -> str:
    """Deterministic Markdown, used only when the formatter LLM call fails."""
    if result.is_empty:
        return "I could not find anything matching that."

    if result.row_count == 1 and len(result.columns) == 1:
        column = result.columns[0]
        value = result.rows[0][column]
        return f"**{_humanise(column)}:** {value}"

    header = "| " + " | ".join(_humanise(c) for c in result.columns) + " |"
    divider = "| " + " | ".join("---" for _ in result.columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(c, "")) for c in result.columns) + " |"
        for row in result.rows[:20]
    ]
    note = "\n\nShowing the first 20 rows." if result.row_count > 20 else ""
    return "\n".join([header, divider, *body]) + note


def _humanise(column: str) -> str:
    return column.replace("_", " ").strip().title()
