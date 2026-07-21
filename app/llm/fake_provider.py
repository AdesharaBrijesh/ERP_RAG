"""Deterministic offline provider.

Lets the whole pipeline - routing, guardrails, execution, formatting, session
state - run in CI and in local tests with no API key and no network. It
recognises the prompt type from a marker the prompt builders emit and returns
a canned, well-formed response.
"""

from __future__ import annotations

import json

from app.llm.base import LLMProvider, LLMResult, approx_tokens


class FakeProvider(LLMProvider):
    name = "fake"
    model_id = "fake-model"

    def __init__(self, scripted: list[str] | None = None) -> None:
        # Optional queue of exact responses, for tests that need a specific path.
        self._scripted = list(scripted or [])
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> LLMResult:
        self.calls.append((system, user))
        if self._scripted:
            text = self._scripted.pop(0)
        elif "ROUTING TASK" in system:
            text = self._route(user)
        else:
            text = "Here is a summary of what the data shows."

        return LLMResult(
            text=text,
            input_tokens=approx_tokens(system + user),
            output_tokens=approx_tokens(text),
            model_id=self.model_id,
            latency_ms=1,
        )

    @staticmethod
    def _route(user: str) -> str:
        lowered = user.lower()
        if "table" not in lowered or "ambiguous" in lowered:
            return json.dumps(
                {
                    "decision": "clarify",
                    "clarifying_question": "Could you tell me a bit more about what you need?",
                    "tables_used": [],
                }
            )
        return json.dumps(
            {
                "decision": "sql",
                "sql": "SELECT count(*) AS total FROM items WHERE is_deleted = false",
                "tables_used": ["items"],
            }
        )
