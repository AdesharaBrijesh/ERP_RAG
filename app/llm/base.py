"""LLM provider interface.

The service is stateless with respect to the model: history, retrieval and
session state are owned here, and each call ships exactly the context it
needs. That is what makes swapping Groq for Bedrock a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.costs import estimate_cost_inr


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_inr(self) -> float:
        return estimate_cost_inr(self.model_id, self.input_tokens, self.output_tokens)


class LLMError(Exception):
    """Provider call failed after retries."""


class LLMProvider(ABC):
    name: str = "base"
    model_id: str = ""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> LLMResult: ...


def approx_tokens(text: str) -> int:
    """Rough fallback when a provider does not report usage."""
    return max(1, len(text) // 4)
