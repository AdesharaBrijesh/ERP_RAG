"""Groq provider - local development and staging.

Same Llama 3.3 70B weights as the Bedrock production target, so prompt
behaviour observed locally carries over.
"""

from __future__ import annotations

import time

from app.config import get_settings
from app.core.logging import get_logger
from app.llm.base import LLMError, LLMProvider, LLMResult, approx_tokens

log = get_logger(__name__)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str, model_id: str, timeout: float = 45.0) -> None:
        from groq import Groq

        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Set it in .env, or set LLM_PROVIDER=fake "
                "to run the pipeline without a live model."
            )
        self.model_id = model_id
        # Groq's free tier has a tokens-per-minute ceiling that a ~3k-token
        # routing prompt hits quickly under load. The SDK honours Retry-After
        # on 429, so give it enough attempts to ride out a burst rather than
        # surfacing a rate limit to the user as a model outage.
        self._client = Groq(api_key=api_key, timeout=timeout, max_retries=5)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> LLMResult:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surface provider errors uniformly
            raise LLMError(f"groq call failed: {exc}") from exc

        usage = response.usage
        return LLMResult(
            text=(response.choices[0].message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else approx_tokens(system + user),
            output_tokens=usage.completion_tokens if usage else 0,
            model_id=self.model_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def build_from_settings() -> GroqProvider:
    settings = get_settings()
    return GroqProvider(
        api_key=settings.groq_api_key,
        model_id=settings.groq_model,
        timeout=settings.llm_timeout_seconds,
    )
