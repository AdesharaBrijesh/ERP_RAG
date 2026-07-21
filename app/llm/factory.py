from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMProvider


@lru_cache
def get_llm() -> LLMProvider:
    provider = get_settings().llm_provider
    if provider == "bedrock":
        from app.llm.bedrock_provider import build_from_settings

        return build_from_settings()
    if provider == "fake":
        from app.llm.fake_provider import FakeProvider

        return FakeProvider()

    from app.llm.groq_provider import build_from_settings

    return build_from_settings()


def reset_llm() -> None:
    get_llm.cache_clear()
