from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider
from app.embeddings.hashing import HashingEmbeddings

log = get_logger(__name__)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    choice = settings.embedding_provider

    if choice == "bedrock":
        try:
            from app.embeddings.bedrock import BedrockTitanEmbeddings

            return BedrockTitanEmbeddings(
                model_id=settings.bedrock_embed_model_id,
                region=settings.bedrock_region,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "bedrock embeddings unavailable, falling back to hashing",
                extra={"error": str(exc)},
            )

    elif choice == "fastembed":
        try:
            from app.embeddings.fastembed_provider import FastEmbedEmbeddings

            return FastEmbedEmbeddings(model_id=settings.fastembed_model)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "fastembed unavailable, falling back to hashing",
                extra={"error": str(exc)},
            )

    return HashingEmbeddings(dim=settings.hashing_dim)
