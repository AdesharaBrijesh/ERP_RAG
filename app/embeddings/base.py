"""Embedding provider interface.

Production target is Bedrock Titan Text Embeddings v2, so embeddings stay
inside the same provider/IAM boundary as the LLM. Local development has no
AWS credentials, hence the two offline alternatives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Vectorises table descriptions (offline) and user queries (per request)."""

    name: str = "base"
    dim: int = 0
    model_id: str = ""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed table descriptions. Called at indexing time only."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a user query. Called once per request - keep it cheap."""
        return self.embed_documents([text])[0]

    @property
    def index_signature(self) -> str:
        """Changing provider or dimensionality invalidates the stored index."""
        return f"{self.name}:{self.model_id}:{self.dim}"
