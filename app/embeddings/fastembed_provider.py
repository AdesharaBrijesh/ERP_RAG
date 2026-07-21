"""Local dense embeddings via fastembed (ONNX runtime, no torch).

Optional. `pip install fastembed`. Gives real semantic similarity offline,
which is useful for validating retrieval behaviour before Bedrock is wired up.
"""

from __future__ import annotations

from app.embeddings.base import EmbeddingProvider


class FastEmbedEmbeddings(EmbeddingProvider):
    name = "fastembed"

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # type: ignore[import-not-found]

        self.model_id = model_id
        self._model = TextEmbedding(model_name=model_id)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models expect a retrieval instruction prefix on the query side.
        prefix = "Represent this sentence for searching relevant passages: "
        return self.embed_documents([prefix + text])[0]
