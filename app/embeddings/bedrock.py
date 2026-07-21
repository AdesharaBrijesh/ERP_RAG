"""AWS Bedrock Titan Text Embeddings v2 - the production embedding path.

Chosen over an external embedding API so embeddings and generation share one
provider, one region and one IAM boundary: no extra vendor contract, no second
egress path for schema metadata.
"""

from __future__ import annotations

import json

from app.embeddings.base import EmbeddingProvider

# Titan v2 supports 256 / 512 / 1024. 512 is the sweet spot for 87 short
# descriptions: negligible quality loss versus 1024, half the storage.
_DEFAULT_DIM = 512


class BedrockTitanEmbeddings(EmbeddingProvider):
    name = "bedrock"

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "ap-south-1",
        dim: int = _DEFAULT_DIM,
    ) -> None:
        import boto3  # imported lazily so local dev needs no AWS SDK config

        self.model_id = model_id
        self.dim = dim
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._supports_dim = "v2" in model_id

    def _embed_one(self, text: str) -> list[float]:
        body: dict[str, object] = {"inputText": text}
        if self._supports_dim:
            body["dimensions"] = self.dim
            body["normalize"] = True
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Titan has no batch endpoint; indexing is offline so serial is fine.
        return [self._embed_one(t) for t in texts]
