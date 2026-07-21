"""AWS Bedrock provider - the production target.

Uses the Converse API, which normalises the request/response shape across
Bedrock model families and reports token usage directly, so cost logging does
not depend on model-specific payload parsing.
"""

from __future__ import annotations

import time

from app.config import get_settings
from app.llm.base import LLMError, LLMProvider, LLMResult


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(
        self,
        model_id: str = "us.meta.llama3-3-70b-instruct-v1:0",
        region: str = "ap-south-1",
        timeout: float = 45.0,
    ) -> None:
        import boto3
        from botocore.config import Config

        self.model_id = model_id
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                read_timeout=timeout,
                connect_timeout=10,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> LLMResult:
        started = time.perf_counter()
        try:
            response = self._client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"bedrock call failed: {exc}") from exc

        content = response["output"]["message"]["content"]
        text = "".join(block.get("text", "") for block in content).strip()
        usage = response.get("usage", {})
        return LLMResult(
            text=text,
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            model_id=self.model_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def build_from_settings() -> BedrockProvider:
    settings = get_settings()
    return BedrockProvider(
        model_id=settings.bedrock_model_id,
        region=settings.bedrock_region,
        timeout=settings.llm_timeout_seconds,
    )
