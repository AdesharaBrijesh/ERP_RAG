"""Token cost estimation.

Prices are USD per 1M tokens. Kept in one place so the per-query cost target
(Rs 0.21-0.25) can be validated against real logged numbers rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings


@dataclass(frozen=True)
class Price:
    input_per_mtok_usd: float
    output_per_mtok_usd: float


# Model id (or a distinctive substring) -> price.
PRICES: dict[str, Price] = {
    # Groq - local dev / staging
    "llama-3.3-70b-versatile": Price(0.59, 0.79),
    "llama-3.1-8b-instant": Price(0.05, 0.08),
    # AWS Bedrock - production target
    "meta.llama3-3-70b-instruct": Price(0.72, 0.72),
    "meta.llama3-1-70b-instruct": Price(0.72, 0.72),
    "amazon.titan-embed-text-v2": Price(0.02, 0.0),
    "amazon.titan-embed-text-v1": Price(0.10, 0.0),
}

_DEFAULT = Price(0.72, 0.72)


def price_for(model_id: str) -> Price:
    if model_id in PRICES:
        return PRICES[model_id]
    for key, price in PRICES.items():
        if key in model_id:
            return price
    return _DEFAULT


def estimate_cost_inr(model_id: str, input_tokens: int, output_tokens: int) -> float:
    price = price_for(model_id)
    usd = (
        input_tokens * price.input_per_mtok_usd
        + output_tokens * price.output_per_mtok_usd
    ) / 1_000_000
    return round(usd * get_settings().usd_to_inr, 6)
