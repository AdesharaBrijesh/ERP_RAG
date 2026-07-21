"""Auth and rate limiting.

API-key auth via `X-API-Key`, matching a simple shared-secret model between
internal services. If the backend team standardises on JWT, swap
``require_api_key`` for a verifier - the dependency signature does not change.
"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Header, HTTPException, Request, status

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    settings = get_settings()
    if not settings.auth_enabled:
        return "anonymous"

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-API-Key header",
        )
    # compare_digest against each configured key: constant time, no early exit
    # that would leak key length or prefix through timing.
    for key in settings.api_keys:
        if hmac.compare_digest(x_api_key, key):
            return key

    log.warning("rejected request with invalid api key")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key"
    )


class SlidingWindowRateLimiter:
    """In-process sliding window.

    Adequate for a single instance. Behind more than one replica this becomes
    per-replica; move it to Redis (same store as SESSION_BACKEND=redis) at that
    point rather than raising the limit.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, identity: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._hits[identity]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(1, int(60 - (now - bucket[0])))
                return False, retry_after
            bucket.append(now)
            return True, 0


_limiter: SlidingWindowRateLimiter | None = None


def get_limiter() -> SlidingWindowRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindowRateLimiter(get_settings().rate_limit_per_minute)
    return _limiter


async def enforce_rate_limit(request: Request) -> None:
    settings = get_settings()
    if settings.rate_limit_per_minute <= 0:
        return

    identity = request.headers.get("x-api-key") or (
        request.client.host if request.client else "unknown"
    )
    allowed, retry_after = get_limiter().check(identity)
    if not allowed:
        log.warning("rate limit exceeded", extra={"retry_after_s": retry_after})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


def reset_limiter() -> None:
    global _limiter
    _limiter = None
