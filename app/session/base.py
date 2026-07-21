"""Session state.

The LLM call is stateless; this service owns the conversation. Two things are
persisted per session:

  * a rolling message history, trimmed to the last N turns with a cheap
    non-LLM digest of older topics so token growth stays bounded
  * pending-clarification state, so that when the assistant asks a question,
    the user's next message resumes that intent instead of starting a fresh
    routing pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime | None = None


@dataclass
class SessionState:
    session_id: str
    user_id: str | None = None
    messages: list[Message] = field(default_factory=list)
    topic_digest: str = ""
    pending_clarification: str | None = None
    pending_intent: str | None = None
    turn_count: int = 0

    @property
    def awaiting_clarification(self) -> bool:
        return bool(self.pending_clarification)

    def history_block(self, max_turns: int = 8) -> str:
        """Render recent history for a prompt. Oldest first."""
        recent = self.messages[-(max_turns * 2) :]
        lines = [
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content.strip()}"
            for m in recent
        ]
        if self.topic_digest:
            lines.insert(0, f"(Earlier in this conversation: {self.topic_digest})")
        return "\n".join(lines)

    def recent_user_messages(self, limit: int = 3) -> list[str]:
        return [m.content for m in self.messages if m.role == "user"][-limit:]


class SessionStore(ABC):
    backend: str = "base"

    @abstractmethod
    def ensure_schema(self) -> None: ...

    @abstractmethod
    def get(self, session_id: str, user_id: str | None = None) -> SessionState: ...

    @abstractmethod
    def append_message(self, session_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    def set_pending_clarification(
        self, session_id: str, question: str, original_message: str
    ) -> None: ...

    @abstractmethod
    def clear_pending_clarification(self, session_id: str) -> None: ...

    @abstractmethod
    def purge_expired(self) -> int: ...


def build_topic_digest(existing: str, evicted: list[Message], limit: int = 240) -> str:
    """Compress messages falling out of the window into a topic list.

    Deliberately not an LLM summarisation call: it would add a third model
    round-trip to every long conversation, and for "what was this about" the
    user's own earlier questions are a better signal than a paraphrase of them.
    """
    topics = [m.content.strip().rstrip("?").strip() for m in evicted if m.role == "user"]
    if not topics:
        return existing
    combined = "; ".join(filter(None, [existing, *topics]))
    if len(combined) <= limit:
        return combined
    # Keep the most recent topics; drop from the front.
    parts = combined.split("; ")
    while parts and len("; ".join(parts)) > limit:
        parts.pop(0)
    return "; ".join(parts)
