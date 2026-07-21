"""Zero-dependency hashing embedder.

A deterministic bag-of-words + character-trigram hashing vectoriser with
sublinear term weighting. It is not semantically clever - it will not know
that "godown" means "warehouse" unless the glossary says so - but it is
exact on term overlap, needs no model download, no network and no AWS
credentials, and makes the whole pipeline runnable and testable offline.

The glossary in `descriptions.py` carries the semantic load; combined with
the lexical scorer in the retriever this is genuinely competitive on a
domain-specific, 87-table corpus. Switch EMBEDDING_PROVIDER to `bedrock`
(Titan v2) for production, or `fastembed` for a local dense model.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from app.embeddings.base import EmbeddingProvider

_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that carry no discriminative signal in this corpus.
_STOPWORDS = frozenset(
    """a an the of to in for on and or is are was were be been being with by from
    at as it its this that these those we our us you your i me my what which who
    whom how when where why all any some each both few more most other such no nor
    not only own same so than too very can will just do does did doing have has had
    please show me tell give list get find want need about into over under""".split()
)


def _tokens(text: str) -> list[str]:
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]
    grams: list[str] = []
    for word in words:
        grams.append(word)
        if len(word) > 4:
            # Character trigrams give partial credit for morphological variants
            # ("warehouse" / "warehousing", "produce" / "production").
            grams.extend(word[i : i + 3] for i in range(len(word) - 2))
    return grams


class HashingEmbeddings(EmbeddingProvider):
    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.model_id = f"hashing-{dim}"

    def _vectorise(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for token in _tokens(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            # Stable across processes, unlike Python's salted hash().
            digest = int.from_bytes(
                hashlib.blake2b(token.encode(), digest_size=8).digest(), "big"
            )
            idx = digest % self.dim
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorise(t) for t in texts]
