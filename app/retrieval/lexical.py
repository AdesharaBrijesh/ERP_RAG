"""Lexical half of hybrid retrieval.

Dense vectors are good at "how is the shop floor doing" -> production_batches.
They are unreliable at exact domain nouns: a user typing "GRN" or "BOM" needs
those three letters matched, not smeared across an embedding. A BM25-style
term scorer plus exact multi-word phrase matching against the curated glossary
covers that, and the two signals are combined in the retriever.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from app.retrieval.descriptions import TABLE_GLOSSARY

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """a an the of to in for on and or is are was were be been being with by from at as
    it its this that these those we our us you your i me my what which who whom how when
    where why all any some each both few more most other such no nor not only own same so
    than too very can will just do does did doing have has had please show me tell give
    list get find want need about into over under many much any""".split()
)


def singularise(word: str) -> str:
    """Crude, deliberate stemming - plurals only.

    Without it "suppliers" does not match the glossary's "supplier" and
    `vendors` is never retrieved for "how many suppliers do we work with?".
    Applied identically to documents and queries, so exactness matters less
    than consistency: `address` -> `address` (guarded by the `ss` check),
    `status` -> `status`, `stocks` -> `stock`, `companies` -> `company`.
    """
    if len(word) <= 3 or word.endswith("ss") or word.endswith("us"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [
        singularise(w)
        for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS and len(w) > 1
    ]


@dataclass
class LexicalIndex:
    """BM25-lite over weighted table keyword documents, plus a phrase index."""

    docs: dict[str, dict[str, float]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    phrases: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, table_keywords: dict[str, dict[str, float]]) -> LexicalIndex:
        index = cls(docs=dict(table_keywords))
        n = max(len(index.docs), 1)
        df: dict[str, int] = {}
        for terms in index.docs.values():
            for term in terms:
                df[term] = df.get(term, 0) + 1
        index.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

        # Multi-word glossary synonyms ("on hand", "bill of materials") are
        # matched as whole phrases - far more precise than their tokens.
        for table_name, (_, synonyms) in TABLE_GLOSSARY.items():
            if table_name not in index.docs:
                continue
            for synonym in synonyms:
                if " " in synonym:
                    index.phrases.setdefault(synonym.lower(), []).append(table_name)
        return index

    def score(self, query: str) -> dict[str, float]:
        """Normalised 0..1 lexical relevance per table."""
        query_lower = query.lower()
        terms = set(tokenize(query))
        if not terms and not self.phrases:
            return {}

        max_possible = sum(self.idf.get(t, 1.0) for t in terms) or 1.0
        scores: dict[str, float] = {}
        for table_name, doc_terms in self.docs.items():
            overlap = terms & doc_terms.keys()
            if not overlap:
                continue
            scores[table_name] = (
                sum(self.idf.get(t, 1.0) * doc_terms[t] for t in overlap) / max_possible
            )

        for phrase, tables in self.phrases.items():
            if phrase in query_lower:
                # A matched multi-word phrase is a near-certain signal.
                for table_name in tables:
                    scores[table_name] = min(1.0, scores.get(table_name, 0.0) + 0.45)

        return scores
