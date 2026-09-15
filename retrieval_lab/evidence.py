"""Deterministic evidence-snippet selection for extractive SciFact RAG."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .bm25 import tokenize
from .data import Document


@dataclass(frozen=True)
class EvidenceSnippet:
    """One abstract sentence returned as an inspectable evidence candidate."""

    sentence_id: int
    text: str
    score: float


def sentence_score(query: str, sentence: str) -> float:
    """Rank a sentence by query-term coverage, with a mild length penalty.

    The selector is intentionally lexical and deterministic. It narrows an
    already retrieved abstract to readable snippets; it does not claim to be
    a learned entailment model.
    """

    query_terms = Counter(tokenize(query))
    sentence_terms = Counter(tokenize(sentence))
    if not query_terms or not sentence_terms:
        return 0.0

    shared_unique = sum(term in sentence_terms for term in query_terms)
    shared_tokens = sum(min(count, sentence_terms[term]) for term, count in query_terms.items())
    coverage = shared_unique / len(query_terms)
    density = shared_tokens / math.sqrt(sum(sentence_terms.values()))
    return coverage + 0.1 * density


def select_evidence_sentences(
    query: str,
    document: Document,
    *,
    top_k: int = 3,
) -> list[EvidenceSnippet]:
    """Return up to ``top_k`` abstract sentences, ranked then re-ordered by ID."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = [
        EvidenceSnippet(sentence_id=index, text=sentence, score=sentence_score(query, sentence))
        for index, sentence in enumerate(document.abstract)
        if sentence.strip()
    ]
    ranked.sort(key=lambda item: (-item.score, item.sentence_id))
    return sorted(ranked[:top_k], key=lambda item: item.sentence_id)
