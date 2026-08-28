from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .data import Document, QueryExample
from .metrics import SearchResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    title_boost: int = 2

    def validate(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be in [0, 1]")
        if self.title_boost < 1:
            raise ValueError("title_boost must be at least 1")


class BM25Index:
    def __init__(
        self,
        documents: Sequence[Document],
        config: BM25Config | None = None,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if not documents:
            raise ValueError("Cannot build BM25 over an empty corpus")
        self.documents = list(documents)
        self.config = config or BM25Config()
        self.config.validate()
        self.tokenizer = tokenizer
        self.doc_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_index, document in enumerate(self.documents):
            tokens = tokenizer(document.text_with_title_boost(self.config.title_boost))
            self.doc_lengths.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                postings[term].append((doc_index, frequency))

        self.postings = dict(postings)
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        corpus_size = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (corpus_size - len(entries) + 0.5) / (len(entries) + 0.5))
            for term, entries in self.postings.items()
        }

    def search(self, query: str, top_k: int = 100) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = Counter(self.tokenizer(query))
        scores: dict[int, float] = defaultdict(float)
        k1 = self.config.k1
        b = self.config.b

        for term, query_frequency in query_terms.items():
            entries = self.postings.get(term)
            if not entries:
                continue
            idf = self.idf[term]
            for doc_index, term_frequency in entries:
                doc_length = self.doc_lengths[doc_index]
                norm = k1 * (1.0 - b + b * doc_length / self.avg_doc_length)
                term_score = idf * (term_frequency * (k1 + 1.0)) / (term_frequency + norm)
                scores[doc_index] += query_frequency * term_score

        best = heapq.nlargest(
            min(top_k, len(scores)),
            scores.items(),
            key=lambda item: (item[1], self.documents[item[0]].doc_id),
        )
        return [
            SearchResult(doc_id=self.documents[doc_index].doc_id, score=float(score))
            for doc_index, score in best
        ]

    def batch_search(
        self,
        queries: Iterable[QueryExample],
        top_k: int = 100,
    ) -> dict[str, list[SearchResult]]:
        return {query.query_id: self.search(query.text, top_k=top_k) for query in queries}
