from __future__ import annotations

from typing import Protocol, Sequence

import torch

from .data import Document, QueryExample, corpus_by_id
from .metrics import Run, SearchResult
from .neural import resolve_device


class PairScorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document."""


class CrossEncoderReranker:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        all_scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_documents = list(documents[start : start + self.batch_size])
            inputs = self.tokenizer(
                [query] * len(batch_documents),
                batch_documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits
            if logits.ndim == 1 or logits.shape[-1] == 1:
                batch_scores = logits.reshape(-1)
            else:
                batch_scores = logits[:, -1]
            all_scores.extend(float(value) for value in batch_scores.cpu())
        return all_scores


def rerank_run(
    run: Run,
    queries: Sequence[QueryExample],
    documents: Sequence[Document],
    scorer: PairScorer,
    *,
    rerank_depth: int = 50,
) -> dict[str, list[SearchResult]]:
    if rerank_depth <= 0:
        raise ValueError("rerank_depth must be positive")
    query_lookup = {query.query_id: query for query in queries}
    document_lookup = corpus_by_id(documents)
    reranked: dict[str, list[SearchResult]] = {}

    for query_id, original_results in run.items():
        query = query_lookup.get(query_id)
        if query is None:
            continue
        head = [result for result in original_results[:rerank_depth] if result.doc_id in document_lookup]
        tail = list(original_results[rerank_depth:])
        texts = [document_lookup[result.doc_id].text for result in head]
        new_scores = scorer.score(query.text, texts)
        rescored = [
            SearchResult(doc_id=result.doc_id, score=float(score))
            for result, score in zip(head, new_scores)
        ]
        rescored.sort(key=lambda result: (result.score, result.doc_id), reverse=True)
        reranked[query_id] = rescored + tail
    return reranked
