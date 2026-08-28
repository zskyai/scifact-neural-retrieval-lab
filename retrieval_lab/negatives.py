from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bm25 import BM25Index
from .data import Document, QueryExample, corpus_by_id
from .rerank import PairScorer


@dataclass(frozen=True)
class FilteredNegatives:
    selected_doc_ids: tuple[str, ...]
    removed_gold: int
    removed_by_teacher: int


def filter_false_negatives(
    candidate_doc_ids: Sequence[str],
    positive_doc_ids: Sequence[str],
    *,
    num_negatives: int,
    candidate_teacher_scores: Sequence[float] | None = None,
    positive_teacher_scores: Sequence[float] | None = None,
    teacher_margin: float = 0.1,
) -> FilteredNegatives:
    """Remove labeled positives and teacher-ambiguous candidates.

    A candidate is considered ambiguous when its teacher score is within
    ``teacher_margin`` of the strongest labeled positive. This conservative
    rule reduces false negatives but does not claim that teacher scores are
    ground truth.
    """

    if num_negatives <= 0:
        raise ValueError("num_negatives must be positive")
    if teacher_margin < 0:
        raise ValueError("teacher_margin must be non-negative")
    if candidate_teacher_scores is not None and len(candidate_teacher_scores) != len(candidate_doc_ids):
        raise ValueError("candidate ids and teacher scores must have the same length")
    if (candidate_teacher_scores is None) != (positive_teacher_scores is None):
        raise ValueError("candidate and positive teacher scores must be provided together")

    positive_set = set(positive_doc_ids)
    selected: list[str] = []
    seen: set[str] = set()
    removed_gold = 0
    removed_by_teacher = 0
    positive_threshold = None
    if positive_teacher_scores:
        positive_threshold = max(positive_teacher_scores) - teacher_margin

    for index, doc_id in enumerate(candidate_doc_ids):
        if doc_id in positive_set:
            removed_gold += 1
            continue
        if doc_id in seen:
            continue
        if positive_threshold is not None and candidate_teacher_scores is not None:
            if candidate_teacher_scores[index] >= positive_threshold:
                removed_by_teacher += 1
                continue
        selected.append(doc_id)
        seen.add(doc_id)
        if len(selected) >= num_negatives:
            break

    return FilteredNegatives(
        selected_doc_ids=tuple(selected),
        removed_gold=removed_gold,
        removed_by_teacher=removed_by_teacher,
    )


def mine_bm25_hard_negatives(
    index: BM25Index,
    queries: Sequence[QueryExample],
    documents: Sequence[Document],
    *,
    top_pool: int = 100,
    num_negatives: int = 8,
    rank_start: int = 1,
    teacher: PairScorer | None = None,
    teacher_margin: float = 0.1,
) -> tuple[list[dict], dict[str, int | float]]:
    if top_pool <= 0 or rank_start <= 0:
        raise ValueError("top_pool and rank_start must be positive")
    if rank_start > top_pool:
        raise ValueError("rank_start cannot exceed top_pool")
    document_lookup = corpus_by_id(documents)
    rows: list[dict] = []
    skipped_unlabeled = 0
    skipped_missing_positive = 0
    removed_gold = 0
    removed_by_teacher = 0

    for query in queries:
        positive_ids = [doc_id for doc_id in query.relevant_doc_ids if doc_id in document_lookup]
        if not query.relevant_doc_ids:
            skipped_unlabeled += 1
            continue
        if not positive_ids:
            skipped_missing_positive += 1
            continue

        retrieved = index.search(query.text, top_k=top_pool)
        candidate_results = retrieved[rank_start - 1 :]
        candidate_ids = [result.doc_id for result in candidate_results]
        candidate_teacher_scores = None
        positive_teacher_scores = None
        if teacher is not None:
            candidate_teacher_scores = teacher.score(
                query.text,
                [document_lookup[doc_id].text for doc_id in candidate_ids],
            )
            positive_teacher_scores = teacher.score(
                query.text,
                [document_lookup[doc_id].text for doc_id in positive_ids],
            )

        filtered = filter_false_negatives(
            candidate_ids,
            positive_ids,
            num_negatives=num_negatives,
            candidate_teacher_scores=candidate_teacher_scores,
            positive_teacher_scores=positive_teacher_scores,
            teacher_margin=teacher_margin,
        )
        removed_gold += filtered.removed_gold
        removed_by_teacher += filtered.removed_by_teacher
        if not filtered.selected_doc_ids:
            continue
        rows.append(
            {
                "query_id": query.query_id,
                "query": query.text,
                "positive_document_ids": positive_ids,
                "negative_document_ids": list(filtered.selected_doc_ids),
                "mining": {
                    "method": "bm25",
                    "top_pool": top_pool,
                    "rank_start": rank_start,
                    "teacher_filtered": teacher is not None,
                    "teacher_margin": teacher_margin if teacher is not None else None,
                },
            }
        )

    stats: dict[str, int | float] = {
        "input_queries": len(queries),
        "written_queries": len(rows),
        "skipped_unlabeled": skipped_unlabeled,
        "skipped_missing_positive": skipped_missing_positive,
        "removed_labeled_positives": removed_gold,
        "removed_teacher_ambiguous": removed_by_teacher,
        "average_negatives": (
            sum(len(row["negative_document_ids"]) for row in rows) / len(rows) if rows else 0.0
        ),
    }
    return rows, stats
