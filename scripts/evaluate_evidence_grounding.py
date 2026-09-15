"""Evaluate citation retrieval and evidence-snippet grounding on SciFact dev.

This evaluates an extractive evidence path. It does not evaluate generative
answer correctness or factuality.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from retrieval_lab.data import corpus_by_id, load_corpus
from retrieval_lab.evidence import select_evidence_sentences
from retrieval_lab.metrics import load_run


def load_claim_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def gold_evidence(evidence: dict) -> dict[str, set[int]]:
    """Flatten SciFact's document -> evidence-set annotation structure."""

    result: dict[str, set[int]] = defaultdict(set)
    for doc_id, evidence_sets in evidence.items():
        for evidence_set in evidence_sets:
            result[str(doc_id)].update(int(index) for index in evidence_set.get("sentences", []))
    return dict(result)


def evaluate_evidence_grounding(
    claim_rows: Iterable[dict],
    run,
    documents_by_id,
    *,
    retrieval_k: int = 10,
    snippets_per_document: int = 3,
) -> dict[str, int | float]:
    if retrieval_k <= 0 or snippets_per_document <= 0:
        raise ValueError("retrieval_k and snippets_per_document must be positive")

    all_claims = list(claim_rows)
    evidence_claims = [row for row in all_claims if row.get("evidence")]
    citation_hits = 0
    evidence_document_hits = 0
    grounded_claim_hits = 0
    selected_gold_sentences = 0
    total_gold_sentences = 0

    for claim in all_claims:
        query_id = str(claim["id"])
        retrieved_doc_ids = {item.doc_id for item in run.get(query_id, ())[:retrieval_k]}
        citation_ids = {str(doc_id) for doc_id in claim.get("cited_doc_ids", ())}
        citation_hits += bool(retrieved_doc_ids & citation_ids)

    for claim in evidence_claims:
        query_id = str(claim["id"])
        evidence_by_doc = gold_evidence(claim["evidence"])
        total_gold_sentences += sum(len(sentence_ids) for sentence_ids in evidence_by_doc.values())
        retrieved = run.get(query_id, ())[:retrieval_k]
        retrieved_ids = {item.doc_id for item in retrieved}
        evidence_document_hits += bool(retrieved_ids & set(evidence_by_doc))

        claim_hit = False
        for result in retrieved:
            gold_ids = evidence_by_doc.get(result.doc_id)
            document = documents_by_id.get(result.doc_id)
            if not gold_ids or document is None:
                continue
            selected_ids = {
                snippet.sentence_id
                for snippet in select_evidence_sentences(
                    str(claim["claim"]),
                    document,
                    top_k=snippets_per_document,
                )
            }
            overlaps = selected_ids & gold_ids
            selected_gold_sentences += len(overlaps)
            claim_hit = claim_hit or bool(overlaps)
        grounded_claim_hits += claim_hit

    claims_count = len(all_claims)
    evidence_claim_count = len(evidence_claims)
    return {
        "scope": "SciFact dev extractive evidence grounding",
        "claims": claims_count,
        "evidence_annotated_claims": evidence_claim_count,
        "retrieval_k": retrieval_k,
        "snippets_per_document": snippets_per_document,
        "citation_document_hit_rate": citation_hits / claims_count if claims_count else 0.0,
        "evidence_document_hit_rate": evidence_document_hits / evidence_claim_count if evidence_claim_count else 0.0,
        "grounded_claim_hit_rate": grounded_claim_hits / evidence_claim_count if evidence_claim_count else 0.0,
        "gold_evidence_sentence_coverage": selected_gold_sentences / total_gold_sentences if total_gold_sentences else 0.0,
        "gold_evidence_sentences": total_gold_sentences,
        "selected_gold_evidence_sentences": selected_gold_sentences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--snippets-per-document", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate_evidence_grounding(
        load_claim_rows(args.data_root / "claims_dev.jsonl"),
        load_run(args.run),
        corpus_by_id(load_corpus(args.data_root / "corpus.jsonl")),
        retrieval_k=args.retrieval_k,
        snippets_per_document=args.snippets_per_document,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
