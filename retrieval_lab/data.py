from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    abstract: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.title, *self.abstract) if part).strip()

    def text_with_title_boost(self, title_boost: int = 1) -> str:
        if title_boost < 1:
            raise ValueError("title_boost must be at least 1")
        title_parts = [self.title] * title_boost if self.title else []
        return " ".join((*title_parts, *self.abstract)).strip()


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    text: str
    relevant_doc_ids: frozenset[str]


def _read_jsonl(path: str | Path) -> Iterator[dict]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"JSONL file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {resolved}:{line_number}") from exc


def load_corpus(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    for row in _read_jsonl(path):
        if "doc_id" not in row:
            raise ValueError("Corpus row is missing 'doc_id'")
        doc_id = str(row["doc_id"])
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate document id: {doc_id}")
        abstract = row.get("abstract", [])
        if isinstance(abstract, str):
            abstract = [abstract]
        documents.append(
            Document(
                doc_id=doc_id,
                title=str(row.get("title", "")),
                abstract=tuple(str(sentence) for sentence in abstract),
            )
        )
        seen_ids.add(doc_id)
    if not documents:
        raise ValueError(f"Corpus is empty: {path}")
    return documents


def _relevant_ids(row: Mapping) -> frozenset[str]:
    cited = row.get("cited_doc_ids")
    if cited is not None:
        return frozenset(str(doc_id) for doc_id in cited)
    evidence = row.get("evidence", {})
    return frozenset(str(doc_id) for doc_id in evidence)


def load_claims(path: str | Path, require_labels: bool = False) -> list[QueryExample]:
    queries: list[QueryExample] = []
    seen_ids: set[str] = set()
    for row in _read_jsonl(path):
        if "id" not in row or "claim" not in row:
            raise ValueError("Claim row must contain 'id' and 'claim'")
        query_id = str(row["id"])
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query id: {query_id}")
        relevant_doc_ids = _relevant_ids(row)
        if require_labels and not relevant_doc_ids:
            raise ValueError(f"Query {query_id} has no retrieval labels")
        queries.append(
            QueryExample(
                query_id=query_id,
                text=str(row["claim"]),
                relevant_doc_ids=relevant_doc_ids,
            )
        )
        seen_ids.add(query_id)
    if not queries:
        raise ValueError(f"Claims file is empty: {path}")
    return queries


def corpus_by_id(documents: Sequence[Document]) -> dict[str, Document]:
    return {document.doc_id: document for document in documents}


def write_json(path: str | Path, payload: Mapping | Sequence, *, indent: int = 2) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

