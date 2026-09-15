"""Local RAG serving components for the SciFact benchmark.

The implementation is intentionally explicit about its scope: Qdrant runs in
local persistent mode and the default answerer is extractive. This makes the
retrieval and service measurements reproducible without claiming production
traffic or generative quality that has not been measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .bm25 import BM25Config, BM25Index
from .data import Document, load_corpus
from .evidence import select_evidence_sentences


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2048)
    top_k: int = Field(default=10, ge=1, le=100)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2048)
    documents: list[str] = Field(min_length=1, max_length=50)
    top_k: int = Field(default=5, ge=1, le=50)


class RagRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2048)
    top_k: int = Field(default=5, ge=1, le=20)


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    score: float
    title: str
    text: str
    source: str = "scifact"


class MeanPoolEmbedder:
    def __init__(self, model_name: str, *, device: str = "cpu", max_length: int = 256) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=False)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors, axis=0)


class CrossEncoderScorer:
    def __init__(self, model_name: str, *, device: str = "cpu", max_length: int = 512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=False)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.inference_mode()
    def score(self, query: str, documents: Sequence[str], *, batch_size: int = 16) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            batch = list(documents[start : start + batch_size])
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits.reshape(-1)
            scores.extend(float(value) for value in logits.cpu())
        return scores


class LocalRAG:
    def __init__(
        self,
        *,
        data_root: str | Path,
        qdrant_path: str | Path,
        embedding_model: str,
        rerank_model: str,
        device: str = "cpu",
        collection: str = "scifact_evidence",
        qdrant_url: str | None = None,
    ) -> None:
        self.documents = load_corpus(Path(data_root) / "corpus.jsonl")
        self.by_id = {document.doc_id: document for document in self.documents}
        self.bm25 = BM25Index(self.documents, BM25Config(k1=1.5, b=0.75, title_boost=2))
        self.embedder = MeanPoolEmbedder(embedding_model, device=device)
        self.reranker = CrossEncoderScorer(rerank_model, device=device)
        self.client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=str(qdrant_path))
        self.collection = collection
        info = self.client.get_collection(collection)
        self.vector_size = int(info.config.params.vectors.size)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.embedder.encode(texts)

    def retrieve(self, query: str, *, top_k: int = 10) -> tuple[list[RetrievedDocument], dict[str, float]]:
        started = time.perf_counter()
        query_vector = self.embed([query])[0].tolist()
        embedding_ms = (time.perf_counter() - started) * 1000.0

        vector_started = time.perf_counter()
        vector_hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=min(50, max(top_k, 10)),
            with_payload=True,
        ).points
        vector_ms = (time.perf_counter() - vector_started) * 1000.0

        bm_started = time.perf_counter()
        bm_hits = self.bm25.search(query, top_k=min(50, max(top_k, 10)))
        bm25_ms = (time.perf_counter() - bm_started) * 1000.0

        fused: dict[str, float] = {}
        for rank, hit in enumerate(vector_hits, start=1):
            payload = hit.payload or {}
            doc_id = str(payload.get("doc_id", hit.id))
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank)
        for rank, hit in enumerate(bm_hits, start=1):
            fused[hit.doc_id] = fused.get(hit.doc_id, 0.0) + 1.0 / (60 + rank)
        ranked = sorted(fused.items(), key=lambda item: (item[1], item[0]), reverse=True)[:top_k]
        results = [
            RetrievedDocument(
                doc_id=doc_id,
                score=float(score),
                title=self.by_id[doc_id].title,
                text=self.by_id[doc_id].text,
            )
            for doc_id, score in ranked
            if doc_id in self.by_id
        ]
        return results, {
            "embedding_ms": embedding_ms,
            "vector_db_ms": vector_ms,
            "bm25_ms": bm25_ms,
            "retrieval_ms": (time.perf_counter() - started) * 1000.0,
        }

    def rerank(self, query: str, documents: Sequence[RetrievedDocument], *, top_k: int) -> tuple[list[RetrievedDocument], float]:
        started = time.perf_counter()
        scores = self.reranker.score(query, [item.text for item in documents])
        reranked = [item for item, score in zip(documents, scores)]
        reranked = [RetrievedDocument(item.doc_id, float(score), item.title, item.text, item.source) for item, score in zip(documents, scores)]
        reranked.sort(key=lambda item: (item.score, item.doc_id), reverse=True)
        return reranked[:top_k], (time.perf_counter() - started) * 1000.0

    def rag(self, query: str, *, top_k: int) -> dict[str, Any]:
        candidates, timings = self.retrieve(query, top_k=max(top_k, 10))
        reranked, rerank_ms = self.rerank(query, candidates, top_k=top_k)
        evidence = [
            {
                "evidence_id": f"E{index}",
                "doc_id": item.doc_id,
                "title": item.title,
                "snippets": [
                    {"sentence_id": snippet.sentence_id, "text": snippet.text, "score": snippet.score}
                    for snippet in select_evidence_sentences(query, self.by_id[item.doc_id], top_k=3)
                ],
            }
            for index, item in enumerate(reranked, start=1)
        ]
        answer = "\n".join(
            f"[{item['evidence_id']}] " + " ".join(snippet["text"] for snippet in item["snippets"])
            for item in evidence
        )
        return {
            "answer": answer,
            "citations": [item["evidence_id"] for item in evidence],
            "answer_mode": "extractive_evidence_only",
            "evidence": evidence,
            "timings_ms": {**timings, "rerank_ms": rerank_ms, "total_ms": timings["retrieval_ms"] + rerank_ms},
        }


def build_app(rag: LocalRAG) -> FastAPI:
    app = FastAPI(title="SciFact Evidence RAG", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "corpus_documents": len(rag.documents), "collection": rag.collection, "vector_size": rag.vector_size}

    @app.post("/v1/embed")
    def embed(request: EmbedRequest) -> dict[str, Any]:
        started = time.perf_counter()
        vectors = rag.embed(request.texts)
        return {"model": rag.embedder.model.name_or_path, "dimension": int(vectors.shape[1]), "vectors": vectors.tolist(), "latency_ms": (time.perf_counter() - started) * 1000.0}

    @app.post("/v1/retrieve")
    def retrieve(request: RetrieveRequest) -> dict[str, Any]:
        results, timings = rag.retrieve(request.query, top_k=request.top_k)
        return {"results": [item.__dict__ for item in results], "timings_ms": timings}

    @app.post("/v1/rerank")
    def rerank(request: RerankRequest) -> dict[str, Any]:
        items = [RetrievedDocument(str(index), 0.0, "", text) for index, text in enumerate(request.documents)]
        results, latency = rag.rerank(request.query, items, top_k=request.top_k)
        return {"results": [item.__dict__ for item in results], "latency_ms": latency}

    @app.post("/v1/rag")
    def rag_endpoint(request: RagRequest) -> dict[str, Any]:
        return rag.rag(request.query, top_k=request.top_k)

    return app


def create_index(
    *,
    data_root: str | Path,
    qdrant_path: str | Path,
    embedding_model: str,
    device: str = "cpu",
    collection: str = "scifact_evidence",
    batch_size: int = 32,
    qdrant_url: str | None = None,
) -> dict[str, Any]:
    documents = load_corpus(Path(data_root) / "corpus.jsonl")
    embedder = MeanPoolEmbedder(embedding_model, device=device)
    vectors = embedder.encode([document.text for document in documents], batch_size=batch_size)
    client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=str(qdrant_path))
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection_name=collection, vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE))
    points = [
        models.PointStruct(
            id=index,
            vector=vector.tolist(),
            payload={"doc_id": document.doc_id, "title": document.title, "text": document.text, "source": "scifact", "acl": ["public"]},
        )
        for index, (document, vector) in enumerate(zip(documents, vectors))
    ]
    for start in range(0, len(points), 256):
        client.upsert(collection_name=collection, points=points[start : start + 256], wait=True)
    return {"documents": len(documents), "dimension": int(vectors.shape[1]), "collection": collection, "qdrant_path": str(qdrant_path)}
