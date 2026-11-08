"""Local RAG serving components for the SciFact benchmark.

The implementation is intentionally explicit about its scope: Qdrant runs in
local persistent mode and the default answerer is extractive. This makes the
retrieval and service measurements reproducible without claiming production
traffic or generative quality that has not been measured.
"""

from __future__ import annotations

import time
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, Response
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .bm25 import BM25Config, BM25Index, tokenize
from .data import Document, load_corpus
from .evidence import select_evidence_sentences
from .adaptive_routing import AdaptiveEvidenceRouter, RetrievalSignals
from .query_planning import ScientificQueryPlanner
from .evidence_agent import EvidenceRetrievalAgent
from .sentence_nli import SentenceNLI
from .case_store import ResearchCaseStore


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


class PlanRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2048)


class RouteRequest(BaseModel):
    top1_score: float
    top1_margin: float
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    support_score: float = Field(ge=0.0, le=1.0)
    contradiction_score: float = Field(ge=0.0, le=1.0)
    retry_budget: int = Field(default=1, ge=0, le=3)


class AgentVerifyRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=2048)
    top_k: int = Field(default=10, ge=1, le=50)
    max_retries: int = Field(default=1, ge=0, le=3)


class CaseRequest(AgentVerifyRequest):
    pass


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=128)
    verdict: Literal["SUPPORTS", "CONTRADICTS", "NEI"]
    note: str = Field(default="", max_length=1000)


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    score: float
    title: str
    text: str
    source: str = "scifact"


class MeanPoolEmbedder:
    def __init__(self, model_name: str, *, device: str = "cpu", max_length: int = 256) -> None:
        checkpoint = Path(model_name)
        metadata = checkpoint / "dual_encoder_config.json"
        self.pooling = "mean"
        if metadata.is_file():
            config = json.loads(metadata.read_text(encoding="utf-8"))
            if not config.get("shared_weights", True):
                raise ValueError("serving currently requires a shared-weight dual encoder")
            self.pooling = config.get("pooling", "mean")
            if not config.get("normalize", True):
                raise ValueError("cosine vector serving requires normalized checkpoint embeddings")
            tokenizer_path = str(checkpoint / "tokenizer")
            model_path = str(checkpoint / config["query_encoder"])
        else:
            tokenizer_path = model_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=checkpoint.exists())
        self.model = AutoModel.from_pretrained(model_path, local_files_only=checkpoint.exists())
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
            if self.pooling == "cls":
                pooled = hidden[:, 0]
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
            logits = self.model(**inputs).logits
            if logits.ndim != 2 or logits.shape[1] not in (1, 2):
                raise ValueError("reranker requires one relevance logit or a binary relevance classifier")
            logits = logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]
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
        sentence_nli_model: str | None = None,
        domain_verifier: str | None = None,
        verifier_calibration: str | None = None,
    ) -> None:
        self.documents = load_corpus(Path(data_root) / "corpus.jsonl")
        self.by_id = {document.doc_id: document for document in self.documents}
        self.bm25 = BM25Index(self.documents, BM25Config(k1=1.5, b=0.75, title_boost=2))
        self.embedder = MeanPoolEmbedder(embedding_model, device=device)
        self.reranker = CrossEncoderScorer(rerank_model, device=device)
        self.query_planner = ScientificQueryPlanner()
        self.router = AdaptiveEvidenceRouter()
        self.sentence_nli = SentenceNLI(sentence_nli_model)
        self.client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=str(qdrant_path))
        self.collection = collection
        info = self.client.get_collection(collection)
        self.vector_size = int(info.config.params.vectors.size)
        self.domain_review = None
        if bool(domain_verifier) != bool(verifier_calibration):
            raise ValueError('Domain verifier and calibration must be provided together')
        if domain_verifier:
            from .domain_verifier import DomainEvidenceVerifier, DomainReviewEngine
            self.domain_review = DomainReviewEngine(
                self.documents, DomainEvidenceVerifier(domain_verifier, device=device, calibration=verifier_calibration))

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
        if len(scores) != len(documents):
            raise ValueError("reranker score count does not match candidate documents")
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

    def verify_claim(self, claim: str, *, top_k: int = 10, max_retries: int = 1) -> dict[str, Any]:
        if self.domain_review is not None:
            return self.domain_review.verify_claim(claim, top_k=top_k, max_retries=max_retries)
        started = time.perf_counter()
        sentence_decisions: dict[tuple[str, int], dict[str, Any]] = {}
        tool_timings: list[dict[str, Any]] = []
        ignored = set('a an and are as at be by does for from has have in is it of on or show shows that the this to was were with'.split())
        def content_terms(text):
            return set(re.findall(r'[a-z0-9]+', text.lower())) - ignored
        def retrieve(query: str, limit: int) -> list[dict[str, Any]]:
            documents, timings = self.retrieve(query, top_k=min(limit, 50))
            documents, rerank_ms = self.rerank(query, documents, top_k=min(limit, 50))
            tool_timings.append({"query": query, **timings, "rerank_ms": rerank_ms})
            return [item.__dict__ for item in documents]

        def verify(original_claim: str, documents: Sequence[dict[str, Any]]) -> dict[str, float]:
            sentence_rows = [
                (str(document["doc_id"]), index, sentence)
                for document in documents
                for index, sentence in (
                    (snippet.sentence_id, snippet.text)
                    for snippet in select_evidence_sentences(
                        original_claim,
                        self.by_id[str(document["doc_id"])],
                        top_k=3,
                    )
                )
            ]
            if not sentence_rows:
                return {"coverage": 0.0, "support": 0.0, "contradiction": 0.0}
            # Topic relevance precedes NLI: unrelated sentences cannot refute a
            # claim merely because an out-of-domain NLI model is confident.
            query_terms = content_terms(original_claim)
            relevant_rows = [row for row in sentence_rows if len(query_terms & content_terms(row[2])) / max(len(query_terms), 1) >= .25]
            for row in sentence_rows:
                sentence_decisions[(row[0], row[1])] = {"label": "not_checked", "score": 0.0, "reason": "low_topic_overlap"}
            if not relevant_rows:
                return {"coverage": 0.0, "support": 0.0, "contradiction": 0.0}
            nli_started = time.perf_counter()
            predictions = self.sentence_nli.predict(original_claim, [row[2] for row in relevant_rows])
            tool_timings.append({"nli_ms": (time.perf_counter() - nli_started) * 1000.0, "sentences": len(relevant_rows), "topic_filtered": len(sentence_rows) - len(relevant_rows)})
            supports, contradictions = [], []
            for row, prediction in zip(relevant_rows, predictions):
                sentence_decisions[(row[0], row[1])] = dict(prediction)
                label = str(prediction.get("label", "")).lower()
                score = float(prediction.get("score", 0.0))
                if "entail" in label or label in {"supports", "support"}:
                    supports.append(score)
                elif "contrad" in label or label in {"refutes", "contradicts"}:
                    contradictions.append(score)
            coverage = max(len(query_terms & content_terms(row[2])) / max(len(query_terms), 1) for row in relevant_rows)
            return {
                "coverage": coverage,
                "support": max(supports, default=0.0),
                "contradiction": max(contradictions, default=0.0),
            }

        result = EvidenceRetrievalAgent(
            retrieve,
            verify,
            planner=self.query_planner,
            router=self.router,
            max_retries=max_retries,
            top_k=top_k,
        ).run(claim)
        evidence = []
        for document in result.evidence:
            doc = self.by_id[str(document["doc_id"])]
            snippets = select_evidence_sentences(claim, doc, top_k=3)
            evidence.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "sentences": [{"sentence_id": item.sentence_id, "text": item.text, "score": item.score,
                               "nli": sentence_decisions.get((doc.doc_id, item.sentence_id))} for item in snippets],
            })
        return {
            "action": result.action,
            "label": result.label,
            "confidence": result.confidence,
            "evidence": evidence,
            "trace": list(result.trace),
            "nli_model": str(self.sentence_nli.model_path) if self.sentence_nli.model_path else "lexical_fallback",
            "citations": [{"doc_id": doc["doc_id"], "sentence_id": sentence["sentence_id"]}
                          for doc in evidence for sentence in doc["sentences"]
                          if result.label != "NEI" and sentence["nli"]
                          and sentence["nli"]["label"].lower() == ("entailment" if result.label == "SUPPORTS" else "contradiction")
                          and sentence["nli"]["score"] >= self.router.min_support],
            "timings_ms": {"tools": tool_timings, "total_ms": (time.perf_counter() - started) * 1000.0},
            "provenance": {"collection": self.collection, "encoder": self.embedder.model.name_or_path,
                           "reranker": self.reranker.model.name_or_path, "top_k": top_k, "max_retries": max_retries,
                           "threshold_status": "heuristic_not_calibrated", "min_sentence_topic_overlap": .25},
        }


def build_app(rag: LocalRAG, *, case_db_path: str | Path = "results/research_cases.sqlite3") -> FastAPI:
    app = FastAPI(title="SciFact Evidence RAG", version="0.1.0")
    case_store = ResearchCaseStore(case_db_path)

    @app.get("/", include_in_schema=False)
    def workbench() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        domain = getattr(rag, 'domain_review', None)
        return {"status": "ok", "corpus_documents": len(rag.documents), "collection": rag.collection, "vector_size": rag.vector_size,
                "verification_mode": 'domain_bm25' if domain else 'hybrid_generic_nli',
                "verification_config": domain.verifier.calibration if domain else None}

    @app.post("/v1/embed")
    def embed(request: EmbedRequest) -> dict[str, Any]:
        started = time.perf_counter()
        vectors = rag.embed(request.texts)
        return {"model": rag.embedder.model.name_or_path, "dimension": int(vectors.shape[1]), "vectors": vectors.tolist(), "latency_ms": (time.perf_counter() - started) * 1000.0}

    @app.post("/v1/plan")
    def plan(request: PlanRequest) -> dict[str, Any]:
        query_plan = rag.query_planner.plan(request.query)
        return {
            "original": query_plan.original,
            "rewrites": list(query_plan.rewrites),
            "subqueries": list(query_plan.subqueries),
            "all_queries": list(query_plan.all_queries),
            "strategy": query_plan.strategy,
        }

    @app.post("/v1/route")
    def route(request: RouteRequest) -> dict[str, Any]:
        decision = rag.router.decide(
            RetrievalSignals(
                request.top1_score,
                request.top1_margin,
                request.evidence_coverage,
                request.support_score,
                request.contradiction_score,
            ),
            retry_budget=request.retry_budget,
        )
        return {"action": decision.action, "confidence": decision.confidence, "reasons": list(decision.reasons), "retry_budget": decision.retry_budget}

    @app.post("/v1/agent/verify")
    def agent_verify(request: AgentVerifyRequest) -> dict[str, Any]:
        try:
            return rag.verify_claim(request.claim, top_k=request.top_k, max_retries=request.max_retries)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/cases")
    def create_case(request: CaseRequest) -> dict[str, Any]:
        if not request.claim.strip():
            raise HTTPException(status_code=422, detail="claim must not be blank")
        try:
            result = rag.verify_claim(request.claim, top_k=request.top_k, max_retries=request.max_retries)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return case_store.create(request.claim, result)

    @app.get("/v1/cases")
    def list_cases(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        if not 1 <= limit <= 200 or offset < 0:
            raise HTTPException(status_code=422, detail="limit must be 1..200 and offset non-negative")
        return case_store.list(limit, offset)

    @app.get("/v1/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        try:
            return case_store.get(case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc

    @app.post("/v1/cases/{case_id}/reviews")
    def review_case(case_id: str, request: ReviewRequest) -> dict[str, Any]:
        if not request.reviewer.strip():
            raise HTTPException(status_code=422, detail="reviewer must not be blank")
        try:
            return case_store.review(case_id, request.reviewer.strip(), request.verdict, request.note.strip())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc

    @app.get("/v1/documents/{doc_id}")
    def source_document(doc_id: str):
        doc = rag.by_id.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return {"doc_id": doc.doc_id, "title": doc.title, "sentences": [
            {"sentence_id": i, "text": sentence} for i, sentence in enumerate(doc.abstract)]}

    @app.get("/v1/cases/{case_id}/report")
    def report(case_id: str):
        try:
            item = case_store.get(case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc
        payload = {"schema_version": 1, "purpose": "research_review", "training_eligible": False, **item}
        return Response(json.dumps(payload, ensure_ascii=False, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="case-{item["case_id"]}.json"'})

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
