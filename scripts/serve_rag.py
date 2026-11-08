"""Start the local FastAPI RAG service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_lab.serving import LocalRAG, build_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--qdrant-path", default="results/qdrant_local")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default="scifact_evidence")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--sentence-nli-model", default=None, help="Optional local sentence-level NLI checkpoint")
    parser.add_argument('--domain-verifier', default=None)
    parser.add_argument('--verifier-calibration', default=None)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--case-db", default="results/research_cases.sqlite3")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    rag = LocalRAG(
        data_root=args.data_root,
        qdrant_path=args.qdrant_path,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
        device=args.device,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        sentence_nli_model=args.sentence_nli_model,
        domain_verifier=args.domain_verifier,
        verifier_calibration=args.verifier_calibration,
    )
    uvicorn.run(build_app(rag, case_db_path=args.case_db), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
