"""Start the local FastAPI RAG service."""

from __future__ import annotations

import argparse

import uvicorn

from retrieval_lab.serving import LocalRAG, build_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--qdrant-path", default="results/qdrant_local")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    rag = LocalRAG(
        data_root=args.data_root,
        qdrant_path=args.qdrant_path,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
        device=args.device,
        qdrant_url=args.qdrant_url,
    )
    uvicorn.run(build_app(rag), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
