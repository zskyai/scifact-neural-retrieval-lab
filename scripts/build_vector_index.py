"""Build a persistent Qdrant local-mode index from the official SciFact corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval_lab.serving import create_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--qdrant-path", type=Path, default=Path("results/qdrant_local"))
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = create_index(
        data_root=args.data_root,
        qdrant_path=args.qdrant_path,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        qdrant_url=args.qdrant_url,
    )
    manifest = Path(args.qdrant_path).parent / "qdrant_index_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
