from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .bm25 import BM25Config, BM25Index
from .data import load_claims, load_corpus, write_json, write_jsonl
from .metrics import evaluate_run, load_run, save_run
from .negatives import mine_bm25_hard_negatives


DEFAULT_KS = [1, 5, 10, 20, 100]


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config must contain a JSON object")
    return payload


def _pick(args, config: dict[str, Any], name: str, default: Any = None, *, required: bool = False):
    cli_value = getattr(args, name, None)
    value = cli_value if cli_value is not None else config.get(name, default)
    if required and value is None:
        raise ValueError(f"Missing required option: --{name.replace('_', '-')}")
    return value


def _paths(args, config: dict[str, Any], *, default_claims: str) -> tuple[Path, Path]:
    data_root = Path(_pick(args, config, "data_root", required=True))
    corpus_file = _pick(args, config, "corpus_file", "corpus.jsonl")
    claims_file = _pick(args, config, "claims_file", default_claims)
    return data_root / corpus_file, data_root / claims_file


def _ks(args, config: dict[str, Any]) -> list[int]:
    value = _pick(args, config, "ks", DEFAULT_KS)
    if isinstance(value, str):
        return [int(item) for item in value.split(",")]
    return [int(item) for item in value]


def run_bm25(args) -> int:
    config = _load_config(args.config)
    corpus_path, claims_path = _paths(args, config, default_claims="claims_dev.jsonl")
    documents = load_corpus(corpus_path)
    queries = load_claims(claims_path, require_labels=True)
    bm25_config = BM25Config(
        k1=float(_pick(args, config, "k1", 1.5)),
        b=float(_pick(args, config, "b", 0.75)),
        title_boost=int(_pick(args, config, "title_boost", 2)),
    )
    top_k = int(_pick(args, config, "top_k", 100))
    start = time.perf_counter()
    index = BM25Index(documents, bm25_config)
    index_seconds = time.perf_counter() - start
    search_start = time.perf_counter()
    run = index.batch_search(queries, top_k=top_k)
    search_seconds = time.perf_counter() - search_start
    metrics = evaluate_run(queries, run, ks=_ks(args, config))
    result = {
        "experiment": "bm25_scifact",
        "status": "measured",
        "corpus_path": str(corpus_path),
        "claims_path": str(claims_path),
        "corpus_documents": len(documents),
        "queries": len(queries),
        "bm25": {
            "k1": bm25_config.k1,
            "b": bm25_config.b,
            "title_boost": bm25_config.title_boost,
            "tokenizer": "lowercase_alphanumeric_regex",
        },
        "timing_seconds": {
            "index": index_seconds,
            "search": search_seconds,
            "search_per_query": search_seconds / len(queries),
        },
        "metrics": metrics,
    }
    output_metrics = Path(_pick(args, config, "output_metrics", "results/bm25_scifact_dev.json"))
    output_run = _pick(args, config, "output_run", None)
    write_json(output_metrics, result)
    if output_run:
        save_run(output_run, run)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def run_mine_negatives(args) -> int:
    config = _load_config(args.config)
    corpus_path, claims_path = _paths(args, config, default_claims="claims_train.jsonl")
    documents = load_corpus(corpus_path)
    queries = load_claims(claims_path, require_labels=True)
    bm25_config = BM25Config(
        k1=float(_pick(args, config, "k1", 1.5)),
        b=float(_pick(args, config, "b", 0.75)),
        title_boost=int(_pick(args, config, "title_boost", 2)),
    )
    teacher_model = _pick(args, config, "teacher_model", None)
    teacher = None
    if teacher_model:
        from .rerank import CrossEncoderReranker

        teacher = CrossEncoderReranker(
            teacher_model,
            batch_size=int(_pick(args, config, "teacher_batch_size", 16)),
            max_length=int(_pick(args, config, "teacher_max_length", 512)),
            device=str(_pick(args, config, "device", "auto")),
        )
    rows, stats = mine_bm25_hard_negatives(
        BM25Index(documents, bm25_config),
        queries,
        documents,
        top_pool=int(_pick(args, config, "top_pool", 100)),
        num_negatives=int(_pick(args, config, "num_negatives", 8)),
        rank_start=int(_pick(args, config, "rank_start", 1)),
        teacher=teacher,
        teacher_margin=float(_pick(args, config, "teacher_margin", 0.1)),
    )
    output = Path(_pick(args, config, "output", required=True))
    write_jsonl(output, rows)
    stats_path = output.with_suffix(output.suffix + ".stats.json")
    write_json(stats_path, stats)
    print(json.dumps({"output": str(output), "stats": stats}, indent=2, ensure_ascii=False))
    return 0


def run_train(args) -> int:
    config_data = _load_config(args.config)
    corpus_path, _ = _paths(args, config_data, default_claims="claims_train.jsonl")
    documents = load_corpus(corpus_path)
    from .training import TrainingConfig, train_dual_encoder

    training_config = TrainingConfig(
        model_name_or_path=str(_pick(args, config_data, "model_name_or_path", required=True)),
        passage_model_name_or_path=_pick(args, config_data, "passage_model_name_or_path", None),
        output_dir=str(_pick(args, config_data, "output_dir", required=True)),
        pooling=str(_pick(args, config_data, "pooling", "mean")),
        normalize=bool(_pick(args, config_data, "normalize", True)),
        batch_size=int(_pick(args, config_data, "batch_size", 8)),
        num_negatives=int(_pick(args, config_data, "num_negatives", 4)),
        max_length=int(_pick(args, config_data, "max_length", 256)),
        epochs=int(_pick(args, config_data, "epochs", 2)),
        max_steps=_pick(args, config_data, "max_steps", None),
        learning_rate=float(_pick(args, config_data, "learning_rate", 2e-5)),
        weight_decay=float(_pick(args, config_data, "weight_decay", 0.01)),
        warmup_ratio=float(_pick(args, config_data, "warmup_ratio", 0.1)),
        temperature=float(_pick(args, config_data, "temperature", 0.05)),
        in_batch_negatives=bool(_pick(args, config_data, "in_batch_negatives", True)),
        gradient_clip_norm=float(_pick(args, config_data, "gradient_clip_norm", 1.0)),
        logging_steps=int(_pick(args, config_data, "logging_steps", 10)),
        seed=int(_pick(args, config_data, "seed", 42)),
        device=str(_pick(args, config_data, "device", "auto")),
    )
    summary = train_dual_encoder(
        documents,
        _pick(args, config_data, "mined_file", required=True),
        training_config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def run_dense_eval(args) -> int:
    config = _load_config(args.config)
    corpus_path, claims_path = _paths(args, config, default_claims="claims_dev.jsonl")
    documents = load_corpus(corpus_path)
    queries = load_claims(claims_path, require_labels=True)
    from transformers import AutoTokenizer

    from .neural import DenseEncodingConfig, DualEncoder, dense_retrieve

    checkpoint = _pick(args, config, "checkpoint", None)
    if checkpoint:
        model = DualEncoder.from_checkpoint(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint) / "tokenizer")
    else:
        model_name = str(_pick(args, config, "model_name_or_path", required=True))
        passage_model = _pick(args, config, "passage_model_name_or_path", None)
        model = DualEncoder.from_pretrained(
            model_name,
            passage_model,
            pooling=str(_pick(args, config, "pooling", "mean")),
            normalize=bool(_pick(args, config, "normalize", True)),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    run = dense_retrieve(
        model,
        tokenizer,
        queries,
        documents,
        DenseEncodingConfig(
            batch_size=int(_pick(args, config, "batch_size", 32)),
            max_length=int(_pick(args, config, "max_length", 256)),
            top_k=int(_pick(args, config, "top_k", 100)),
            device=str(_pick(args, config, "device", "auto")),
        ),
    )
    metrics = evaluate_run(queries, run, ks=_ks(args, config))
    output_metrics = Path(_pick(args, config, "output_metrics", "results/dense_scifact_dev.json"))
    output_run = _pick(args, config, "output_run", None)
    result = {
        "experiment": "dense_scifact",
        "status": "measured",
        "checkpoint": checkpoint,
        "model_name_or_path": None if checkpoint else _pick(args, config, "model_name_or_path", None),
        "metrics": metrics,
    }
    write_json(output_metrics, result)
    if output_run:
        save_run(output_run, run)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def run_rerank(args) -> int:
    config = _load_config(args.config)
    corpus_path, claims_path = _paths(args, config, default_claims="claims_dev.jsonl")
    documents = load_corpus(corpus_path)
    queries = load_claims(claims_path, require_labels=True)
    from .rerank import CrossEncoderReranker, rerank_run

    input_run = load_run(_pick(args, config, "input_run", required=True))
    scorer = CrossEncoderReranker(
        str(_pick(args, config, "model_name_or_path", required=True)),
        batch_size=int(_pick(args, config, "batch_size", 16)),
        max_length=int(_pick(args, config, "max_length", 512)),
        device=str(_pick(args, config, "device", "auto")),
    )
    run = rerank_run(
        input_run,
        queries,
        documents,
        scorer,
        rerank_depth=int(_pick(args, config, "rerank_depth", 50)),
    )
    metrics = evaluate_run(queries, run, ks=_ks(args, config))
    result = {
        "experiment": "cross_encoder_rerank_scifact",
        "status": "measured",
        "model_name_or_path": _pick(args, config, "model_name_or_path", None),
        "metrics": metrics,
    }
    output_metrics = Path(_pick(args, config, "output_metrics", "results/rerank_scifact_dev.json"))
    output_run = _pick(args, config, "output_run", None)
    write_json(output_metrics, result)
    if output_run:
        save_run(output_run, run)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _add_common_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--corpus-file")
    parser.add_argument("--claims-file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scifact-retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bm25 = subparsers.add_parser("bm25", help="Run and evaluate the BM25 baseline")
    _add_common_data_options(bm25)
    bm25.add_argument("--k1", type=float)
    bm25.add_argument("--b", type=float)
    bm25.add_argument("--title-boost", type=int)
    bm25.add_argument("--top-k", type=int)
    bm25.add_argument("--ks")
    bm25.add_argument("--output-metrics")
    bm25.add_argument("--output-run")
    bm25.set_defaults(func=run_bm25)

    mine = subparsers.add_parser("mine-negatives", help="Mine BM25 hard negatives")
    _add_common_data_options(mine)
    mine.add_argument("--k1", type=float)
    mine.add_argument("--b", type=float)
    mine.add_argument("--title-boost", type=int)
    mine.add_argument("--top-pool", type=int)
    mine.add_argument("--rank-start", type=int)
    mine.add_argument("--num-negatives", type=int)
    mine.add_argument("--teacher-model")
    mine.add_argument("--teacher-batch-size", type=int)
    mine.add_argument("--teacher-max-length", type=int)
    mine.add_argument("--teacher-margin", type=float)
    mine.add_argument("--device")
    mine.add_argument("--output")
    mine.set_defaults(func=run_mine_negatives)

    train = subparsers.add_parser("train-dual", help="Train the Transformer dual encoder")
    _add_common_data_options(train)
    train.add_argument("--mined-file")
    train.add_argument("--model-name-or-path")
    train.add_argument("--passage-model-name-or-path")
    train.add_argument("--output-dir")
    train.add_argument("--pooling")
    train.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--num-negatives", type=int)
    train.add_argument("--max-length", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--warmup-ratio", type=float)
    train.add_argument("--temperature", type=float)
    train.add_argument("--in-batch-negatives", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument("--gradient-clip-norm", type=float)
    train.add_argument("--logging-steps", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.set_defaults(func=run_train)

    dense = subparsers.add_parser("dense-eval", help="Evaluate a zero-shot or trained dual encoder")
    _add_common_data_options(dense)
    dense.add_argument("--checkpoint")
    dense.add_argument("--model-name-or-path")
    dense.add_argument("--passage-model-name-or-path")
    dense.add_argument("--pooling")
    dense.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    dense.add_argument("--batch-size", type=int)
    dense.add_argument("--max-length", type=int)
    dense.add_argument("--top-k", type=int)
    dense.add_argument("--device")
    dense.add_argument("--ks")
    dense.add_argument("--output-metrics")
    dense.add_argument("--output-run")
    dense.set_defaults(func=run_dense_eval)

    rerank = subparsers.add_parser("rerank", help="Cross-encoder rerank an existing run")
    _add_common_data_options(rerank)
    rerank.add_argument("--input-run")
    rerank.add_argument("--model-name-or-path")
    rerank.add_argument("--rerank-depth", type=int)
    rerank.add_argument("--batch-size", type=int)
    rerank.add_argument("--max-length", type=int)
    rerank.add_argument("--device")
    rerank.add_argument("--ks")
    rerank.add_argument("--output-metrics")
    rerank.add_argument("--output-run")
    rerank.set_defaults(func=run_rerank)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
