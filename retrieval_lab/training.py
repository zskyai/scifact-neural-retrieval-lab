from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .data import Document, corpus_by_id, write_json
from .neural import DualEncoder, info_nce_loss, resolve_device


class MinedTripletDataset(Dataset):
    def __init__(
        self,
        mined_path: str | Path,
        documents: Sequence[Document],
        *,
        num_negatives: int,
    ) -> None:
        if num_negatives <= 0:
            raise ValueError("num_negatives must be positive")
        document_lookup = corpus_by_id(documents)
        examples: list[dict] = []
        with Path(mined_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                positive_ids = [
                    str(doc_id)
                    for doc_id in row.get("positive_document_ids", [])
                    if str(doc_id) in document_lookup
                ]
                negative_ids = [
                    str(doc_id)
                    for doc_id in row.get("negative_document_ids", [])
                    if str(doc_id) in document_lookup
                ]
                if not positive_ids or not negative_ids:
                    continue
                repeated_negatives = (negative_ids * math.ceil(num_negatives / len(negative_ids)))[:num_negatives]
                examples.append(
                    {
                        "query_id": str(row.get("query_id", line_number)),
                        "query": str(row["query"]),
                        "positive": document_lookup[positive_ids[0]].text,
                        "negatives": [document_lookup[doc_id].text for doc_id in repeated_negatives],
                    }
                )
        if not examples:
            raise ValueError(f"No usable triplets found in {mined_path}")
        self.examples = examples
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class TripletCollator:
    def __init__(self, tokenizer, *, max_length: int, num_negatives: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives

    def _tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, examples: Sequence[dict]) -> dict:
        queries = [example["query"] for example in examples]
        positives = [example["positive"] for example in examples]
        negatives = [text for example in examples for text in example["negatives"]]
        return {
            "query": self._tokenize(queries),
            "positive": self._tokenize(positives),
            "negative": self._tokenize(negatives),
            "batch_size": len(examples),
            "num_negatives": self.num_negatives,
        }


@dataclass(frozen=True)
class TrainingConfig:
    model_name_or_path: str
    output_dir: str
    passage_model_name_or_path: str | None = None
    pooling: str = "mean"
    normalize: bool = True
    batch_size: int = 8
    num_negatives: int = 4
    max_length: int = 256
    epochs: int = 2
    max_steps: int | None = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    temperature: float = 0.05
    in_batch_negatives: bool = True
    gradient_clip_norm: float = 1.0
    logging_steps: int = 10
    seed: int = 42
    device: str = "auto"
    trust_remote_code: bool = False


def _move_batch(batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _learning_rate_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(step, 1) / warmup_steps
    remaining = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_dual_encoder(
    documents: Sequence[Document],
    mined_path: str | Path,
    config: TrainingConfig,
) -> dict:
    if config.batch_size <= 0 or config.epochs <= 0 or config.logging_steps <= 0:
        raise ValueError("batch_size, epochs, and logging_steps must be positive")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    model = DualEncoder.from_pretrained(
        config.model_name_or_path,
        config.passage_model_name_or_path,
        pooling=config.pooling,
        normalize=config.normalize,
        trust_remote_code=config.trust_remote_code,
    )
    dataset = MinedTripletDataset(
        mined_path,
        documents,
        num_negatives=config.num_negatives,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TripletCollator(
            tokenizer,
            max_length=config.max_length,
            num_negatives=config.num_negatives,
        ),
    )
    planned_steps = len(loader) * config.epochs
    total_steps = min(planned_steps, config.max_steps) if config.max_steps else planned_steps
    if total_steps <= 0:
        raise ValueError("Training has zero optimization steps")

    device = resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_multiplier(step, total_steps, warmup_steps),
    )

    logs: list[dict[str, float | int]] = []
    global_step = 0
    rolling_loss = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        for batch in loader:
            query_inputs = _move_batch(batch["query"], device)
            positive_inputs = _move_batch(batch["positive"], device)
            negative_inputs = _move_batch(batch["negative"], device)
            query_embeddings, positive_embeddings, flat_negative_embeddings = model(
                query_inputs,
                positive_inputs,
                negative_inputs,
            )
            negative_embeddings = flat_negative_embeddings.view(
                batch["batch_size"],
                batch["num_negatives"],
                -1,
            )
            loss = info_nce_loss(
                query_embeddings,
                positive_embeddings,
                negative_embeddings,
                temperature=config.temperature,
                in_batch_negatives=config.in_batch_negatives,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            rolling_loss += float(loss.detach().cpu())
            if global_step % config.logging_steps == 0 or global_step == total_steps:
                interval = config.logging_steps if global_step % config.logging_steps == 0 else global_step % config.logging_steps
                logs.append(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": rolling_loss / max(interval, 1),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                rolling_loss = 0.0
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    output = Path(config.output_dir)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output / "tokenizer")
    summary = {
        "status": "completed",
        "training_examples": len(dataset),
        "optimization_steps": global_step,
        "config": asdict(config),
        "logs": logs,
        "note": "This file records training loss only. Retrieval quality must be measured separately.",
    }
    write_json(output / "training_summary.json", summary)
    return summary

