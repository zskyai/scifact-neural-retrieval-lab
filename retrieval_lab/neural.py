from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import Document, QueryExample
from .metrics import SearchResult


def mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * expanded_mask).sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class DualEncoder(nn.Module):
    """A shared- or untied-weight Transformer dual encoder."""

    def __init__(
        self,
        query_encoder: nn.Module,
        passage_encoder: nn.Module | None = None,
        *,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder or query_encoder
        self.shared_weights = passage_encoder is None
        self.pooling = pooling
        self.normalize = normalize

    @classmethod
    def from_pretrained(
        cls,
        query_model_name_or_path: str,
        passage_model_name_or_path: str | None = None,
        *,
        pooling: str = "mean",
        normalize: bool = True,
        trust_remote_code: bool = False,
    ) -> "DualEncoder":
        from transformers import AutoModel

        query_encoder = AutoModel.from_pretrained(
            query_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        passage_encoder = None
        if passage_model_name_or_path:
            passage_encoder = AutoModel.from_pretrained(
                passage_model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
        return cls(
            query_encoder=query_encoder,
            passage_encoder=passage_encoder,
            pooling=pooling,
            normalize=normalize,
        )

    def _pool(self, outputs, attention_mask: Tensor) -> Tensor:
        hidden = outputs.last_hidden_state
        embeddings = hidden[:, 0] if self.pooling == "cls" else mean_pool(hidden, attention_mask)
        return F.normalize(embeddings, p=2, dim=-1) if self.normalize else embeddings

    def encode(self, inputs: Mapping[str, Tensor], *, is_query: bool) -> Tensor:
        encoder = self.query_encoder if is_query else self.passage_encoder
        outputs = encoder(**inputs)
        return self._pool(outputs, inputs["attention_mask"])

    def forward(
        self,
        query_inputs: Mapping[str, Tensor],
        positive_inputs: Mapping[str, Tensor],
        negative_inputs: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_embeddings = self.encode(query_inputs, is_query=True)
        positive_embeddings = self.encode(positive_inputs, is_query=False)
        negative_embeddings = self.encode(negative_inputs, is_query=False)
        return query_embeddings, positive_embeddings, negative_embeddings

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        query_dir = output / "query_encoder"
        self.query_encoder.save_pretrained(query_dir)
        passage_dir: str | None = None
        if not self.shared_weights:
            passage_path = output / "passage_encoder"
            self.passage_encoder.save_pretrained(passage_path)
            passage_dir = passage_path.name
        metadata = {
            "query_encoder": query_dir.name,
            "passage_encoder": passage_dir,
            "shared_weights": self.shared_weights,
            "pooling": self.pooling,
            "normalize": self.normalize,
        }
        with (output / "dual_encoder_config.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> "DualEncoder":
        checkpoint = Path(checkpoint_dir)
        with (checkpoint / "dual_encoder_config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        passage_path = config.get("passage_encoder")
        return cls.from_pretrained(
            str(checkpoint / config["query_encoder"]),
            str(checkpoint / passage_path) if passage_path else None,
            pooling=config["pooling"],
            normalize=bool(config["normalize"]),
        )


def info_nce_loss(
    query_embeddings: Tensor,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    *,
    temperature: float = 0.05,
    in_batch_negatives: bool = True,
    negative_mask: Tensor | None = None,
) -> Tensor:
    """InfoNCE over one positive and N hard negatives for every query.

    Other examples' positives are optionally added as in-batch negatives. A
    false-negative mask can remove known unsafe negatives before softmax.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if query_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
        raise ValueError("query and positive embeddings must have shape [batch, dim]")
    if negative_embeddings.ndim == 2:
        negative_embeddings = negative_embeddings.unsqueeze(1)
    if negative_embeddings.ndim != 3:
        raise ValueError("negative embeddings must have shape [batch, negatives, dim]")
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError("query and positive embeddings must have identical shapes")
    if negative_embeddings.shape[0] != query_embeddings.shape[0]:
        raise ValueError("negative batch size must match query batch size")
    if negative_embeddings.shape[2] != query_embeddings.shape[1]:
        raise ValueError("embedding dimensions do not match")

    queries = F.normalize(query_embeddings, p=2, dim=-1)
    positives = F.normalize(positive_embeddings, p=2, dim=-1)
    negatives = F.normalize(negative_embeddings, p=2, dim=-1)

    positive_logits = (queries * positives).sum(dim=-1, keepdim=True)
    hard_negative_logits = torch.einsum("bd,bnd->bn", queries, negatives)
    if negative_mask is not None:
        if negative_mask.shape != hard_negative_logits.shape:
            raise ValueError("negative_mask must have shape [batch, negatives]")
        hard_negative_logits = hard_negative_logits.masked_fill(
            ~negative_mask.to(dtype=torch.bool, device=hard_negative_logits.device),
            torch.finfo(hard_negative_logits.dtype).min,
        )

    pieces = [positive_logits, hard_negative_logits]
    batch_size = queries.shape[0]
    if in_batch_negatives and batch_size > 1:
        cross_scores = queries @ positives.transpose(0, 1)
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=queries.device)
        pieces.append(cross_scores[off_diagonal].view(batch_size, batch_size - 1))

    logits = torch.cat(pieces, dim=1) / temperature
    targets = torch.zeros(batch_size, dtype=torch.long, device=queries.device)
    return F.cross_entropy(logits, targets)


@dataclass(frozen=True)
class DenseEncodingConfig:
    batch_size: int = 32
    max_length: int = 256
    top_k: int = 100
    device: str = "auto"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.inference_mode()
def _encode_texts(
    model: DualEncoder,
    tokenizer,
    texts: Sequence[str],
    *,
    is_query: bool,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Tensor:
    batches: list[Tensor] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        batches.append(model.encode(inputs, is_query=is_query).cpu())
    return torch.cat(batches, dim=0)


def dense_retrieve(
    model: DualEncoder,
    tokenizer,
    queries: Sequence[QueryExample],
    documents: Sequence[Document],
    config: DenseEncodingConfig | None = None,
) -> dict[str, list[SearchResult]]:
    config = config or DenseEncodingConfig()
    if config.batch_size <= 0 or config.max_length <= 0 or config.top_k <= 0:
        raise ValueError("batch_size, max_length, and top_k must be positive")
    device = resolve_device(config.device)
    model.to(device)

    passage_embeddings = _encode_texts(
        model,
        tokenizer,
        [document.text for document in documents],
        is_query=False,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=device,
    )
    query_embeddings = _encode_texts(
        model,
        tokenizer,
        [query.text for query in queries],
        is_query=True,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=device,
    )
    similarity = query_embeddings @ passage_embeddings.transpose(0, 1)
    result_count = min(config.top_k, len(documents))
    scores, indices = torch.topk(similarity, k=result_count, dim=1)

    run: dict[str, list[SearchResult]] = {}
    for row, query in enumerate(queries):
        run[query.query_id] = [
            SearchResult(doc_id=documents[int(doc_index)].doc_id, score=float(score))
            for score, doc_index in zip(scores[row], indices[row])
        ]
    return run

