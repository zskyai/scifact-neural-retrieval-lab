"""Longformer joint stance classification and rationale membership.

Sentence membership is distinct from independent sentence entailment. Windows
preserve complete sentences; label supervision can be masked for windows that
do not contain a complete gold rationale.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

LABELS = ('NEI', 'SUPPORTS', 'CONTRADICTS')


@dataclass
class DocumentWindow:
    input_ids: list[int]
    global_attention_mask: list[int]
    sentence_positions: list[int]
    sentence_ids: list[int]


def encode_windows(tokenizer, claim, sentences, *, max_length=1024, overlap=1):
    if type(max_length) is not int or max_length < 16 or type(overlap) is not int or overlap < 0:
        raise ValueError('Invalid token budget or sentence overlap')
    bos = tokenizer.bos_token_id
    sep = tokenizer.sep_token_id
    if bos is None or sep is None:
        raise ValueError('Expected Longformer BOS and SEP tokens')
    claim_ids = tokenizer(claim, add_special_tokens=False)['input_ids']
    prefix = [bos, *claim_ids, sep, sep]
    tokens = [tokenizer(s, add_special_tokens=False)['input_ids'] for s in sentences]
    if not tokens:
        raise ValueError('Document has no sentences')
    if any(len(prefix) + len(s) + 1 > max_length for s in tokens):
        raise ValueError('A complete sentence exceeds the claim/document token budget')
    windows = []
    start = 0
    while start < len(tokens):
        ids = list(prefix); positions = []; source_ids = []
        end = start
        while end < len(tokens) and len(ids) + len(tokens[end]) + 1 <= max_length:
            ids.extend(tokens[end]); ids.append(sep)
            positions.append(len(ids) - 1); source_ids.append(end); end += 1
        if end == start:
            raise ValueError('Window did not retain a complete sentence')
        global_mask = [1] * len(prefix) + [0] * (len(ids) - len(prefix))
        windows.append(DocumentWindow(ids, global_mask, positions, source_ids))
        if end == len(tokens):
            break
        start = max(start + 1, end - overlap)
    return windows


def joint_loss(label_logits, rationale_logits, labels, rationale_targets,
               sentence_mask, *, label_mask=None, weights=None, rationale_weight=1.0):
    if label_mask is None:
        label_mask = torch.ones_like(labels, dtype=torch.bool)
    if weights is None:
        weights = torch.ones_like(labels, dtype=label_logits.dtype)
    if weights.shape != labels.shape or (weights < 0).any() or not torch.isfinite(weights).all():
        raise ValueError('Expected finite non-negative sample weights')
    active = label_mask.to(weights.dtype) * weights
    ce = F.cross_entropy(label_logits, labels, reduction='none')
    label_loss = (ce * active).sum() / active.sum().clamp_min(1)
    valid = sentence_mask & rationale_targets.ge(0)
    raw = F.binary_cross_entropy_with_logits(rationale_logits, rationale_targets.clamp_min(0).to(rationale_logits.dtype), reduction='none')
    per_row = raw.masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1)
    rationale_active = weights * valid.any(-1).to(weights.dtype)
    rationale_loss = (per_row * rationale_active).sum() / rationale_active.sum().clamp_min(1)
    return {'loss': label_loss + rationale_weight * rationale_loss,
            'label_loss': label_loss, 'rationale_loss': rationale_loss}


class MultiVerSJointModel(nn.Module):
    def __init__(self, checkpoint=None, *, encoder=None):
        super().__init__()
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(checkpoint, local_files_only=True)
        if encoder.config.model_type != 'longformer':
            raise ValueError('Joint SciFact model requires a Longformer encoder')
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        dropout = getattr(encoder.config, 'hidden_dropout_prob', .1)
        self.dropout = nn.Dropout(dropout)
        self.label_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.rationale_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask, global_attention_mask, sentence_positions, sentence_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        hidden = self.dropout(encoded.last_hidden_state)
        document = hidden[:, 0]
        positions = sentence_positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        sentences = hidden.gather(1, positions)
        document_expanded = document.unsqueeze(1).expand_as(sentences)
        logits = self.rationale_head(torch.cat([document_expanded, sentences], dim=-1)).squeeze(-1)
        return {'label_logits': self.label_head(document),
                'rationale_logits': logits.masked_fill(~sentence_mask, 0.0)}
