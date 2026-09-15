"""Source-aware retrieval router for SciFact plus optional Wikipedia evidence.

Wikipedia is an auxiliary background channel, never a replacement for the
SciFact gold corpus.  The router preserves source and confidence metadata so
downstream answer generation can reject unsupported claims.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re

@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    source: str
    title: str = ""

def load_wiki_jsonl(path: str | Path) -> list[KnowledgeChunk]:
    chunks=[]
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line); chunks.append(KnowledgeChunk(str(row.get('id',len(chunks))), str(row.get('text','')), 'wikipedia', str(row.get('title',''))))
    return chunks

def lexical_router(query: str, chunks: list[KnowledgeChunk], top_k: int = 5) -> list[KnowledgeChunk]:
    q=set(re.findall(r'[a-z0-9]+', query.lower())); ranked=[]
    for c in chunks:
        t=set(re.findall(r'[a-z0-9]+', c.text.lower())); ranked.append((len(q&t)/max(len(q),1), c))
    return [c for _,c in sorted(ranked,key=lambda x:(x[0],x[1].chunk_id),reverse=True)[:top_k]]
