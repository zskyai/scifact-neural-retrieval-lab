"""Build a Milvus index from the official SciFact corpus."""
from __future__ import annotations
import argparse
from pathlib import Path
from retrieval_lab.data import load_corpus
from retrieval_lab.serving import MeanPoolEmbedder
from retrieval_lab.neural import DualEncoder
from transformers import AutoTokenizer
from retrieval_lab.vector_store import MilvusStore

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--embedding-model'); p.add_argument('--checkpoint',type=Path); p.add_argument('--uri',default='http://127.0.0.1:19530'); p.add_argument('--collection',default='scifact_evidence'); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args()
    docs=load_corpus(a.data_root/'corpus.jsonl')
    if a.checkpoint:
        model=DualEncoder.from_checkpoint(a.checkpoint); tokenizer=AutoTokenizer.from_pretrained(a.checkpoint/'tokenizer')
        import torch
        vectors=[]
        for i in range(0,len(docs),a.batch_size):
            batch=tokenizer([d.text for d in docs[i:i+a.batch_size]],padding=True,truncation=True,max_length=256,return_tensors='pt')
            with torch.no_grad(): vectors.append(model.encode(batch,is_query=False).cpu().numpy())
        import numpy as np
        vectors=np.concatenate(vectors)
    else:
        if not a.embedding_model: raise SystemExit('--embedding-model or --checkpoint is required')
        embed=MeanPoolEmbedder(a.embedding_model); vectors=embed.encode([d.text for d in docs],batch_size=a.batch_size)
    store=MilvusStore(a.uri,a.collection,int(vectors.shape[1]))
    rows=[{'id':d.doc_id,'vector':v.tolist(),'doc_id':d.doc_id,'title':d.title,'text':d.text,'source':'scifact'} for d,v in zip(docs,vectors)]; store.upsert(rows); print({'collection':a.collection,'documents':len(rows),'dimension':int(vectors.shape[1]),'uri':a.uri})
if __name__=='__main__': main()
