"""Fine-tune a Cross-Encoder on SciFact train claim/document pairs."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from retrieval_lab.data import load_claims, load_corpus
from retrieval_lab.bm25 import BM25Config, BM25Index

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--max-train',type=int,default=0); a=p.parse_args()
    docs=load_corpus(a.data_root/'corpus.jsonl'); byid={str(d.doc_id):d for d in docs}; claims=load_claims(a.data_root/'claims_train.jsonl',require_labels=True); index=BM25Index(docs,BM25Config())
    pairs=[]
    for q in claims:
        pos=list(q.relevant_doc_ids)
        if not pos: continue
        neg=[x.doc_id for x in index.search(q.text,top_k=20) if x.doc_id not in q.relevant_doc_ids][:3]
        pairs.extend([(q.text,byid[x].text,1) for x in pos if x in byid]); pairs.extend((q.text,byid[x].text,0) for x in neg if x in byid)
    random.Random(42).shuffle(pairs); pairs=pairs[:a.max_train] if a.max_train else pairs
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(a.model,local_files_only=True,num_labels=2,ignore_mismatched_sizes=True); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
    losses=[]
    for epoch in range(a.epochs):
        for start in range(0,len(pairs),8):
            batch=pairs[start:start+8]; enc=tok([x[0] for x in batch],[x[1] for x in batch],padding=True,truncation=True,max_length=256,return_tensors='pt'); labels=torch.tensor([x[2] for x in batch]); out=model(**enc,labels=labels); out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True); losses.append(float(out.loss))
    a.output.mkdir(parents=True,exist_ok=True); model.save_pretrained(a.output); tok.save_pretrained(a.output); (a.output/'training_summary.json').write_text(json.dumps({'pairs':len(pairs),'positive':sum(x[2] for x in pairs),'negative':sum(not x[2] for x in pairs),'epochs':a.epochs,'final_loss':losses[-1] if losses else None},indent=2),encoding='utf8'); print({'pairs':len(pairs),'final_loss':losses[-1] if losses else None})
if __name__=='__main__': main()
