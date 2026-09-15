"""Reproducible lightweight ablation for late interaction, sparse and CRAG retry."""
from __future__ import annotations
import argparse,json,math,re
from pathlib import Path
from retrieval_lab.data import load_claims,load_corpus
from retrieval_lab.metrics import evaluate_run,SearchResult
from retrieval_lab.bm25 import BM25Index,BM25Config
def toks(s):return re.findall(r'[a-z0-9]+',s.lower())
def score(q,d,mode):
 q=toks(q);dt=toks(d)
 if mode=='splade': return sum((1+math.log(dt.count(x))) if x in dt else 0 for x in set(q))
 if mode=='colbert': return sum(max((1.0 if x==y else .5 if x[:5]==y[:5] else 0 for y in dt),default=0) for x in q)
 return 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate-k',type=int,default=100);a=p.parse_args();docs=load_corpus(a.data_root/'corpus.jsonl');qs=load_claims(a.data_root/'claims_dev.jsonl',True);bm=BM25Index(docs,BM25Config());by={d.doc_id:d for d in docs};report={}
 for mode in ('splade','colbert'):
  run={}
  for q in qs:
   cand=bm.search(q.text,top_k=a.candidate_k);ranked=sorted(((x.doc_id,score(q.text,by[x.doc_id].text,mode)) for x in cand),key=lambda x:(x[1],x[0]),reverse=True);run[q.query_id]=[SearchResult(x,s) for x,s in ranked]
  report[mode+'_proxy']=evaluate_run(qs,run,ks=[1,5,10,20,100])
 # CRAG/Self-RAG proxy retries with expanded lexical query only on low-margin cases.
 run={};retried=0
 for q in qs:
  first=bm.search(q.text,top_k=100);margin=(first[0].score-first[4].score) if len(first)>4 else 0
  if margin<1.0:
   retried+=1; expanded=q.text+' scientific evidence study result'; first=bm.search(expanded,top_k=100)
  run[q.query_id]=first
 report['crag_retry_proxy']={**evaluate_run(qs,run,ks=[1,5,10,20,100]),'retried_queries':retried,'note':'deterministic proxy; replace scorers with trained ColBERT/SPLADE checkpoints for claimed model results'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
