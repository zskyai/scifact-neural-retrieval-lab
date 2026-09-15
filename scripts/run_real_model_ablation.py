"""Run real ColBERT-v2 and SPLADE-v2 reranking on SciFact Dev.

Candidates come from BM25 only; gold evidence is used exclusively by the
standard retrieval metrics.  This avoids the proxy implementations used in the
initial roadmap and records exact model paths and timings.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import string
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, BertModel, BertConfig
from safetensors import safe_open
from retrieval_lab.data import load_claims,load_corpus
from retrieval_lab.bm25 import BM25Config,BM25Index
from retrieval_lab.metrics import SearchResult,evaluate_run

def batches(values,size):
 for i in range(0,len(values),size): yield values[i:i+size]

def colbert_model(path):
    """Load the BERT encoder plus ColBERT-v2's 768 -> 128 projection."""
    try:
        encoder = AutoModel.from_pretrained(path, local_files_only=True)
    except Exception:
        cfg = BertConfig.from_pretrained(path, local_files_only=True)
        encoder = BertModel.from_pretrained(path, config=cfg, local_files_only=True)
    projection = torch.nn.Linear(encoder.config.hidden_size, 128, bias=False)
    projection_path = path / "model.safetensors"
    if projection_path.is_file():
        with safe_open(str(projection_path), framework="pt", device="cpu") as handle:
            if "linear.weight" in handle.keys():
                projection.weight.data.copy_(handle.get_tensor("linear.weight"))
            else:
                raise RuntimeError(f"ColBERT projection not found in {projection_path}")
    else:
        raise FileNotFoundError(f"Safe ColBERT checkpoint is required: {projection_path}")
    return encoder, projection

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--bm25-depth',type=int,default=50);p.add_argument('--colbert-model',type=Path,required=True);p.add_argument('--splade-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--query-limit',type=int,default=0);a=p.parse_args()
 docs=load_corpus(a.data_root/'corpus.jsonl'); qs=load_claims(a.data_root/'claims_dev.jsonl',True); qs=qs[:a.query_limit] if a.query_limit else qs; by={d.doc_id:d for d in docs}; bm=BM25Index(docs,BM25Config()); candidates={q.query_id:bm.search(q.text,a.bm25_depth) for q in qs}; report={'models':{},'queries':len(qs),'candidate_depth':a.bm25_depth}
 tok=AutoTokenizer.from_pretrained(a.colbert_model,local_files_only=True); model, projection=colbert_model(a.colbert_model); model.eval(); projection.eval();
 punctuation_ids={tok.convert_tokens_to_ids(x) for x in string.punctuation}
 punctuation_ids.discard(tok.unk_token_id)
 def enc(texts,kind):
  out=[]
  for chunk in batches(texts,16):
   marker='[unused0]' if kind=='query' else '[unused1]'
   maxlen=32 if kind=='query' else 180
   x=tok([marker+' '+text for text in chunk],padding='max_length',truncation=True,max_length=maxlen,return_tensors='pt')
   if kind=='query':
    pad=x['input_ids'].eq(tok.pad_token_id)
    x['input_ids']=x['input_ids'].masked_fill(pad,tok.mask_token_id)
    x['attention_mask']=torch.ones_like(x['attention_mask'])
   with torch.no_grad():
    h=model(**x).last_hidden_state
    h=projection(h)
    h=torch.nn.functional.normalize(h,p=2,dim=-1)
    mask=x['attention_mask'].bool()
    if kind=='document':
     for punctuation_id in punctuation_ids: mask &= x['input_ids'].ne(punctuation_id)
    out.append((h,mask))
  return out
 started=time.perf_counter(); col_run={}
 unique_doc_ids=sorted({x.doc_id for values in candidates.values() for x in values})
 col_cache={}
 for chunk_ids in batches(unique_doc_ids,16):
  hs,masks=enc([by[doc_id].text for doc_id in chunk_ids],'document')[0]
  for i,doc_id in enumerate(chunk_ids):
   col_cache[doc_id]=(hs[i].cpu(),masks[i].cpu())
 for q in qs:
  cand=[x.doc_id for x in candidates[q.query_id]]; qh, _=enc([q.text],'query')[0]; qh=qh[0]
  dh=torch.stack([col_cache[doc_id][0] for doc_id in cand]); dm=torch.stack([col_cache[doc_id][1] for doc_id in cand])
  token_sim=torch.einsum('qh,ndh->nqd',qh,dh)
  token_sim=token_sim.masked_fill(~dm[:,None,:].bool(),-1e4)
  vals=token_sim.max(-1).values.sum(-1).cpu().tolist()
  ranked=sorted(zip(candidates[q.query_id],vals),key=lambda x:(x[1],x[0].doc_id),reverse=True); col_run[q.query_id]=[SearchResult(x.doc_id,float(s)) for x,s in ranked]
 report['models']['colbert_v2']={**evaluate_run(qs,col_run,ks=[1,5,10,20,100]),'latency_seconds':time.perf_counter()-started,'latency_ms_per_query':(time.perf_counter()-started)*1000/max(len(qs),1),'model_path':str(a.colbert_model),'projection_dim':128,'query_max_length':32,'document_max_length':180,'query_marker':'[unused0]','document_marker':'[unused1]','query_mask_augmentation':True,'document_punctuation_mask':True,'scoring':'sum_query_tokens_of_max_document_token_similarity'}
 stok=AutoTokenizer.from_pretrained(a.splade_model,local_files_only=True); smodel=AutoModelForMaskedLM.from_pretrained(a.splade_model,local_files_only=True).eval(); sparse_cache={}
 def sparse(texts):
  out=[]
  for chunk in batches(texts,8):
   x=stok(chunk,padding=True,truncation=True,max_length=256,return_tensors='pt')
   with torch.no_grad(): logits=smodel(**x).logits; logits=torch.log1p(torch.relu(logits)); logits=logits.masked_fill(~x['attention_mask'].unsqueeze(-1).bool(),0); out.append(logits.max(1).values)
  return torch.cat(out)
 started=time.perf_counter(); spl_run={}
 unique_doc_ids=sorted({x.doc_id for values in candidates.values() for x in values})
 sparse_cache={}
 for chunk_ids in batches(unique_doc_ids,8):
  vectors=sparse([by[doc_id].text for doc_id in chunk_ids]).half()
  for i,doc_id in enumerate(chunk_ids):
   sparse_cache[doc_id]=vectors[i].cpu()
 for q in qs:
  cand=[x.doc_id for x in candidates[q.query_id]]; qv=sparse([q.text]).half()[0]
  dv=torch.stack([sparse_cache[doc_id] for doc_id in cand]); vals=(dv*qv).sum(-1).float().tolist()
  ranked=sorted(zip(candidates[q.query_id],vals),key=lambda x:(x[1],x[0].doc_id),reverse=True); spl_run[q.query_id]=[SearchResult(x.doc_id,float(s)) for x,s in ranked]
 splade_seconds=time.perf_counter()-started
 report['models']['splade_v2']={**evaluate_run(qs,spl_run,ks=[1,5,10,20,100]),'latency_seconds':splade_seconds,'latency_ms_per_query':splade_seconds*1000/max(len(qs),1),'model_path':str(a.splade_model),'pooling':'max(log1p(relu(mlm_logits)))','vector_dimension':int(qv.numel())}
 report['models']['bm25_candidates']={**evaluate_run(qs,candidates,ks=[1,5,10,20,100])}
 report['protocol']={'status':'real_models','seed':42,'split':'SciFact official dev','gold_evidence_used_for_input':False,'test_used':False,'candidate_source':'BM25 train-independent Dev retrieval','unique_candidate_documents':len(unique_doc_ids)}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
