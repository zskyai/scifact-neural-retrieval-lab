"""Compare Qdrant local and Milvus using the same hard-negative encoder."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from retrieval_lab.data import load_claims, load_corpus
from retrieval_lab.neural import DualEncoder
from transformers import AutoTokenizer

def pct(values,p): return float(np.percentile(values,p))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--milvus-uri',default='http://127.0.0.1:19530'); p.add_argument('--milvus-collection',default='scifact_hn_milvus'); p.add_argument('--qdrant-path',type=Path,default=Path('results/qdrant_hn_local')); p.add_argument('--output',type=Path,required=True); p.add_argument('--top-k',type=int,default=10); a=p.parse_args()
    docs=load_corpus(a.data_root/'corpus.jsonl'); queries=load_claims(a.data_root/'claims_dev.jsonl',require_labels=True); model=DualEncoder.from_checkpoint(a.checkpoint); tok=AutoTokenizer.from_pretrained(a.checkpoint/'tokenizer')
    import torch
    def enc(texts):
        out=[]
        for i in range(0,len(texts),64):
            batch=tok(texts[i:i+64],padding=True,truncation=True,max_length=256,return_tensors='pt')
            with torch.no_grad(): out.append(model.encode(batch,is_query=True if texts is qtexts else False).cpu().numpy())
        return np.concatenate(out)
    dtexts=[d.text for d in docs]; dvec=[]
    for i in range(0,len(dtexts),64):
        batch=tok(dtexts[i:i+64],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.no_grad(): dvec.append(model.encode(batch,is_query=False).cpu().numpy())
    dvec=np.concatenate(dvec); qtexts=[q.text for q in queries]; qvec=[]
    for i in range(0,len(qtexts),64):
        batch=tok(qtexts[i:i+64],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.no_grad(): qvec.append(model.encode(batch,is_query=True).cpu().numpy())
    qvec=np.concatenate(qvec)
    from qdrant_client import QdrantClient, models
    qc=QdrantClient(path=str(a.qdrant_path)); qname='scifact_hn';
    if qc.collection_exists(qname): qc.delete_collection(qname)
    qc.create_collection(qname, vectors_config=models.VectorParams(size=int(dvec.shape[1]),distance=models.Distance.COSINE))
    qc.upsert(qname,[models.PointStruct(id=i,vector=v.tolist(),payload={'doc_id':d.doc_id}) for i,(d,v) in enumerate(zip(docs,dvec))],wait=True)
    from pymilvus import MilvusClient
    mc=MilvusClient(uri=a.milvus_uri); mc.load_collection(collection_name=a.milvus_collection); byid={str(d.doc_id):d for d in docs}
    report={'encoder':str(a.checkpoint),'queries':len(queries),'top_k':a.top_k,'stores':{}}
    for name in ('qdrant','milvus'):
        lat=[]; hits=[]
        for q,v in zip(queries,qvec):
            t=time.perf_counter()
            if name=='qdrant': raw=qc.query_points(collection_name=qname,query=v.tolist(),limit=a.top_k,with_payload=True).points; ids=[str(x.payload['doc_id']) for x in raw]
            else: raw=mc.search(collection_name=a.milvus_collection,data=[v.tolist()],limit=a.top_k,output_fields=['doc_id']); ids=[str(x['entity']['doc_id']) for x in raw[0]]
            lat.append((time.perf_counter()-t)*1000); hits.append(len(set(ids)&set(q.relevant_doc_ids))/len(q.relevant_doc_ids))
        size=0
        root=a.qdrant_path
        if name=='qdrant' and root.exists(): size=sum(x.stat().st_size for x in root.rglob('*') if x.is_file())
        if name=='milvus': size_note='Docker volume; inspect with docker system df / volume size'
        report['stores'][name]={'recall_at_k':float(np.mean(hits)),'p50_ms':pct(lat,50),'p95_ms':pct(lat,95),'p99_ms':pct(lat,99),'index_size_bytes':size if name=='qdrant' else None,'index_size_note':size_note if name=='milvus' else None}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
