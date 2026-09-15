"""CRAG-style conditional retrieval retry with real CE and sentence NLI models.

This is explicitly a CRAG-style ablation, not a reproduction of the complete
CRAG or Self-RAG training recipes.  SciFact Dev is split into disjoint
calibration/evaluation query subsets; official test is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from transformers import pipeline

from retrieval_lab.data import load_claims, load_corpus
from retrieval_lab.metrics import SearchResult, evaluate_run, load_run, save_run
from retrieval_lab.rerank import CrossEncoderReranker


def qsplit(queries):
    calibration, evaluation = [], []
    for query in queries:
        bucket = int(hashlib.sha1(query.query_id.encode()).hexdigest()[:8], 16) / 2**32
        (calibration if bucket < .3 else evaluation).append(query)
    return calibration, evaluation


def union_candidates(first, second, depth):
    seen = set(); output = []
    for result in list(first[:depth]) + list(second[:depth]):
        if result.doc_id not in seen:
            output.append(result); seen.add(result.doc_id)
    return output


def nli_probabilities(pipe, pairs, batch_size):
    raw = pipe(pairs, batch_size=batch_size, truncation=True, top_k=None)
    scores = []
    for labels in raw:
        mapping = {str(x["label"]).lower(): float(x["score"]) for x in labels}
        scores.append(max((score for label, score in mapping.items() if "entail" in label or "contrad" in label), default=0.0))
    return scores


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--bm25-run',type=Path,required=True); p.add_argument('--dense-run',type=Path,required=True); p.add_argument('--cross-encoder',type=Path,required=True); p.add_argument('--nli-model',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--initial-depth',type=int,default=10); p.add_argument('--retry-depth',type=int,default=20); p.add_argument('--query-limit',type=int,default=0); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--retry-penalty',type=float,default=.02); args=p.parse_args()
    docs=load_corpus(args.data_root/'corpus.jsonl'); by={x.doc_id:x for x in docs}; queries=load_claims(args.data_root/'claims_dev.jsonl',True); queries=queries[:args.query_limit] if args.query_limit else queries
    bm25=load_run(args.bm25_run); dense=load_run(args.dense_run); ce=CrossEncoderReranker(str(args.cross_encoder),batch_size=args.batch_size,max_length=256,device='cpu'); nli=pipeline('text-classification',model=str(args.nli_model),tokenizer=str(args.nli_model),local_files_only=True,device=-1)
    initial={}; retry={}; reflection={}; ce_started=time.perf_counter()
    for query in queries:
        head=bm25[query.query_id][:args.initial_depth]; scores=ce.score(query.text,[by[x.doc_id].text for x in head]); ranked=sorted(zip(head,scores),key=lambda x:(x[1],x[0].doc_id),reverse=True); initial[query.query_id]=[SearchResult(x.doc_id,float(s)) for x,s in ranked]
        expanded=union_candidates(bm25[query.query_id],dense[query.query_id],args.retry_depth); rscores=ce.score(query.text,[by[x.doc_id].text for x in expanded]); rranked=sorted(zip(expanded,rscores),key=lambda x:(x[1],x[0].doc_id),reverse=True); retry[query.query_id]=[SearchResult(x.doc_id,float(s)) for x,s in rranked]
    ce_seconds=time.perf_counter()-ce_started
    pairs=[]; keys=[]
    for query in queries:
        doc_id=initial[query.query_id][0].doc_id
        for sentence_id,sentence in enumerate(by[doc_id].abstract):
            pairs.append({'text':sentence,'text_pair':query.text}); keys.append((query.query_id,doc_id,sentence_id))
    nli_started=time.perf_counter(); probs=nli_probabilities(nli,pairs,args.batch_size); nli_seconds=time.perf_counter()-nli_started
    for (qid,doc_id,sentence_id),score in zip(keys,probs):
        reflection[qid]=max(reflection.get(qid,0.0),score)
    calibration,evaluation=qsplit(queries)
    candidates=[]
    # Every calibration score is a decision boundary. This avoids collapsing a
    # saturated NLI distribution into only "none" or "all" retry decisions.
    thresholds=sorted({0.0,1.0,*[min(1.0,reflection.get(q.query_id,0.0)+1e-8) for q in calibration]})
    for threshold in thresholds:
        run={q.query_id:(retry if reflection.get(q.query_id,0)<threshold else initial)[q.query_id] for q in calibration}
        values=evaluate_run(calibration,run,ks=[1,5,10]); objective=float(values['ndcg@10'])-args.retry_penalty*sum(reflection.get(q.query_id,0)<threshold for q in calibration)/max(len(calibration),1)
        candidates.append({'threshold':threshold,'objective':objective,'retry_rate':sum(reflection.get(q.query_id,0)<threshold for q in calibration)/max(len(calibration),1),**values})
    best=max(candidates,key=lambda x:(x['objective'],-x['retry_rate']))
    conditional={q.query_id:(retry if reflection.get(q.query_id,0)<best['threshold'] else initial)[q.query_id] for q in queries}
    output={
      'status':'real_model_crag_style_ablation','scope':'conditional retrieval correction; not full CRAG/Self-RAG reproduction','split':'SciFact official Dev split into SHA1 calibration/evaluation queries','test_used':False,'query_count':len(queries),'calibration_queries':len(calibration),'evaluation_queries':len(evaluation),'models':{'cross_encoder':str(args.cross_encoder),'sentence_nli':str(args.nli_model),'retry_retriever':'SciFact hard-negative dense checkpoint saved run'},'selected_threshold':best['threshold'],'selection_objective':f'calibration nDCG@10 - {args.retry_penalty} * retry_rate','calibration_curve':candidates,'evaluation':{'bm25':evaluate_run(evaluation,{q.query_id:bm25[q.query_id] for q in evaluation},ks=[1,5,10]),'initial_bm25_ce':evaluate_run(evaluation,initial,ks=[1,5,10]),'retry_all_hybrid_ce':evaluate_run(evaluation,retry,ks=[1,5,10]),'conditional_retry':evaluate_run(evaluation,conditional,ks=[1,5,10])},'retry_count_evaluation':sum(reflection.get(q.query_id,0)<best['threshold'] for q in evaluation),'retry_rate_evaluation':sum(reflection.get(q.query_id,0)<best['threshold'] for q in evaluation)/max(len(evaluation),1),'latency':{'cross_encoder_seconds':ce_seconds,'sentence_nli_seconds':nli_seconds,'total_ms_per_query':(ce_seconds+nli_seconds)*1000/max(len(queries),1)},'reflection_score':'max P(entailment or contradiction) over top-1 abstract sentences','gold_evidence_used_for_input':False}
    args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/'metrics.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf8'); save_run(args.output_dir/'conditional_retry.run.jsonl',conditional); (args.output_dir/'reflection_scores.json').write_text(json.dumps(reflection,indent=2),encoding='utf8'); print(json.dumps(output,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
