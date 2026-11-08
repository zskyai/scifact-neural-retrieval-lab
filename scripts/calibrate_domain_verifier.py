"""Select evidence thresholds only on held-out official-training groups."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.bm25 import BM25Index, BM25Config
from retrieval_lab.data import load_corpus
from retrieval_lab.domain_verifier import DomainEvidenceVerifier, aggregate_units, decide, classification_metrics


def make_units(documents):
    return [unit for doc in documents for unit in ([{'doc_id':doc.doc_id,'kind':'abstract','text':doc.text,'sentence_id':None}] +
            [{'doc_id':doc.doc_id,'kind':'sentence','text':text,'sentence_id':i} for i,text in enumerate(doc.abstract) if text.strip()]) ]


def gold_label(row):
    mapping = {'SUPPORT':'SUPPORTS','CONTRADICT':'CONTRADICTS'}
    labels = {mapping[a['label']] for groups in row['evidence'].values() for a in groups}
    if len(labels) > 1: raise ValueError('Conflicting official labels')
    return next(iter(labels), 'NEI')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data/scifact'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)
    manifest_path = a.checkpoint/'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    ids = set(manifest['calibration_ids'])
    if ids & set(manifest['train_ids']): raise ValueError('Calibration claims were used for training')
    rows = [json.loads(line) for line in (a.data_root/'claims_train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    calibration = [r for r in rows if r['id'] in ids]
    if len(calibration) != len(ids): raise ValueError('Calibration membership mismatch')
    docs = load_corpus(a.data_root/'corpus.jsonl'); by_id = {d.doc_id:d for d in docs}
    index = BM25Index(docs, BM25Config())
    verifier = DomainEvidenceVerifier(a.checkpoint)
    scored = []
    with (a.output/'scores.jsonl').open('x',encoding='utf-8') as stream:
        for i,row in enumerate(calibration):
            candidates = [by_id[h.doc_id] for h in index.search(row['claim'],top_k=a.top_k)]
            units = verifier.score_pairs(row['claim'],make_units(candidates))
            result = {'claim_id':row['id'],'gold_label':gold_label(row),'ranked':aggregate_units(units),'units':units}
            scored.append(result); stream.write(json.dumps(result)+'\n'); stream.flush()
            if (i+1)%25 == 0: print(json.dumps({'calibrated_queries':i+1,'expected':len(calibration)}),flush=True)
    gold = [r['gold_label'] for r in scored]; grid=[]
    for threshold in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
        for margin in [0,.03,.06,.1,.15]:
            for bias in [-.1,-.05,0,.05,.1]:
                config = {'threshold':threshold,'min_stance_margin':margin,'support_bias':bias}
                predictions = [decide(r['ranked'],config)['label'] for r in scored]
                metrics = classification_metrics(gold,predictions)
                grid.append({'config':config,'metrics':metrics,'coverage':sum(x!='NEI' for x in predictions)/len(predictions)})
    best = max(grid,key=lambda r:(r['metrics']['macro_f1'],r['metrics']['accuracy'],-r['metrics']['unsafe_answer_rate'],-r['metrics']['over_refusal_rate']))
    config = {**best['config'],'status':'heldout_training_groups_calibrated','calibration_claims':len(calibration),
              'retrieval':'bm25','top_k':a.top_k,'max_retries':0,'objective':'claim_macro_f1',
              'calibration_ids':sorted(ids),'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'dev_used':False,'test_used':False,'calibration_metrics':best['metrics']}
    (a.output/'calibration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (a.output/'coverage_risk_grid.json').write_text(json.dumps(grid,indent=2),encoding='utf-8')
    print(json.dumps(config),flush=True)


if __name__ == '__main__': main()
