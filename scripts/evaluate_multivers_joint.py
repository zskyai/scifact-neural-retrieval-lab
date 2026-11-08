"""End-to-end BM25 + joint Longformer inference, with train-only calibration."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from retrieval_lab.bm25 import BM25Index, BM25Config
from retrieval_lab.data import load_corpus
from retrieval_lab.multivers_joint import MultiVerSJointModel, encode_windows, LABELS
from retrieval_lab.domain_verifier import classification_metrics
from train_multivers_joint import collate
from train_domain_verifier import group_split

def decision(candidates, stance_threshold, rationale_threshold):
    ranked=[]
    for doc in candidates:
        for window in doc['windows']:
            probs=window['label_probabilities'];label=max(range(3),key=probs.__getitem__)
            if label==0:continue
            evidence=[sid for sid,p in zip(window['sentence_ids'],window['rationale_probabilities']) if p>=rationale_threshold]
            if evidence and probs[label]>=stance_threshold:
                ranked.append((probs[label],doc['doc_id'],label,evidence))
    if not ranked:return {'label':'NEI','confidence':0.,'citations':[],'reason':'no_joint_evidence_above_threshold'}
    ranked.sort(key=lambda x:(-x[0],x[1]));score,doc,label,sentences=ranked[0]
    return {'label':LABELS[label],'confidence':score,'citations':[{'doc_id':doc,'sentence_id':sid} for sid in sentences],'reason':'joint_stance_and_rationale'}

def claim_label(row):
    values={a['label'] for annotations in row['evidence'].values() for a in annotations}
    if len(values)>1:raise ValueError('Conflicting gold claim labels')
    return {'SUPPORT':'SUPPORTS','CONTRADICT':'CONTRADICTS'}.get(next(iter(values),None),'NEI')

def select_thresholds(rows):
    # The whole grid is reported, including refusal behavior; selecting F1
    # cannot silently hide a loss of coverage. No official dev is read here.
    results=[];gold=[claim_label(r['gold']) for r in rows]
    for stance in (.0,.3,.5,.7,.9):
        for rationale in (.3,.5,.7):
            predicted=[decision(r['candidates'],stance,rationale)['label'] for r in rows]
            results.append({'stance_threshold':stance,'rationale_threshold':rationale,**classification_metrics(gold,predicted)})
    best=max(results,key=lambda r:(r['macro_f1'],-r['over_refusal_rate'],-r['stance_threshold']))
    return best,results

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data-root',type=Path,default=Path('data/scifact'));p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['calibration','official_dev'],required=True);p.add_argument('--calibration',type=Path);p.add_argument('--top-k',type=int,default=10);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads)
    manifest=json.loads((a.run/'run_manifest.json').read_text(encoding='utf8'));config=json.loads((a.run/'preflight.json').read_text(encoding='utf8'))
    if not manifest.get('complete'):raise ValueError('Training has not completed')
    train_source=a.data_root/'claims_train.jsonl'
    if hashlib.sha256(train_source.read_bytes()).hexdigest()!=config['source_sha256']:raise ValueError('Training source changed')
    checkpoint_hash=hashlib.sha256((a.run/'model.pt').read_bytes()).hexdigest()
    all_train=[json.loads(s) for s in train_source.read_text(encoding='utf8').splitlines() if s.strip()]
    _,calibration_rows,_=group_split(all_train)
    policy=None
    if a.split=='official_dev':
        if not a.calibration:raise ValueError('Official dev requires a frozen train-only calibration')
        policy=json.loads(a.calibration.read_text(encoding='utf8'))
        if policy['split']!='train_calibration' or policy['checkpoint_sha256']!=checkpoint_hash or policy['top_k']!=a.top_k:raise ValueError('Calibration checkpoint, split or retrieval configuration mismatch')
        if set(policy['claim_ids'])!={str(r['id']) for r in calibration_rows}:raise ValueError('Threshold selection used different claims')
        rows=[json.loads(s) for s in (a.data_root/'claims_dev.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    else:rows=calibration_rows
    a.output.mkdir(parents=True,exist_ok=False)
    documents=load_corpus(a.data_root/'corpus.jsonl');by_id={d.doc_id:d for d in documents};index=BM25Index(documents,BM25Config())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True);model=MultiVerSJointModel(checkpoint=config['model']);model.load_state_dict(torch.load(a.run/'model.pt',weights_only=True,map_location='cpu'));model.eval()
    scored=[]
    with (a.output/'scores.jsonl').open('x',encoding='utf8') as stream:
        for row in rows:
            start=time.perf_counter();hits=index.search(row['claim'],top_k=a.top_k);retrieval_ms=(time.perf_counter()-start)*1000;candidates=[];start=time.perf_counter()
            for hit in hits:
                doc=by_id[str(hit.doc_id)];predictions=[]
                for window in encode_windows(tokenizer,row['claim'],list(doc.abstract),max_length=config['max_length']):
                    batch=collate([dict(input_ids=window.input_ids,global_attention_mask=window.global_attention_mask,sentence_positions=window.sentence_positions,label=0,label_mask=False,rationale_targets=[-1]*len(window.sentence_ids),weight=1.)],tokenizer.pad_token_id)
                    with torch.inference_mode():out=model(**{k:v for k,v in batch.items() if k not in {'labels','label_mask','rationale_targets','weights'}})
                    predictions.append({'sentence_ids':window.sentence_ids,'label_probabilities':out['label_logits'][0].softmax(-1).tolist(),'rationale_probabilities':out['rationale_logits'][0].sigmoid().tolist()})
                candidates.append({'doc_id':str(hit.doc_id),'retrieval_score':float(hit.score),'windows':predictions})
            saved={'claim_id':str(row['id']),'claim':row['claim'],'gold':row,'candidates':candidates,'retrieval_ms':retrieval_ms,'joint_verification_ms':(time.perf_counter()-start)*1000};scored.append(saved);stream.write(json.dumps(saved)+'\n');stream.flush()
            if len(scored)%10==0:print(json.dumps({'completed':len(scored),'expected':len(rows),'split':a.split}),flush=True)
    if a.split=='calibration':
        selected,grid=select_thresholds(scored);policy={'split':'train_calibration','claim_ids':[str(r['id']) for r in rows],'checkpoint_sha256':checkpoint_hash,'top_k':a.top_k,'selected':selected,'grid':grid,'test_used':False,'official_dev_used':False};(a.output/'calibration.json').write_text(json.dumps(policy,indent=2),encoding='utf8')
    with (a.output/'predictions.jsonl').open('x',encoding='utf8') as stream:
        for row in scored:
            prediction=decision(row['candidates'],policy['selected']['stance_threshold'],policy['selected']['rationale_threshold']);prediction['evidence']=[{'doc_id':d['doc_id']} for d in row['candidates']]
            stream.write(json.dumps({'claim_id':row['claim_id'],'claim':row['claim'],'prediction':prediction,'timings_ms':{'retrieval':row['retrieval_ms'],'joint_verification':row['joint_verification_ms']}})+'\n')
    report={'complete':True,'split':a.split,'claims':len(rows),'checkpoint_sha256':checkpoint_hash,'calibration_sha256':hashlib.sha256(a.calibration.read_bytes()).hexdigest() if a.calibration else None,'test_used':False,'retrieval':'BM25','top_k':a.top_k,'generation_tokens':0,'api_cost':0,'mean_retrieval_ms':sum(r['retrieval_ms'] for r in scored)/len(scored),'mean_joint_verification_ms':sum(r['joint_verification_ms'] for r in scored)/len(scored),'thresholds':policy['selected']}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
