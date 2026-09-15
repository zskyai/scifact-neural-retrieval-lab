"""Audit official SciFact corpus and claim splits for leakage and labels."""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

def norm(s): return re.sub(r'[^a-z0-9]+',' ',str(s or '').lower()).strip()
def read(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]
def near(rows):
    seen={}; out=[]
    for r in rows:
        t=norm(r.get('claim')); sig=hashlib.sha1(t.encode()).hexdigest()[:10]
        if sig in seen: out.append({'a':seen[sig],'b':str(r['id'])})
        seen[sig]=str(r['id'])
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    corpus=read(a.data_root/'corpus.jsonl'); splits={s:read(a.data_root/f'claims_{s}.jsonl') for s in ('train','dev','test')}
    labels=Counter(); ev_counts=Counter(); evidence_sentences=0
    for rows in splits.values():
        for r in rows:
            evidence=r.get('evidence') or {}; ev_counts[len(evidence)]+=1
            for annotation in evidence.values():
                values = annotation if isinstance(annotation, list) else [annotation]
                for value in values:
                    if isinstance(value, dict):
                        labels[str(value.get('label', 'UNKNOWN')).upper()] += 1
            evidence_sentences += sum(len(x.get('sentences') or []) if isinstance(x, dict) else len(x or []) for x in evidence.values())
    ids={s:{str(x['id']) for x in rows} for s,rows in splits.items()}
    claim_text={s:{norm(x.get('claim')) for x in rows} for s,rows in splits.items()}
    report={'corpus_abstracts':len(corpus),'splits':{s:len(v) for s,v in splits.items()},'label_distribution':dict(labels),'evidence_sentence_annotations':evidence_sentences,'evidence_documents_per_claim':dict(ev_counts),'duplicate_claim_ids':{f'{a}_{b}':len(ids[a]&ids[b]) for a in ids for b in ids if a<b},'duplicate_claim_text':{f'{a}_{b}':len(claim_text[a]&claim_text[b]) for a in claim_text for b in claim_text if a<b},'near_duplicate_claims':{s:near(v) for s,v in splits.items()},'test_used_for_training':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
