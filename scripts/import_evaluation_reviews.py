"""Put real evaluation errors in the review desk without rerunning inference."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_lab.case_store import ResearchCaseStore


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--db', type=Path, default=Path('results/research_cases.sqlite3'))
    a = p.parse_args()
    summary = json.loads(a.predictions.with_name('summary.json').read_text(encoding='utf-8'))
    if not summary.get('complete') or summary.get('split') != 'official_dev':
        raise ValueError('Requires a complete official-dev evaluation')
    digest = hashlib.sha256(a.predictions.read_bytes()).hexdigest()
    store = ResearchCaseStore(a.db)
    existing = {item['result'].get('review_context', {}).get('origin_key') for item in store.list(100000)}
    imported = 0
    for line in a.predictions.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if not row.get('errors'): continue
        origin = digest + ':' + str(row['claim_id'])
        if origin in existing: continue
        result = dict(row['prediction'])
        result['review_context'] = {'origin_key':origin, 'source_run_sha256':digest, 'dataset':'SciFact',
                                    'split':'official_dev', 'claim_id':row['claim_id'], 'gold_label':row['gold_label'],
                                    'errors':row['errors'], 'training_eligible':False}
        store.create(row['claim'], result)
        existing.add(origin); imported += 1
    print(json.dumps({'imported_for_review':imported, 'human_reviews_created':0, 'training_eligible':False}))


if __name__ == '__main__': main()
