"""Export latest human case decisions for reproducible error analysis."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', type=Path, default=Path('results/research_cases.sqlite3'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(a.db.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        records = db.execute('SELECT c.*, r.reviewer, r.verdict, r.note, r.created_at AS reviewed_at FROM cases c JOIN reviews r ON c.case_id=r.case_id WHERE r.review_id=(SELECT MAX(r2.review_id) FROM reviews r2 WHERE r2.case_id=c.case_id) ORDER BY c.created_at').fetchall()
    with a.output.open('x', encoding='utf-8') as out:
        for record in records:
            row = dict(record); row['result'] = json.loads(row.pop('result_json'))
            row.update({'purpose':'human_review_candidate', 'training_eligible':False,
                        'disagrees_with_model':row['verdict'] != row['result'].get('label'),
                        'required_checks':['split_membership', 'human_evidence_selection', 'source_span_match']})
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'exported':len(records), 'training_eligible':False, 'output':str(a.output)}))


if __name__ == '__main__': main()
