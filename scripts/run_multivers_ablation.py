"""Run matched, one-factor Longformer ablations with the same scorer."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

CONFIGURATIONS = [
    ('baseline', 10, 1024, 'known_related', 'complete'),
    ('neg20', 20, 1024, 'known_related', 'complete'),
    ('window512', 10, 512, 'known_related', 'complete'),
    ('window4096', 10, 4096, 'known_related', 'complete'),
    ('no_false_negative_filter', 10, 1024, 'none', 'complete'),
    ('membership_supervision', 10, 1024, 'known_related', 'membership'),
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--existing-baseline',type=Path)
    p.add_argument('--official-evaluator',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=[];scripts=Path(__file__).resolve().parent
    def invoke(command,log):
        with log.open('a',encoding='utf8') as stream:
            completed=subprocess.run([sys.executable,*map(str,command)],stdout=stream,stderr=subprocess.STDOUT,check=False)
        if completed.returncode:raise RuntimeError('Command failed; inspect '+str(log))
    def save():
        report={'complete':len(records)==len(CONFIGURATIONS),'mode':'train_and_official_dev_evaluation' if a.execute else 'preflight_only','seed':42,'test_used':False,'official_evaluator_sha256':hashlib.sha256(a.official_evaluator.read_bytes()).hexdigest(),'ablations':records}
        (a.output/'suite.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    for name,negatives,length,filter_mode,rationale in CONFIGURATIONS:
        run=a.existing_baseline if name=='baseline' and a.existing_baseline else a.output/name
        command=[scripts/'train_multivers_joint.py','--model',a.model,'--output',run,'--negatives',negatives,'--max-length',length,'--false-negative-filter',filter_mode,'--rationale-mode',rationale,'--threads',a.threads]
        record={'name':name,'run':str(run),'negatives':negatives,'max_length':length,'false_negative_filter':filter_mode,'rationale_mode':rationale}
        try:
            if not (run/'preflight.json').exists():
                invoke([*command,'--preflight-only'],a.output/(name+'.log'))
            preflight=json.loads((run/'preflight.json').read_text(encoding='utf8'))
            for key,value in [('negatives',negatives),('max_length',length),('false_negative_filter',filter_mode),('rationale_mode',rationale)]:
                if preflight[key]!=value:raise ValueError('Existing run configuration mismatch: '+key)
            record['preflight']=preflight
            record['status']='prepared'
            if a.execute:
                if name=='baseline' and a.existing_baseline and not (run/'run_manifest.json').exists():
                    raise RuntimeError('Existing baseline training is still incomplete; refusing to start a duplicate trainer')
                if not (run/'run_manifest.json').exists():invoke([*command,'--resume'],a.output/(name+'.log'))
                calibration=a.output/(name+'_calibration');dev=a.output/(name+'_dev300');official=a.output/(name+'_official')
                common=[scripts/'evaluate_multivers_joint.py','--run',run,'--threads',a.threads]
                if not (calibration/'summary.json').exists():invoke([*common,'--split','calibration','--output',calibration],a.output/(name+'.log'))
                if not (dev/'summary.json').exists():invoke([*common,'--split','official_dev','--calibration',calibration/'calibration.json','--output',dev],a.output/(name+'.log'))
                if not (official/'summary.json').exists():invoke([scripts/'score_official_scifact.py','--predictions',dev/'predictions.jsonl','--official-evaluator',a.official_evaluator,'--output',official],a.output/(name+'.log'))
                record['status']='evaluated';record['metrics']=json.loads((official/'summary.json').read_text(encoding='utf8'))
        except (RuntimeError,ValueError,FileNotFoundError) as error:
            record['status']='failed';record['error']=str(error)
        records.append(record);save();print(json.dumps({'name':name,'status':record['status']}),flush=True)
    if any(r['status']=='failed' for r in records):raise SystemExit(1)
if __name__=='__main__':main()
