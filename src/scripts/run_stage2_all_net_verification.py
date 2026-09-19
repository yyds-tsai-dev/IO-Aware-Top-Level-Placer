"""Supplement frozen Stage2 evidence with all-routed-net geometry checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(root, row):
    name = row['design']+'__'+row['arm']
    case = (root/name).resolve()
    receipt_path = case/'evidence.execution.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'completed':
        raise ValueError('primary evidence incomplete')
    commands = [r['command'] for r in receipt['commands']
                if any(str(x).endswith('/verify_routed_def.py') for x in r['command'])]
    if len(commands) != 1:
        raise ValueError('expected exactly one recorded verifier command')
    command = commands[0]
    lefs = [command[i+1] for i,x in enumerate(command) if x == '--lef']
    def_path = command[command.index('--def')+1]
    script = Path(command[command.index('-python')+1])
    files = [command[0], str(script), str(script.parent/'dump_segments.py'),
             str(script.parent.parent/'def_text_parser.py'), def_path, *lefs]
    inputs = {str(Path(p).resolve()): sha(p) for p in files}
    for path,digest in inputs.items():
        if receipt['inputs'].get(path) != digest:
            raise ValueError('unregistered or changed verifier input: '+path)
    key = (sha(def_path), tuple(sha(p) for p in lefs), sha(command[0]), sha(script))
    return dict(name=name, original_command=command, inputs=inputs,
                receipt_path=str(receipt_path), receipt_sha256=sha(receipt_path), key=key)


def run_one(item, outroot):
    out = outroot/item['name']
    out.mkdir(parents=True, exist_ok=False)
    command = list(item['original_command'])
    for flag,value in (('--sample','0'), ('--json-out',str(out/'verify.json')),
                       ('--report-out',str(out/'wire_length.rpt'))):
        command[command.index(flag)+1] = value
    started = time.time()
    with (out/'verify.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    log_text = (out/'verify.log').read_text()
    report = json.loads((out/'verify.json').read_text()) if (out/'verify.json').exists() else {}
    check = report.get('check2_odb_vs_text_parser', {})
    n = check.get('routed_net_count', 0)
    checks = dict(
        full_population=n>0 and check.get('sample_requested')==0 and
            all(check.get(k)==n for k in ('n_sampled','n_checked','n_match')),
        zero_disagreement=all(check.get(k)==0 for k in
            ('n_mismatch','n_mismatch_rect_only','n_mismatch_other','n_missing_from_text')),
        wire_oracle=report.get('check1_wire_length',{}).get('pass_lt_1pct') is True,
        no_unvalidated_junction=report.get('junction_unvalidated') is False,
        overall=report.get('overall_pass') is True and '[verify] OVERALL: PASS' in log_text,
        inputs_unchanged=all(sha(p)==v for p,v in item['inputs'].items()))
    # This build propagates Python SystemExit(0) as OpenROAD rc=1. Require
    # the exact marker plus independent completed report, never generic rc=1.
    embedded_zero = result.returncode==1 and 'SystemExit: 0' in log_text
    checks['process_completion'] = result.returncode==0 or embedded_zero
    record = dict(item, key=list(item['key']), command=command,
        started=started, ended=time.time(), returncode=result.returncode,
        embedded_python_system_exit_zero=embedded_zero, checks=checks,
        status='completed' if all(checks.values()) else 'failed',
        outputs={str(p):sha(p) for p in out.iterdir() if p.is_file()})
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(item['name'],record['status'],n,'nets',flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve()
    out=root/'all_net_verification'; out.mkdir(exist_ok=True)
    items=[prepare(root,row) for row in json.loads((root/'cohort.json').read_text())]
    groups={}
    for item in items: groups.setdefault(item['key'],[]).append(item)
    registration=dict(started=time.time(),source_sha256=sha(__file__),max_workers=2,
        case_count=len(items),unique_routes=len(groups),
        rule='all routed nets canonical geometry-set equality; wire widths and duplicate multiplicity are not compared',
        cases=[{k:v for k,v in i.items() if k!='key'} for i in items])
    (out/'protocol.json').write_text(json.dumps(registration,indent=2)+'\n')
    completed={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_one,rows[0],out):rows for rows in groups.values()}
        for future in as_completed(futures):
            rows=futures[future]; record=future.result()
            for item in rows:
                completed[item['name']]=dict(status=record['status'],
                    representative=rows[0]['name'], input_def_sha256=item['key'][0],
                    lef_sha256=list(item['key'][1]),
                    verification_receipt=str(out/rows[0]['name']/'execution.json'),
                    verification_receipt_sha256=sha(out/rows[0]['name']/'execution.json'))
            (out/'execution.json').write_text(json.dumps(dict(status='running',cases=completed),indent=2)+'\n')
    okay=len(completed)==len(items) and all(r['status']=='completed' for r in completed.values())
    (out/'execution.json').write_text(json.dumps(dict(status='completed' if okay else 'failed',
        ended=time.time(),protocol_sha256=sha(out/'protocol.json'),cases=completed),indent=2)+'\n')
    return 0 if okay else 1


if __name__=='__main__':
    raise SystemExit(main())
