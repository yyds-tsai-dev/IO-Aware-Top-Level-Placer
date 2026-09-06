"""Resume the same Stage2 cohort after removing inherited tile signal wires."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'results/stage2_routing_input_fix_20260906'
OLD=REPO/'results/stage2_followup_20260906'
STATE=ROOT/'execution.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write(data):
    p=STATE.with_suffix('.tmp');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(STATE)

def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def reuse_flat(design):
    source=ROOT/(design+'__flat_k16');target=ROOT/(design+'__flat_k32')
    for name in ('out.def','coord.json','netmap.json'):
        if sha(source/name)!=sha(target/name):raise ValueError(f'flat inputs differ:{design}/{name}')
    configs=[json.loads((case/'metrics.json').read_text())['config'] for case in (source,target)]
    if sha(configs[0])!=sha(configs[1]):raise ValueError('flat config differs')
    src=source/'or_run';dst=target/'or_run'
    receipt=json.loads((src/'execution.json').read_text())
    if receipt['status']!='completed' or sha(src/'routed.def')!=receipt['artifacts']['routed.def']:
        raise ValueError('source route incomplete')
    if not json.loads((src/'verify_identity.json').read_text())['overall_pass']:
        raise ValueError('source identity failed')
    dst.mkdir(exist_ok=False)
    for name in ('fixed.def','route.guide','congestion.rpt','drc.rpt','openroad_route.log',
                 'verify_identity.json','routing_input.def','cleared_signal_nets.txt'):
        if (src/name).exists():shutil.copy2(src/name,dst/name)
    record=dict(receipt,run_dir=str(target),record_kind='route_reuse',reused_from=str(source),
                source_execution_sha256=sha(src/'execution.json'),reuse_time=time.time(),
                reuse_basis='byte-identical placement DEF, netmap, CoordMap, config/LEFs',
                target_def_sha256=sha(target/'out.def'))
    record['artifacts']={name:sha(dst/name) for name in record['artifacts'] if (dst/name).exists()}
    record['artifacts']['routed.def']=receipt['artifacts']['routed.def']
    os.link(src/'routed.def',dst/'routed.def')
    temporary=dst/'execution.json.tmp'
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(dst/'execution.json')

def main():
    if STATE.exists():raise FileExistsError('inspect previous corrected execution')
    protocol=json.loads((ROOT/'protocol.json').read_text())
    source=Path(protocol['routing_source'])
    manifest=json.loads((source/'source_manifest.json').read_text())
    for name,want in manifest['files'].items():
        if sha(source/name)!=want:raise ValueError('routing source changed:'+name)
    state=dict(status='running',pid=os.getpid(),started=time.time(),protocol_sha256=sha(ROOT/'protocol.json'),
               legacy_cases=dict(protocol['legacy_active_cases']),routes=[],watchers=[],calibration=None)
    write(state)
    watchers=[]
    # Their scopes are disjoint. Reused FFT and DES evidence retains original
    # absolute paths; calibration resolves the intentional case symlinks.
    for base,pattern,label in ((OLD,'des_perf_1__*','des'),(ROOT,'mempool_tile_wrap__*','tile')):
        log=ROOT/(label+'_evidence.log')
        with log.open('w') as f:
            p=subprocess.Popen([sys.executable,str(Path(protocol['evidence_source'])/'scripts/run_stage2_evidence.py'),
                                '--root',str(base),'--case',pattern,'--watch'],cwd=REPO,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-53b4a8f5-a736-4461-7100-64281fbbfa6d'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        watchers.append(p);state['watchers'].append(dict(scope=pattern,pid=p.pid,log=str(log)))
    pending=list(protocol['new_unique_routes']);active=[];legacy=dict(protocol['legacy_active_cases'])
    try:
        while pending or active or legacy:
            for case,pid in list(legacy.items()):
                if live(pid):continue
                receipt=json.loads((ROOT/case/'or_run/execution.json').read_text())
                if receipt['status']!='completed':raise ValueError('legacy DES route failed:'+case)
                if case.endswith('__flat_k16'):reuse_flat(case.split('__')[0])
                del legacy[case];state['legacy_cases'][case]='completed'
                print('legacy completed',case,flush=True)
            for item in list(active):
                p,row=item
                if p.poll() is None:continue
                row.update(returncode=p.returncode,ended=time.time())
                if p.returncode:raise RuntimeError('corrected route failed:'+row['case'])
                case=row['case'];receipt=json.loads((ROOT/case/'or_run/execution.json').read_text())
                expected=4360 if case.startswith('mempool_tile_wrap') else 0
                if receipt['input_routing_cleanup']['removed_signal_wire_count']!=expected:
                    raise ValueError('unexpected input cleanup set:'+case)
                if case.endswith('__flat_k16'):reuse_flat(case.split('__')[0])
                active.remove(item);print('corrected completed',case,flush=True)
            while pending and len(active)+len(legacy)<2:
                case=pending.pop(0);log=ROOT/(case+'.route.log')
                env=dict(os.environ,IOPLACE_REPO=str(source),STAGE2_S8_ROOT=str((ROOT/case).resolve().parent),
                         STAGE2_ROUTE_THREADS='8',STAGE2_ROUTE_TIMEOUT_S='43200')
                with log.open('w') as f:
                    p=subprocess.Popen(['bash',str(source/'scripts/stage2_s8_route.sh'),'--shard='+case],
                                        cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                row=dict(case=case,pid=p.pid,started=time.time(),log=str(log));active.append((p,row));state['routes'].append(row)
                print('corrected route',case,flush=True)
            for p in watchers:
                if p.poll() not in (None,0):raise RuntimeError('evidence watcher failed; inspect its log')
            write(state);time.sleep(10)
        state['status']='waiting_evidence';write(state)
        while any(p.poll() is None for p in watchers):time.sleep(15)
        if any(p.returncode for p in watchers):raise RuntimeError('paired evidence failed')
        state['status']='calibrating';write(state)
        commands=[ [sys.executable,'-m','ioplace.diagnostics.stage2_calibration','--s8-dir',str(ROOT),'--out-dir',str(ROOT/'calibration')],
                   [sys.executable,str(REPO/'scripts/report_stage2_followup.py'),'--tables',str(ROOT/'calibration/tables.json')] ]
        for i,cmd in enumerate(commands):
            with (ROOT/f'finalize_{i}.log').open('w') as f:subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
        tables=json.loads((ROOT/'calibration/tables.json').read_text())
        if not tables['complete_registered_cohort'] or len(tables['valid_samples'])!=12:
            raise ValueError('corrected cohort did not yield all12 valid samples')
        state.update(status='completed',ended=time.time(),calibration_sha256=sha(ROOT/'calibration/tables.json'))
    except BaseException as error:
        state.update(status='failed',ended=time.time(),error=repr(error));raise
    finally:write(state)

if __name__=='__main__':main()
