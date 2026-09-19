"""Create schema4 caches in a new namespace; preserve every schema3 experiment."""
import gc
import json
import os
from pathlib import Path
import sys
import time
REPO = Path(__file__).resolve().parents[1]
REPO = REPO.parent if REPO.name == "src" else REPO
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ioplace.bench.bookshelf_netlist import build_tiled_netlist_cache,verify_against_bookshelf
from scripts.validate_full_gpu_evaluator import sha,write
ROOT=REPO/'results/recovery_visible_20260906'
OUT=ROOT/'cache_schema4'

def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'protocol.json',dict(started=time.time(),reason='native ordered net pin traversal including sequential output-front swaps',source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).resolve().parents[1] / 'ioplace/bench/bookshelf_netlist.py',Path(__file__).resolve().parents[1] / 'ioplace/bench/native_pin_order.py',Path(__file__).resolve().parents[1] / 'ioplace/bench/native_pin_order.cpp']},preserve_legacy_caches=True))
    state=dict(status='running',pid=os.getpid(),cases={})
    write(OUT/'execution.json',state)
    for case in ('3x3_n2','1x2_n2','2x2_n2'):
        target=OUT/case
        manifest=ROOT/'arrays'/case/(case+'.manifest.json')
        before=time.time()
        meta=build_tiled_netlist_cache(str(manifest),str(target))
        state['cases'][case]=dict(status='built',build_s=time.time()-before,meta_sha256=sha(target/'meta.json'))
        write(OUT/'execution.json',state)
        gc.collect()
    for case in ('3x3_n2','1x2_n2','2x2_n2'):
        target=OUT/case
        verify=verify_against_bookshelf(str(target),str(ROOT/'arrays'/case/case),mode='full')
        write(target/'verification.json',dict(verify=verify,source_protocol_sha256=sha(OUT/'protocol.json')))
        if not verify['ok'] or not verify['native_pin_order_verified']:
            raise ValueError(f'{case} failed full ordered verification')
        state['cases'][case].update(status='verified',verification_sha256=sha(target/'verification.json'))
        write(OUT/'execution.json',state)
        gc.collect()
    state.update(status='completed',ended=time.time());write(OUT/'execution.json',state)

if __name__=='__main__': main()
