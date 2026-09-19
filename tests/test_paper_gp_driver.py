"""Real DREAMPlace integration: an extra scalar must actually change GP updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ioplace.paths import REPO_ROOT


@pytest.mark.slow
def test_paper_gradient_enters_gp_after_warmup_with_frozen_step_topology(tmp_path):
    root=Path(REPO_ROOT)
    config=root/'results/route_feedback_20260914/gcd.json'
    if not config.exists():pytest.skip('GCD runtime config required')
    reports={}
    for mode in ('wa','paper'):
        out=tmp_path/mode
        command=[sys.executable,str(root/'src/scripts/run_paper_gp.py'),'--config',str(config),
            '--out',str(out),'--mode',mode,'--iterations','24','--start','5','--rebuild','1']
        run=subprocess.run(command,capture_output=True,text=True)
        assert run.returncode==0,run.stderr[-3000:]+run.stdout[-1000:]
        reports[mode]=json.loads((out/'result.json').read_text())
    a,b=reports['wa'],reports['paper']
    assert a['initial_sha256']==b['initial_sha256']
    assert a['legal'] and b['legal'] and a['fixed_unchanged'] and b['fixed_unchanged']
    assert a['trace'][4]['position_sha256']==b['trace'][4]['position_sha256']
    active=[row for row in b['trace'] if row['active']]
    assert active and all(row['cache_generation']==row['generation'] for row in active)
    assert any(row['interior_gradient_l1']>0 for row in active)
    assert b['gradient_audit']['added_l1']>0
    assert b['gradient_audit']['gradient_sum_max_error']<=b['gradient_audit']['tolerance']
    assert b['topology_rebuilds']==len(active)
    assert a['cache_refreshes']==b['cache_refreshes']==len(active)
    assert b['gp_sha256']!=a['gp_sha256']
    assert all(row['forward_generation_min']==row['forward_generation_max']==row['generation'] for row in active)
    with np.load(tmp_path/'paper/gp.npz') as positions:
        assert np.isfinite(positions['node_x']).all()
