import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from ioplace.paths import REPO_ROOT


@pytest.mark.slow
def test_real_online_driver_consumes_router_feedback_before_next_update(tmp_path):
    binary=os.environ.get('OPENROAD_BIN')
    if not binary or not Path(binary).exists():pytest.skip('OpenROAD runtime required')
    root=Path(REPO_ROOT)
    config=root/'results/route_feedback_20260914/gcd.json'
    placement=root/'results/route_feedback_20260914/gcd_swap32_seed1000/baseline/placement.npz'
    if not placement.exists():pytest.skip('real GCD checkpoint required')
    command=[sys.executable,str(root/'src/scripts/run_joint_route_feedback.py'),
        '--config',str(config),'--placement',str(placement),'--openroad',binary,
        '--out',str(tmp_path/'online'),'--rounds','2','--max-active','16','--neighbors','8']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr[-2000:]
    report=json.loads((tmp_path/'online/result.json').read_text())
    assert report['cohort']['eligible_multi_pin_nets']>0
    assert len(report['rounds'])==2
    assert sum(r['proposal']['multi_pin_rebuilds'] for r in report['rounds'])>0
    assert report['rounds'][1]['generation']>report['rounds'][0]['generation']
    assert report['rounds'][1]['last_observation']
    assert len(report['observations'])>=2
    assert (tmp_path/'online/selected.npz').exists()
