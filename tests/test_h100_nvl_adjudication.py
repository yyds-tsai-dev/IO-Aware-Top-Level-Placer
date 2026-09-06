import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_h100_nvl_prediction.py'
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixture(tmp_path, timeout=False, bad_history=False, bad_coords=False, outside_band=False):
    (tmp_path/'forecast').mkdir(); snap=tmp_path/'source'; snap.mkdir()
    source=snap/'module.py'; source.write_text('value = 1\n')
    cfg=tmp_path/'config.json'; cfg.write_text('{}'); inp=tmp_path/'input.json'; inp.write_text('{}')
    log=tmp_path/'full_3x3_flat_k16.log'; log.write_text('log')
    dev=tmp_path/'full_3x3_flat_k16.device.jsonl'; dev.write_text(json.dumps({'used_gib':1})+'\n')
    phases={n:{'t_s':10.,'peak_alloc_gb':1.} for n in ('read','gp','lg','eval')}
    pred={'status':'frozen','execution_protocol_sha256':'','wall_time_forecast':{'total_fast_s':35,'total_slow_s':45,'phases':{n:{'t_l4_extrapolated_27m_s':10,'s_p_lo':1,'s_p_hi':1} for n in phases}},'host_rss_forecast':{'point_estimate_gb':1},'gpu_memory_forecast':{'lower_bound_gb':1,'upper_bound_gb':2}}
    protocol={'registered_at_unix':1,'gpu_uuid':'u','source_snapshot_digest':'x','config':str(cfg),'config_sha256':h(cfg),'source_snapshot':str(snap),'source_sha256':{'module.py':h(source)},'input_manifest':str(inp),'input_manifest_sha256':h(inp),'physical_nodes':2,'movable_nodes':2,'nets':1,'raw_pins':2,'canonical_pins':2}
    (tmp_path/'forecast/protocol.json').write_text(json.dumps(protocol)); pred['execution_protocol_sha256']=h(tmp_path/'forecast/protocol.json'); (tmp_path/'forecast/h100_nvl_prediction.json').write_text(json.dumps(pred))
    actual={'k':16,'rtype':'grid','mode':'io','rho_max':0,'dp_seed':1000,'callback_order':'legacy','f_ft_max':0,'ft_reweight':'off','wl_reweight':'off','no_diag':True,'phases':phases,'peak_mem_mb':1024,'device_used_gb':1,'host_peak_rss_gb':1,'legalization_status':'success','num_unplaced_cells':0,'stop_overflow_reached':False,'final_overflow':0}
    if outside_band: actual['phases']['eval']['t_s']=100
    (tmp_path/'full_3x3_flat_k16.json').write_text(json.dumps(actual)); np.savez(tmp_path/'full_3x3_flat_k16.json.npz',node_x=np.array([1.,2.]) if not bad_coords else np.array([1.]),node_y=np.array([1.,2.]))
    receipt={'status':'completed' if not timeout else 'timeout','returncode':0,'output_sha256':h(tmp_path/'full_3x3_flat_k16.json'),'protocol_sha256':h(tmp_path/'forecast/protocol.json'),'prediction_sha256':h(tmp_path/'forecast/h100_nvl_prediction.json'),'started':2,'gpu_uuid':'u','config_sha256':h(cfg),'source_snapshot_digest':'x','measurement_mode':'shared','supervisor_wall_s':40,'log_sha256':h(log),'device_history_sha256':h(dev)}
    (tmp_path/'full_3x3_flat_k16.execution.json').write_text(json.dumps(receipt))
    if bad_history: dev.write_text(json.dumps({'used_gib':2})+'\n')
    return tmp_path

def run(root): return subprocess.run([sys.executable,str(SCRIPT),'--root',str(root)],capture_output=True,text=True)
def test_success_protocol_compatible(tmp_path):
    r=run(fixture(tmp_path)); assert r.returncode==0; out=json.loads((tmp_path/'prediction_check.json').read_text()); assert out['protocol_compatible'] and out['mainline_hypothesis']['confirmed']; assert out['gpu_memory']['verdict']=='not_applicable_shared_attribution' and not out['exclusive_speedup_claim']
def test_timeout_receipt_is_rejected(tmp_path):
    r=run(fixture(tmp_path,timeout=True)); assert r.returncode != 0; assert 'did not complete' in r.stderr
def test_corrupted_history_hash_rejected(tmp_path):
    r=run(fixture(tmp_path,bad_history=True)); assert r.returncode != 0 and 'execution artifact changed' in r.stderr
def test_wrong_coordinate_count_rejected(tmp_path):
    r=run(fixture(tmp_path,bad_coords=True)); assert r.returncode != 0 and 'physical node count mismatch' in r.stderr

def test_outside_frozen_band_is_disclosed_without_widening(tmp_path):
    root=fixture(tmp_path,outside_band=True)
    pred=root/'forecast/h100_nvl_prediction.json'; before=h(pred)
    r=run(root); assert r.returncode==0,r.stderr
    out=json.loads((root/'prediction_check.json').read_text())
    assert out['mainline_hypothesis']['confirmed'] is False
    assert out['required_disclosure'] and out['wall_time']['band_high']==45
    assert out['phase_checks']['phases']['eval']['same_order'] is False
    assert h(pred)==before

def test_changed_frozen_source_is_rejected(tmp_path):
    root=fixture(tmp_path); (root/'source/module.py').write_text('value = 2\n')
    r=run(root); assert r.returncode!=0 and 'frozen source changed' in r.stderr
