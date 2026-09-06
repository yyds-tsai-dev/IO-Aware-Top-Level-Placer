"""A polling evidence reader must never observe a partial reused route."""
import json
from pathlib import Path
import pytest
from scripts import run_corrected_stage2_routes as route


def test_completed_reuse_is_published_after_all_artifacts(tmp_path,monkeypatch):
    monkeypatch.setattr(route,'ROOT',tmp_path)
    config=tmp_path/'config.json';config.write_text('{}')
    for k in (16,32):
        case=tmp_path/f'toy__flat_k{k}';case.mkdir()
        for name in ('out.def','coord.json','netmap.json'): (case/name).write_text(name)
        (case/'metrics.json').write_text(json.dumps({'config':str(config)}))
    src=tmp_path/'toy__flat_k16/or_run';src.mkdir()
    (src/'routed.def').write_text('complete geometry')
    (src/'fixed.def').write_text('placement geometry')
    (src/'verify_identity.json').write_text('{"overall_pass":true}')
    (src/'execution.json').write_text(json.dumps({'status':'completed','artifacts':{'routed.def':route.sha(src/'routed.def')}}))
    original_write=Path.write_text;original_replace=Path.replace
    def ready(path):
        assert (path.parent/'routed.def').read_text()=='complete geometry'
        assert (path.parent/'fixed.def').exists()
        assert json.loads((path.parent/'verify_identity.json').read_text())['overall_pass']
    def write(path,*args,**kwargs):
        if path.name=='execution.json':ready(path)
        return original_write(path,*args,**kwargs)
    def replace(path,target):
        if Path(target).name=='execution.json':ready(Path(target))
        return original_replace(path,target)
    monkeypatch.setattr(Path,'write_text',write);monkeypatch.setattr(Path,'replace',replace)
    route.reuse_flat('toy')
    dst=tmp_path/'toy__flat_k32/or_run'
    receipt=json.loads((dst/'execution.json').read_text())
    assert receipt['record_kind']=='route_reuse'
    assert receipt['artifacts']['routed.def']==route.sha(dst/'routed.def')
    assert not (dst/'execution.json.tmp').exists()
    with pytest.raises(FileExistsError):route.reuse_flat('toy')
