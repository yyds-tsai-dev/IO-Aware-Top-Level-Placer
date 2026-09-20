"""Bounded intermediate routing must keep congestion and provenance visible."""
from types import SimpleNamespace
import json

import pytest

from ioplace.route_eval import online_openroad as adapter
from ioplace.route_eval.or_scripts import dump_online_route as router


@pytest.mark.parametrize("options", [
    {"congestion_iterations": 0}, {"congestion_iterations": True},
    {"threads": -1}, {"threads": 1.5}, {"allow_congestion": "false"},
])
def test_invalid_policy_rejected_before_creating_output(tmp_path, options):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        adapter.run_openroad("missing.def", [], out, "missing-openroad", **options)
    assert not out.exists()


def test_limited_routing_keeps_congestion_and_writes_segments(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "checked_eval", lambda _, command, label: calls.append(command))
    router._run_commands(object(), dict(clear_script="/clear.tcl", cleared="/cleared.txt",
        segments="/segments.txt", congestion_iterations=5, allow_congestion=True, threads=8))
    assert calls[0] == "set_thread_count 8"
    assert calls[4] == "global_route -congestion_iterations 5 -allow_congestion -verbose"
    assert calls[-1] == "grt::write_segments {/segments.txt}"


def test_native_congestion_uses_final_report_not_intermediate_totals():
    log = """Total 100 1 1.00% 0 / 0 / 0
[INFO GRT-0096] Final congestion report:
Layer Resource Demand Usage (%) Max H / Max V / Total Overflow
metal1 50 80 160.00% 3 / 0 / 9
Total 100 120 120.00% 3 / 2 / 13
[INFO GRT-0018] Total wirelength: 456 um
[INFO GRT-0014] Routed nets: 7
"""
    measured = adapter.parse_native_congestion(log)
    assert measured == dict(resource=100, demand=120, usage_percent=120.,
        max_horizontal_overflow=3, max_vertical_overflow=2, total_overflow=13)
    with pytest.raises(ValueError, match="final congestion"):
        adapter.parse_native_congestion("Total 100 0 0% 0 / 0 / 0")


def test_final_routing_does_not_inherit_fast_feedback_policy(monkeypatch):
    monkeypatch.setenv("IOPLACE_ENABLE_GR_IN_LOOP", "1")
    from scripts.run_route_gp import routing_policy
    args = SimpleNamespace(feedback_grt_iterations=5, feedback_allow_congestion=True,
        final_grt_iterations=50, final_allow_congestion=False, grt_threads=4)
    assert routing_policy(args, final=False) == dict(congestion_iterations=5, allow_congestion=True, threads=4)
    assert routing_policy(args, final=True) == dict(congestion_iterations=50, allow_congestion=False, threads=4)


def test_net_audit_preserves_high_degree_signals_and_connectivity(tmp_path):
    def pin(name):
        return SimpleNamespace(getName=lambda: name)
    def net(name, count, special=False):
        return SimpleNamespace(getName=lambda: name, getSigType=lambda: "POWER" if special else "SIGNAL",
            isSpecial=lambda: special, getITerms=lambda: [], getBTerms=lambda: [pin(f"p{i}") for i in range(count)])
    nets = [net("wide_signal", 1001), net("VDD", 2000, True), net("local", 2)]
    manifest = tmp_path / "nets.jsonl"
    audit = router.audit_netlist(SimpleNamespace(getNets=lambda: nets), manifest)
    assert audit["net_count"] == 3 and audit["special_net_count"] == 1
    assert audit["degree_above_counts"] == {"100": 2, "1000": 2, "10000": 0}
    assert audit["requested_exclusions"] == [] and not audit["sampling"]
    assert len(manifest.read_text().splitlines()) == 3
    reverse = router.audit_netlist(SimpleNamespace(getNets=lambda: list(reversed(nets))), manifest)
    assert reverse["connectivity_sha256"] == audit["connectivity_sha256"]
    nets[0] = net("wide_signal", 1000)
    changed = router.audit_netlist(SimpleNamespace(getNets=lambda: nets), manifest)
    assert changed["connectivity_sha256"] != audit["connectivity_sha256"]


def test_net_audit_distinguishes_delimiters_in_instance_and_pin_names(tmp_path):
    def block(instance, terminal):
        pin=SimpleNamespace(getInst=lambda:SimpleNamespace(getName=lambda:instance),
            getMTerm=lambda:SimpleNamespace(getName=lambda:terminal))
        net=SimpleNamespace(getName=lambda:'n',getSigType=lambda:'SIGNAL',isSpecial=lambda:False,
            getITerms=lambda:[pin],getBTerms=lambda:[])
        return SimpleNamespace(getNets=lambda:[net])
    a=router.audit_netlist(block('a/b','c'),tmp_path/'a.jsonl')
    b=router.audit_netlist(block('a','b/c'),tmp_path/'b.jsonl')
    assert a['connectivity_sha256'] != b['connectivity_sha256']


def test_strict_failure_keeps_native_overflow_and_hashed_log(tmp_path, monkeypatch):
    design=tmp_path/'in.def';design.write_text('fixture')
    binary=tmp_path/'openroad';binary.write_text('fixture')
    def fail(command,stdout,stderr):
        stdout.write('[INFO GRT-0096] Final congestion report:\nTotal 100 120 120.00% 3 / 2 / 13\n[ERROR GRT-0116] overflow\n')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(adapter.subprocess,'run',fail)
    out=tmp_path/'route'
    with pytest.raises(RuntimeError):adapter.run_openroad(design,[],out,binary,congestion_iterations=5)
    receipt=json.loads((out/'receipt.json').read_text())
    assert receipt['returncode']==1 and receipt['native_congestion']['total_overflow']==13
    assert receipt['outputs']['run.log']==adapter.digest(out/'run.log')


def test_signal_layer_policy_runs_before_global_routing(monkeypatch):
    calls=[]
    monkeypatch.setattr(router,'checked_eval',lambda _,command,label:calls.append(command))
    router._run_commands(object(),dict(clear_script='/c.tcl',cleared='/c.txt',segments='/s.txt',
        signal_layers='metal2-metal10',congestion_iterations=5,allow_congestion=True))
    layer=calls.index('set_routing_layers -signal metal2-metal10')
    assert layer < next(i for i,c in enumerate(calls) if c.startswith('global_route '))
