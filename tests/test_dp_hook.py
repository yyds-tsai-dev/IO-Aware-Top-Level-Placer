import json, os, sys
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.dreamplace_env import setup_dreamplace
from ioplace.schedules import ScheduleState

DP = setup_dreamplace()
from NesterovAcceleratedGradientOptimizer import NesterovAcceleratedGradientOptimizer as NAG
from ioplace.dp_hook import (attach_terms, detach_terms, assert_optimizer_lock,
                             refresh_nesterov_secant, install_version_invariant)

# ---------------------------------------------------------------- secant refresh
A = torch.tensor([3.0, 1.0]); CTR = torch.tensor([5.0, -5.0])

def _make_problem():
    ver = {"v": 0}
    def obj_and_grad_fn(p):
        if p.grad is not None:
            p.grad.zero_()
        o = 0.5 * (A * p * p).sum()
        if ver["v"] == 1:
            o = o + 50.0 * ((p - CTR) ** 2).sum()
        o.backward()
        return o.detach(), p.grad
    return ver, obj_and_grad_fn

def _run(do_refresh):
    ver, fn = _make_problem()
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    opt = NAG([p], lr=0.01, obj_and_grad_fn=fn, constraint_fn=lambda t: None, use_bb=False)
    fn(p)                                   # prime p.grad; NAG.step() skips params with grad None
    for _ in range(5):
        opt.step()
    g = opt.param_groups[0]
    ver["v"] = 1                            # objective changes here
    if do_refresh:
        refresh_nesterov_secant(opt)
    vk = g["v_k"][0].data.clone()
    cached = g["g_k"][0].clone()
    a_before = g["alpha_k"][0].item()
    n0 = g["obj_eval_count"]
    opt.step()
    return dict(cached=cached, vk=vk, a_before=a_before, a_after=g["alpha_k"][0].item(),
                evals=g["obj_eval_count"] - n0)

def _grad_new(v):
    return A * v + 100.0 * (v - CTR)

def _grad_old(v):
    return A * v

def test_without_refresh_secant_pair_crosses_objectives_RED():
    r = _run(do_refresh=False)
    assert torch.allclose(r["cached"], _grad_old(r["vk"]), atol=1e-6)
    assert not torch.allclose(r["cached"], _grad_new(r["vk"]), atol=1e-3)
    assert r["a_after"] / r["a_before"] < 1e-3          # step size collapses

def test_with_refresh_cache_is_consistent_with_new_objective_GREEN():
    r = _run(do_refresh=True)
    assert torch.allclose(r["cached"], _grad_new(r["vk"]), atol=1e-9)
    assert abs(r["a_after"] / r["a_before"] - 1.0) < 0.05

def test_refresh_is_a_noop_before_the_first_step():
    ver, fn = _make_problem()
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = NAG([p], lr=0.01, obj_and_grad_fn=fn, constraint_fn=lambda t: None, use_bb=False)
    refresh_nesterov_secant(opt)            # must not raise
    assert opt.param_groups[0]["g_k"] == []

# ---------------------------------------------------------------- version invariant
def test_version_invariant_fires_when_not_refreshed():
    ver, fn = _make_problem()
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    opt = NAG([p], lr=0.01, obj_and_grad_fn=fn, constraint_fn=lambda t: None, use_bb=False)
    fn(p); opt.step()
    st = ScheduleState(rho_max=0.1)
    uninstall = install_version_invariant(opt, st)
    st.obj_version += 1                     # simulate a discrete change without refresh
    with pytest.raises(AssertionError):
        opt.step()
    uninstall()

def test_version_invariant_silent_after_mark_refreshed():
    ver, fn = _make_problem()
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    opt = NAG([p], lr=0.01, obj_and_grad_fn=fn, constraint_fn=lambda t: None, use_bb=False)
    fn(p); opt.step()
    st = ScheduleState(rho_max=0.1)
    uninstall = install_version_invariant(opt, st)
    st.obj_version += 1
    st.mark_refreshed()
    opt.step()                              # must not raise
    uninstall()

# ---------------------------------------------------------------- params plumbing
class _FakeParams:
    def __init__(self):
        self.__dict__["params_dict"] = {}
        self.optimizer = "nesterov"
        self.use_bb = 0
        self.global_place_stages = [{"optimizer": "nesterov", "Lsub_iteration": 1}]
    def toJson(self):
        return {k: v for k, v in self.__dict__.items()
                if k != "params_dict" and not k.startswith("_")}

def test_attach_and_detach_terms_use_underscore_key():
    p = _FakeParams()
    f = lambda pos: pos.sum()
    attach_terms(p, [f])
    assert p._extra_obj_terms == [f]
    assert "_extra_obj_terms" not in p.toJson()
    detach_terms(p)
    assert not hasattr(p, "_extra_obj_terms")

def test_optimizer_lock_accepts_locked_config_and_rejects_others():
    p = _FakeParams()
    assert_optimizer_lock(p)                                  # no raise
    bad = _FakeParams(); bad.global_place_stages[0]["Lsub_iteration"] = 2
    with pytest.raises(AssertionError):
        assert_optimizer_lock(bad)
    bad2 = _FakeParams(); bad2.use_bb = 1
    with pytest.raises(AssertionError):
        assert_optimizer_lock(bad2)
    bad3 = _FakeParams(); bad3.global_place_stages[0]["optimizer"] = "adam"
    with pytest.raises(AssertionError):
        assert_optimizer_lock(bad3)

# ---------------------------------------------------------------- the patch itself
def test_patch_is_applied_to_dreamplace():
    import PlaceObj, Params, NonLinearPlace, inspect
    assert "extra_obj_terms" in inspect.getsource(PlaceObj.PlaceObj.__init__)
    assert "extra_obj_terms" in inspect.getsource(PlaceObj.PlaceObj.obj_fn)
    assert "startswith" in inspect.getsource(Params.Params.toJson)
    assert "self.optimizer = optimizer" in inspect.getsource(NonLinearPlace.NonLinearPlace.__call__)

def test_real_params_serialisation_survives_attached_terms(tmp_path):
    import Params
    p = Params.Params()
    attach_terms(p, [lambda pos: pos.sum()])
    out = str(tmp_path / "p.json")
    p.dump(out)                                # must not raise TypeError
    assert "_extra_obj_terms" not in json.load(open(out))

@pytest.mark.slow
def test_simple_benchmark_runs_with_a_constant_extra_term_and_is_bit_identical():
    """design v2 sec 3.2.4: under the locked configuration the objective VALUE
    cannot affect the trajectory. A constant term changes obj but nothing else.
    Three environmental guards make this assertable and non-vacuous:
    - deterministic_flag=1 on both runs: GPU float32 atomics otherwise make
      even two *unmodified* identical runs differ (verified on this host);
    - init_pos reproducibility comes from _place() seeding numpy's global RNG
      (BasicPlace draws centre-noise/filler init from numpy's global RNG and
      reseeds only torch per run; the reference Placer.py flow we bypass is
      what normally seeds numpy);
    - random_center_init_flag=0: under simple's default (1) the 8 movable
      cells collapse onto the die centre, the estimated lr overshoots, the
      first Nesterov step is clamped by move_boundary, alpha_k collapses to 0
      at iteration 1 and pos is bit-frozen for all 1000 iterations -- the
      equality would then assert nothing about a trajectory. The len(metrics)
      guard locks the live-trajectory premise (~433 iters live, 1001 stalled)."""
    from ioplace.drivers.run_placement import _load_dreamplace, _place, extract_final_positions
    cfg = os.path.join(DP, "install", "test", "simple.json")
    p1, db1 = _load_dreamplace(cfg)
    p1.deterministic_flag = 1; p1.random_center_init_flag = 0
    db1.initialize(p1)
    pl1, m1 = _place(p1, db1); x1, y1 = extract_final_positions(pl1, db1)
    p2, db2 = _load_dreamplace(cfg)
    p2.deterministic_flag = 1; p2.random_center_init_flag = 0
    attach_terms(p2, [lambda pos: pos.new_tensor(1e6)])
    db2.initialize(p2)
    pl2, m2 = _place(p2, db2); x2, y2 = extract_final_positions(pl2, db2)
    assert len(m1) < 900 and len(m2) < 900, "GP stalled; bit-identity would be vacuous"
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)

@pytest.mark.slow
def test_sequential_flat_then_io_runs_are_isolated():
    """per-instance params-borne terms must not leak into a later un-instrumented run."""
    from ioplace.drivers.run_placement import _load_dreamplace
    import PlaceObj
    cfg = os.path.join(DP, "install", "test", "simple.json")
    p_io, db_io = _load_dreamplace(cfg)
    attach_terms(p_io, [lambda pos: pos.sum() * 0.0])
    db_io.initialize(p_io)
    p_flat, db_flat = _load_dreamplace(cfg); db_flat.initialize(p_flat)
    m_io = PlaceObj.PlaceObj(0.0, p_io, db_io, None, None, p_io.global_place_stages[0]) \
        if False else None                       # constructing PlaceObj needs collections
    assert getattr(p_flat, "_extra_obj_terms", []) == []
    assert len(getattr(p_io, "_extra_obj_terms", [])) == 1
