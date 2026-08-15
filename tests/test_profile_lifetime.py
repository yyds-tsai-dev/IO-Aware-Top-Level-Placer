"""M4 T2b (`ioplace/profile_lifetime.py`). `scan_cuda_tensors`'s own tests
use plain CPU-only fake tensor objects (design draft `docs/superpowers/
specs/2026-08-13-m4-scale-up-design-draft.md` sec 7.1 T2b row: "可用 CPU
tensor mock" -- `_looks_like_tensor` is a duck-type check, not `isinstance
(obj, torch.Tensor)`, precisely so these tests don't need a CUDA device).
`LifetimeRecorder`'s bookkeeping (`mark`/`to_record`) is tested the same
way, bypassing `__init__` (which does real `torch.cuda.*` calls) via
`object.__new__` + directly populated attributes / a monkeypatched
`_live_scan` -- `mark`/`to_record` themselves never call `torch.cuda.*`.

`LifetimeRecorder.phase_begin`/`phase_end` (which genuinely do call
`torch.cuda.reset_peak_memory_stats`/`synchronize`/etc.) and any test that
drives a real `run_placement_io.run_io` placement are out of scope here --
the former needs a live CUDA device and the latter is a real DREAMPlace
placement (this repo's own convention, see `tests/test_driver_io.py`,
marks *any* `run_io`/`run_flat` call `@pytest.mark.slow`, even on the
tiny `simple.json` toy config)."""
import pytest
import torch

from ioplace.profile_lifetime import LifetimeRecorder, scan_cuda_tensors

_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class FakeStorage:
    def __init__(self, ptr, nbytes):
        self._ptr = ptr
        self._nbytes = nbytes

    def data_ptr(self):
        return self._ptr

    def nbytes(self):
        return self._nbytes


class FakeTensor:
    """Duck-typed CUDA-tensor stand-in: has exactly the attributes
    `scan_cuda_tensors` looks for (`is_cuda`, `dtype`, `shape`,
    `untyped_storage()`), nothing torch-specific."""

    def __init__(self, ptr, nbytes, *, is_cuda=True, dtype="torch.float32", shape=(4,)):
        self.is_cuda = is_cuda
        self.dtype = dtype
        self.shape = shape
        self._storage = FakeStorage(ptr, nbytes)

    def untyped_storage(self):
        return self._storage


class Holder:
    """Plain object with a __dict__ -- the generic "walk vars(obj)" case."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# scan_cuda_tensors
# ---------------------------------------------------------------------------

def test_scan_cuda_tensors_finds_a_direct_attribute():
    root = Holder(t=FakeTensor(0x1000, 4096))
    found = scan_cuda_tensors(root, "root")
    assert len(found) == 1
    f = found[0]
    assert f["name"] == "root.t"
    assert f["root"] == "root"
    assert f["bytes"] == 4096
    assert f["storage_ptr"] == 0x1000
    assert f["dtype"] == "torch.float32"
    assert f["shape"] == (4,)


def test_scan_cuda_tensors_skips_non_cuda():
    root = Holder(t=FakeTensor(0x1000, 4096, is_cuda=False))
    assert scan_cuda_tensors(root, "root") == []


def test_scan_cuda_tensors_dedups_by_storage_not_object_identity():
    """Two distinct tensor *objects* viewing the same (data_ptr, nbytes)
    storage (a detach()/view/second-Parameter-style alias) count once."""
    shared_ptr, shared_bytes = 0x2000, 8192
    root = Holder(a=FakeTensor(shared_ptr, shared_bytes),
                  b=FakeTensor(shared_ptr, shared_bytes))
    found = scan_cuda_tensors(root, "root")
    assert len(found) == 1


def test_scan_cuda_tensors_first_reached_name_is_deterministic_under_key_sort():
    """Traversal visits dict/attr keys in sorted order, so of the two paths
    to the same storage ("a" vs "z"), the alphabetically-first one always
    wins -- run it twice to make sure there's no hash-seed/insertion-order
    dependence."""
    shared_ptr, shared_bytes = 0x3000, 16384

    def build():
        return Holder(z=FakeTensor(shared_ptr, shared_bytes),
                      a=FakeTensor(shared_ptr, shared_bytes))

    for _ in range(3):
        found = scan_cuda_tensors(build(), "root")
        assert len(found) == 1
        assert found[0]["name"] == "root.a"


def test_scan_cuda_tensors_walks_dict_list_tuple_values():
    t1, t2, t3 = FakeTensor(0x10, 1), FakeTensor(0x20, 2), FakeTensor(0x30, 3)
    root = Holder(d={"x": t1}, lst=[t2], tup=(t3,))
    found = {f["storage_ptr"]: f for f in scan_cuda_tensors(root, "root")}
    assert set(found) == {0x10, 0x20, 0x30}
    assert found[0x10]["name"] == "root.d.x"
    assert found[0x20]["name"] == "root.lst[0]"
    assert found[0x30]["name"] == "root.tup[0]"


def test_scan_cuda_tensors_walks_module_style_containers():
    """nn.Module-style `_parameters`/`_buffers`/`_modules` dicts, whether or
    not they're also reachable via a generic __dict__ walk (real nn.Module
    instances have both -- the literal attrs and the __dict__ entries are
    the *same* dict objects, so no double-count either way)."""
    # scan_cuda_tensors walks the dict's own keys, not `.<attr>.<key>` --
    # so "w" (the parameter name) ends up directly under the module's
    # path, mirroring how nn.Module.__getattr__ resolves self.w through
    # self._parameters transparently.
    t = FakeTensor(0x40, 4)
    root = Holder(_parameters={"w": t}, _buffers={}, _modules={})
    found = scan_cuda_tensors(root, "root")
    assert len(found) == 1
    assert found[0]["name"] == "root.w"


def test_scan_cuda_tensors_is_cycle_safe():
    a = Holder()
    b = Holder(back=a)
    a.fwd = b
    a.t = FakeTensor(0x50, 8)
    found = scan_cuda_tensors(a, "root")     # must terminate, not infinite-loop
    assert len(found) == 1


def test_scan_cuda_tensors_max_depth_bounds_container_descent():
    # Chain of Holders past max_depth; the tensor at the far end must not
    # be found once its container is beyond max_depth, but a tensor found
    # *before* the cutoff still is.
    near = Holder(t=FakeTensor(0x60, 1))
    far_leaf = Holder(t=FakeTensor(0x70, 1))
    cur = far_leaf
    for _ in range(10):
        cur = Holder(next=cur)
    root = Holder(near=near, chain=cur)
    found = {f["storage_ptr"] for f in scan_cuda_tensors(root, "root", max_depth=2)}
    assert 0x60 in found
    assert 0x70 not in found


def test_scan_cuda_tensors_node_budget_truncates():
    root = Holder(**{f"t{i}": FakeTensor(0x1000 + i, 1) for i in range(50)})
    found_full = scan_cuda_tensors(root, "root", node_budget=200_000)
    found_tiny = scan_cuda_tensors(root, "root", node_budget=3)
    assert len(found_full) == 50
    assert len(found_tiny) < 50                # truncated, not an error


# ---------------------------------------------------------------------------
# LifetimeRecorder -- pure-Python bookkeeping (mark/to_record), no CUDA.
# ---------------------------------------------------------------------------

def _bare_recorder():
    """Bypasses __init__ (real torch.cuda.* calls, a live DeviceMemSampler
    thread) -- mark()/to_record() only touch self._roots/_buffers/
    _checkpoints/_phases/scan_iters, all set here by hand."""
    rec = object.__new__(LifetimeRecorder)
    rec._roots = {}
    rec._checkpoints = []
    rec._buffers = {}
    rec._phases = []
    rec._open_phase = None
    rec.scan_iters = frozenset((0, 1, 2, -1))
    rec.device_baseline_gb = 0.1
    return rec


def test_wants_iter():
    rec = _bare_recorder()
    assert rec.wants_iter(0) and rec.wants_iter(1) and rec.wants_iter(2)
    assert not rec.wants_iter(3)
    assert rec.wants_iter(999, is_last=True)     # -1 sentinel via is_last
    assert not rec.wants_iter(999, is_last=False)


def _scan_result(*items):
    """items: (storage_ptr, nbytes, name) -> the dict shape _live_scan
    returns (same shape scan_cuda_tensors produces)."""
    return {(ptr, nb): {"name": name, "root": "r", "dtype": "torch.float32",
                        "shape": (nb,), "bytes": nb, "storage_ptr": ptr}
           for ptr, nb, name in items}


def test_mark_records_create_checkpoint_on_first_appearance():
    rec = _bare_recorder()
    rec._live_scan = lambda: _scan_result((1, 100, "r.a"))
    rec.mark("cp0")
    assert rec._buffers[(1, 100)]["create_checkpoint"] == "cp0"
    assert rec._buffers[(1, 100)]["destroy_checkpoint"] is None
    assert rec._checkpoints == ["cp0"]


def test_mark_does_not_overwrite_create_checkpoint_on_later_sightings():
    rec = _bare_recorder()
    rec._live_scan = lambda: _scan_result((1, 100, "r.a"))
    rec.mark("cp0")
    rec.mark("cp1")
    assert rec._buffers[(1, 100)]["create_checkpoint"] == "cp0"


def test_mark_sets_destroy_checkpoint_at_first_absence():
    rec = _bare_recorder()
    live = {"cp0": _scan_result((1, 100, "r.a")), "cp1": _scan_result()}
    calls = iter(["cp0", "cp1"])
    rec._live_scan = lambda: live[next(calls)]
    rec.mark("cp0")
    rec.mark("cp1")
    assert rec._buffers[(1, 100)]["destroy_checkpoint"] == "cp1"


def test_mark_leaves_destroy_checkpoint_none_if_still_live_at_end():
    rec = _bare_recorder()
    rec._live_scan = lambda: _scan_result((1, 100, "r.a"))
    rec.mark("cp0")
    rec.mark("cp1")           # still live both times
    assert rec._buffers[(1, 100)]["destroy_checkpoint"] is None


def test_mark_scan_false_records_checkpoint_without_scanning():
    rec = _bare_recorder()
    rec._live_scan = lambda: (_ for _ in ()).throw(AssertionError("must not scan"))
    rec.mark("cp0", scan=False)
    assert rec._checkpoints == ["cp0"]
    assert rec._buffers == {}


def test_add_root_registers_object():
    rec = _bare_recorder()
    obj = object()
    rec.add_root("eval_ctx", obj)
    assert rec._roots["eval_ctx"] is obj


def _phase(name, *, resident_gb, alloc_start_gb, transient_gb, peak_alloc_gb,
          peak_reserved_gb, device_used_peak_gb):
    return {"name": name, "t_s": 1.0, "resident_gb": resident_gb,
           "alloc_start_gb": alloc_start_gb,
           "unattributed_gb": alloc_start_gb - resident_gb,
           "transient_gb": transient_gb, "peak_alloc_gb": peak_alloc_gb,
           "peak_reserved_gb": peak_reserved_gb,
           "device_used_peak_gb": device_used_peak_gb,
           "host_rss_hwm_at_phase_end": 2.0}


def test_to_record_reconstructs_run_wide_peaks_as_max_over_phases():
    rec = _bare_recorder()
    rec._phases = [
        _phase("read", resident_gb=0.0, alloc_start_gb=0.0, transient_gb=1.0,
              peak_alloc_gb=1.0, peak_reserved_gb=1.5, device_used_peak_gb=1.2),
        _phase("gp", resident_gb=1.0, alloc_start_gb=1.0, transient_gb=3.0,
              peak_alloc_gb=4.0, peak_reserved_gb=4.5, device_used_peak_gb=4.2),
        _phase("eval", resident_gb=1.0, alloc_start_gb=1.0, transient_gb=0.5,
              peak_alloc_gb=1.5, peak_reserved_gb=2.0, device_used_peak_gb=1.6),
    ]
    rec._buffers = {
        (1, 100): {"name": "a", "root": "r", "dtype": "f32", "shape": (100,),
                   "bytes": 100, "storage_ptr": 1,
                   "create_checkpoint": "cp0", "destroy_checkpoint": None},
        (2, 200): {"name": "b", "root": "r", "dtype": "f32", "shape": (200,),
                   "bytes": 200, "storage_ptr": 2,
                   "create_checkpoint": "cp0", "destroy_checkpoint": "cp1"},
    }
    record = rec.to_record()
    totals = record["totals"]
    assert totals["peak_alloc_gb_run"] == pytest.approx(4.0)     # max over phases
    assert totals["peak_reserved_gb_run"] == pytest.approx(4.5)
    assert totals["device_used_gb_run"] == pytest.approx(4.2)
    assert totals["resident_max_gb"] == pytest.approx(1.0)
    assert totals["max_phase_transient_gb"] == pytest.approx(3.0)
    # sec 2.2's corrected formula: peak ~= resident + max_transient.
    assert totals["identity_residual_gb"] == pytest.approx(4.0 - (1.0 + 3.0))
    assert totals["standalone_upper_bound_gb"] == pytest.approx(300 / 2**30)
    assert totals["upper_bound_over_measured"] == pytest.approx(
        totals["standalone_upper_bound_gb"] / 4.0)
    assert len(record["buffers"]) == 2
    by_name = {b["name"]: b for b in record["buffers"]}
    assert by_name["a"]["create_phase"] == "cp0" and by_name["a"]["destroy_phase"] is None
    assert by_name["b"]["destroy_phase"] == "cp1"


def test_to_record_empty_recorder_does_not_raise():
    rec = _bare_recorder()
    record = rec.to_record()
    assert record["buffers"] == [] and record["phases"] == []
    assert record["totals"]["peak_alloc_gb_run"] == 0.0
    assert record["totals"]["upper_bound_over_measured"] is None   # no run-wide peak to divide by


def test_phase_begin_rejects_nested_open_phase():
    rec = _bare_recorder()
    rec._open_phase = {"name": "gp"}
    with pytest.raises(AssertionError):
        rec.phase_begin("eval")


def test_phase_end_rejects_mismatched_name():
    rec = _bare_recorder()
    rec._open_phase = {"name": "gp"}
    with pytest.raises(AssertionError):
        rec.phase_end("eval")


# ---------------------------------------------------------------------------
# LifetimeRecorder end-to-end -- real (small) CUDA tensors. Not a placement
# (no run_placement_io.run_io call -- see module docstring): same "small
# unit-level CUDA op" tier as tests/test_profile.py's own PhaseTimer/
# DeviceMemSampler tests, not this repo's @pytest.mark.slow tier.
# ---------------------------------------------------------------------------

class RootObj:
    def __init__(self):
        self.buf = None


@_needs_cuda
def test_lifetime_recorder_phase_resident_and_transient_gb():
    rec = LifetimeRecorder(sample_interval_s=0.05)
    try:
        root = RootObj()
        rec.add_root("root", root)

        rec.phase_begin("p0")
        root.buf = torch.empty(50_000_000, dtype=torch.uint8, device="cuda")  # ~50MB resident
        p0 = rec.phase_end("p0")
        assert p0["resident_gb"] < 0.01           # scanned BEFORE root.buf was assigned
        assert p0["transient_gb"] > 0.01           # the ~50MB shows up as this phase's own delta

        rec.phase_begin("p1")
        transient = torch.empty(20_000_000, dtype=torch.uint8, device="cuda")  # ~20MB extra
        del transient
        p1 = rec.phase_end("p1")
        assert p1["resident_gb"] > 0.01            # root.buf now counted as resident baseline
        assert p1["alloc_start_gb"] >= p1["resident_gb"]
    finally:
        rec.stop()


@_needs_cuda
def test_lifetime_recorder_mark_tracks_create_and_destroy_with_real_tensors():
    rec = LifetimeRecorder(sample_interval_s=0.05)
    try:
        root = RootObj()
        rec.add_root("root", root)
        rec.mark("before")                          # nothing live yet
        root.buf = torch.empty(1_000_000, dtype=torch.uint8, device="cuda")
        rec.mark("after_alloc")
        root.buf = None
        rec.mark("after_free")

        record = rec.to_record()
        assert len(record["buffers"]) == 1
        b = record["buffers"][0]
        assert b["create_phase"] == "after_alloc"
        assert b["destroy_phase"] == "after_free"
        assert b["root"] == "root"
    finally:
        rec.stop()


@_needs_cuda
def test_lifetime_recorder_to_record_totals_are_populated():
    rec = LifetimeRecorder(sample_interval_s=0.05)
    try:
        root = RootObj()
        rec.add_root("root", root)
        rec.phase_begin("only_phase")
        root.buf = torch.empty(10_000_000, dtype=torch.uint8, device="cuda")
        rec.phase_end("only_phase")
        record = rec.to_record()
        totals = record["totals"]
        assert totals["peak_alloc_gb_run"] > 0.0
        assert totals["upper_bound_over_measured"] is not None
    finally:
        rec.stop()
