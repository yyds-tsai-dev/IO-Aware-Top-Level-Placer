import time
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from ioplace.profile import (PhaseTimer, EventTimer, DeviceMemSampler, host_rss_gb,
                             percentiles_ms, build_profile_record, PROFILE_SCHEMA_FIELDS)


def test_phase_peaks_do_not_cross_contaminate():
    """M4 design draft sec 1.4 B1's bug, reproduced at the phase level: a
    naive tracker that never resets would have phase "b" inherit phase "a"'s
    (much larger) high-water mark. PhaseTimer resets at every phase entry,
    so each phase's peak_alloc_gb reflects only what happened inside it."""
    timer = PhaseTimer()
    with timer.phase("a"):
        big = torch.empty(300_000_000, dtype=torch.uint8, device="cuda")  # ~300MB
        del big
        torch.cuda.empty_cache()
    with timer.phase("b"):
        small = torch.empty(20_000_000, dtype=torch.uint8, device="cuda")  # ~20MB
        del small

    assert timer.phases["a"]["peak_alloc_gb"] > timer.phases["b"]["peak_alloc_gb"]
    assert timer.phases["b"]["peak_alloc_gb"] < 0.05          # nowhere near 300MB
    assert timer.phases["a"]["peak_alloc_gb"] > 0.2
    for name in ("a", "b"):
        assert timer.phases[name]["t_s"] >= 0.0
        assert timer.phases[name]["host_rss_hwm_at_phase_end"] > 0.0


def test_host_rss_phase_field_is_a_monotonic_hwm_not_a_per_phase_peak():
    """`ru_maxrss` cannot be reset mid-process (no host analogue of
    reset_peak_memory_stats()), so `host_rss_hwm_at_phase_end` is the
    process's cumulative peak *as of* that phase's end -- unlike
    peak_alloc_gb, it must never decrease between phases even when a later
    phase itself allocates nothing on the host."""
    timer = PhaseTimer()
    with timer.phase("a"):
        big = bytearray(60_000_000)                            # ~60MB host alloc, raises ru_maxrss
    with timer.phase("b"):
        pass                                                   # no host allocation at all
    del big
    assert (timer.phases["b"]["host_rss_hwm_at_phase_end"]
           >= timer.phases["a"]["host_rss_hwm_at_phase_end"])


def test_sampler_thread_does_not_change_computed_result():
    """Decisiveness check (M4 design draft sec 6.1): the mem_get_info()
    background sampler is purely observational -- running it alongside a
    computation must not perturb that computation's result."""
    def compute():
        g = torch.Generator(device="cuda").manual_seed(1234)
        x = torch.randn(2_000_000, generator=g, device="cuda")
        for _ in range(20):
            x = torch.sin(x) + x.mean()
        return x.detach().clone()

    without = compute()

    sampler = DeviceMemSampler(interval_s=0.01)
    sampler.start()
    try:
        with_sampler = compute()
    finally:
        sampler.stop()

    assert torch.equal(without, with_sampler)
    assert sampler.device_used_gb > 0.0


def test_sampler_start_stop_is_idempotent_and_bounded():
    sampler = DeviceMemSampler(interval_s=0.05)
    sampler.start()
    time.sleep(0.15)
    sampler.stop()
    used1 = sampler.device_used_gb
    assert used1 > 0.0
    # stopping again must not raise or resume sampling
    sampler.stop()
    assert sampler.device_used_gb == used1


def test_sampler_device_used_is_whole_gpu_not_just_this_process():
    """sec 6.1 caveat: total-free includes every process on the device, so
    it is always >= what this process alone has allocated (CUDA context
    overhead alone guarantees a strict '>' in practice, but '>=' is the
    only thing the semantics actually promise)."""
    sampler = DeviceMemSampler(interval_s=0.5)
    sampler.start()
    x = torch.empty(50_000_000, dtype=torch.uint8, device="cuda")  # ~50MB
    this_process_gb = torch.cuda.memory_allocated() / 2**30
    sampler.stop()
    del x
    assert sampler.device_used_gb >= this_process_gb


def test_event_timer_pairs_and_batches_queries():
    timer = EventTimer()
    for _ in range(3):
        timer.start()
        x = torch.randn(1_000_000, device="cuda")
        (x * 2).sum()
        timer.stop()
    ms = timer.elapsed_ms()
    assert len(ms) == 3
    assert all(v >= 0.0 for v in ms)


def test_event_timer_rejects_unpaired_stop():
    timer = EventTimer()
    timer.start()
    timer.stop()
    with pytest.raises(RuntimeError):
        timer.stop()                      # stop() without a preceding start()

def test_event_timer_rejects_elapsed_ms_with_pending_start():
    timer = EventTimer()
    timer.start()                         # never stopped
    with pytest.raises(RuntimeError):
        timer.elapsed_ms()


def test_percentiles_ms_known_values():
    p = percentiles_ms([10.0, 20.0, 30.0, 40.0, 50.0])
    assert p["p50"] == 30.0
    assert p["max"] == 50.0
    assert percentiles_ms([]) == {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}


def test_host_rss_gb_is_positive():
    assert host_rss_gb() > 0.0


def test_phase_timer_reset_peak_false_skips_reset_and_marks_disabled():
    """M4 T2b: `PhaseTimer(reset_peak=False)` -- for driver runs where a
    `profile_lifetime.LifetimeRecorder` owns GPU-peak accounting instead
    (see profile.py's PhaseTimer/_Phase docstrings). peak_alloc_gb/
    peak_reserved_gb are explicitly None (not 0.0 -- "not measured here",
    not "measured, found empty"), and peak_semantics names why."""
    timer = PhaseTimer(reset_peak=False)
    with timer.phase("a"):
        torch.empty(1_000_000, dtype=torch.uint8, device="cuda")
    phase = timer.phases["a"]
    assert phase["peak_alloc_gb"] is None
    assert phase["peak_reserved_gb"] is None
    assert phase["peak_semantics"] == "disabled_owned_by_lifetime_recorder"
    assert phase["t_s"] >= 0.0
    assert phase["host_rss_hwm_at_phase_end"] > 0.0


def test_phase_timer_reset_peak_true_is_unchanged_default():
    """reset_peak defaults to True -- pre-T2b behavior (no peak_semantics
    key, peak_alloc_gb is a real reading) is untouched."""
    timer = PhaseTimer()
    with timer.phase("a"):
        torch.empty(1_000_000, dtype=torch.uint8, device="cuda")
    phase = timer.phases["a"]
    assert phase["peak_alloc_gb"] is not None and phase["peak_alloc_gb"] >= 0.0
    assert "peak_semantics" not in phase


def test_device_mem_sampler_reset_hwm_zeroes_and_resamples():
    """M4 T2b (`profile_lifetime.LifetimeRecorder.phase_begin`): the
    sampler-thread analogue of reset_peak_memory_stats() -- a later
    phase's device_used_gb must not inherit an earlier phase's HWM."""
    sampler = DeviceMemSampler(interval_s=0.5)
    sampler.start()
    try:
        big = torch.empty(300_000_000, dtype=torch.uint8, device="cuda")  # ~300MB
        hwm_before_reset = sampler.device_used_gb
        del big
        torch.cuda.empty_cache()
        sampler.reset_hwm()
        hwm_after_reset = sampler.device_used_gb
        assert hwm_after_reset < hwm_before_reset
    finally:
        sampler.stop()


def test_phase_summary_skips_none_peak_alloc_gb():
    """M4 T2b: `run_placement._phase_summary`'s max() must skip phases
    with peak_alloc_gb=None (a PhaseTimer(reset_peak=False) phase) rather
    than treating None as a genuine 0.0 peak, which would silently drag
    peak_mem_mb down when a LifetimeRecorder is active."""
    from ioplace.drivers.run_placement import _phase_summary

    class FakeTimer:
        phases = {
            "read": {"t_s": 1.0, "peak_alloc_gb": None, "peak_reserved_gb": None,
                     "peak_semantics": "disabled_owned_by_lifetime_recorder",
                     "host_rss_hwm_at_phase_end": 1.0},
            "gp": {"t_s": 2.0, "peak_alloc_gb": 3.5, "peak_reserved_gb": 4.0,
                  "host_rss_hwm_at_phase_end": 2.0},
        }

    class FakeSampler:
        device_used_gb = 5.0

    summary = _phase_summary(FakeTimer(), FakeSampler())
    assert summary["peak_mem_mb"] == pytest.approx(3.5 * 1024.0)


def test_phase_summary_all_none_peaks_defaults_to_zero():
    from ioplace.drivers.run_placement import _phase_summary

    class FakeTimer:
        phases = {
            "read": {"t_s": 1.0, "peak_alloc_gb": None, "peak_reserved_gb": None,
                     "host_rss_hwm_at_phase_end": 1.0},
        }

    class FakeSampler:
        device_used_gb = 0.0

    summary = _phase_summary(FakeTimer(), FakeSampler())
    assert summary["peak_mem_mb"] == 0.0


def test_profile_schema_has_all_fields():
    timer = PhaseTimer()
    with timer.phase("read"):
        pass
    record = build_profile_record(
        case="toy", K=8, rtype="grid", arm="flat", phase_timer=timer,
        iter_ms=[1.0, 2.0, 3.0], op_fwd_ms=[0.5], op_bwd_ms=[0.7], eval_ms=[3.0],
        device_used_gb=1.23,
        scale_meta=dict(n_movable=10, n_physical=12, n_filler=2, n_nets=8,
                        n_pins=30, lattice=512, n_bins=64, k_chunk=1, n_active=8),
        env={"torch_version": torch.__version__},
    )
    for f in PROFILE_SCHEMA_FIELDS:
        assert f in record, f
    assert record["case"] == "toy" and record["K"] == 8
    assert record["n_pins"] == 30
    assert record["iter_ms_p50"] == pytest.approx(2.0)     # nearest-rank median of [1,2,3]
