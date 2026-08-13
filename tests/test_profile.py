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
