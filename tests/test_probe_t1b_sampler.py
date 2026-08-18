"""M4 T1b (e) tests for `ioplace.diagnostics.probes_m4.probe_t1b_sampler` --
design draft `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md`
sec 6.1/1.4 T1b row (e).

CPU-only: `run_one_trial`/`run_sampler_probe` are exercised against a fake
sampler with a scripted `device_used_gb` sequence -- no real
`profile.DeviceMemSampler`, no CUDA allocation, no `time.sleep()` for an
actual spike duration. `_alloc_and_free_spike` (the real CUDA-only spike
source) is never called.
"""
import json
import os

import pytest

from ioplace.diagnostics.probes_m4 import probe_t1b_sampler as ts


class _FakeSampler:
    """Scripted stand-in for `profile.DeviceMemSampler`: `reset_hwm()`
    zeroes a running max over a caller-provided `readings` iterator;
    `device_used_gb` returns that running max. `_sample_once()` (optional
    on the real sampler's protocol) advances the running max by consuming
    one more scripted reading, mirroring a background-thread sample."""

    def __init__(self, readings, interval_s=0.5):
        self.interval_s = interval_s
        self._readings = iter(readings)
        self._hwm = 0.0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True
        return self

    def reset_hwm(self):
        self._hwm = next(self._readings, self._hwm)

    def _sample_once(self):
        self._hwm = max(self._hwm, next(self._readings, self._hwm))

    @property
    def device_used_gb(self):
        return self._hwm


# ---------------------------------------------------------------------------
# run_one_trial
# ---------------------------------------------------------------------------

def test_run_one_trial_detects_spike_caught_by_sample_once():
    # reset_hwm() reads baseline 0.1; spike_fn is a no-op (fake doesn't
    # need real timing); _sample_once() catches the post-spike bump to 0.35.
    sampler = _FakeSampler(readings=[0.1, 0.35])
    result = ts.run_one_trial(sampler, spike_fn=lambda: None, size_gb=0.25)
    assert result["baseline_before_gb"] == pytest.approx(0.1)
    assert result["peak_after_gb"] == pytest.approx(0.35)
    assert result["delta_gb"] == pytest.approx(0.25)
    assert result["detected"] is True


def test_run_one_trial_misses_spike_never_sampled():
    # baseline 0.1, and the "catch" sample never advances past baseline --
    # the spike started and ended entirely between two samples.
    sampler = _FakeSampler(readings=[0.1, 0.1])
    result = ts.run_one_trial(sampler, spike_fn=lambda: None, size_gb=0.25)
    assert result["delta_gb"] == pytest.approx(0.0)
    assert result["detected"] is False


def test_run_one_trial_partial_catch_below_detect_frac_is_missed():
    # only 30% of the intended 0.25 GB bump was caught -- below the
    # default 50% detect_frac threshold.
    sampler = _FakeSampler(readings=[0.1, 0.175])
    result = ts.run_one_trial(sampler, spike_fn=lambda: None, size_gb=0.25)
    assert result["detected"] is False


def test_run_one_trial_custom_detect_frac():
    sampler = _FakeSampler(readings=[0.1, 0.175])
    result = ts.run_one_trial(sampler, spike_fn=lambda: None, size_gb=0.25, detect_frac=0.2)
    assert result["detected"] is True


def test_run_one_trial_works_without_sample_once(monkeypatch):
    """Sampler protocol only strictly requires reset_hwm()/device_used_gb
    -- _sample_once() is optional (real DeviceMemSampler has it; a caller
    without it should still get a (possibly less sensitive) reading, not
    an AttributeError."""
    class _MinimalSampler:
        def __init__(self):
            self._hwm = 0.3

        def reset_hwm(self):
            self._hwm = 0.1

        @property
        def device_used_gb(self):
            return self._hwm

    sampler = _MinimalSampler()
    result = ts.run_one_trial(sampler, spike_fn=lambda: None, size_gb=0.25)
    assert result["baseline_before_gb"] == pytest.approx(0.1)
    assert result["detected"] is False   # no _sample_once -> reads back 0.1, no bump caught


# ---------------------------------------------------------------------------
# run_sampler_probe
# ---------------------------------------------------------------------------

def test_run_sampler_probe_aggregates_detection_rate_over_repeats():
    """Injects a fake sampler class + a fake spike_fn_factory so the whole
    probe runs deterministically on CPU: every trial detects."""
    def fake_sampler_cls(interval_s=0.5):
        # every reset_hwm()/[implicit _sample_once via getattr check] call
        # cycles 0.1 (baseline) then 0.4 (bump) forever.
        import itertools
        return _FakeSampler(readings=itertools.cycle([0.1, 0.4]))

    def fake_spike_fn_factory(size_gb, spike_s):
        return lambda: None   # no real timing/allocation

    results = ts.run_sampler_probe(
        spike_durations=(0.05, 0.5), n_repeats=4, size_gb=0.25,
        sampler_cls=fake_sampler_cls, spike_fn_factory=fake_spike_fn_factory)

    assert set(results.keys()) == {"0.05", "0.5"}
    for spike_s_key in ("0.05", "0.5"):
        r = results[spike_s_key]
        assert r["n_repeats"] == 4
        assert r["n_detected"] == 4
        assert r["detection_rate"] == pytest.approx(1.0)
        assert r["miss_rate"] == pytest.approx(0.0)
        assert len(r["trials"]) == 4


def test_run_sampler_probe_mixed_detection_rate():
    """Alternates detected/missed trials deterministically: baseline 0.1
    every reset, then _sample_once alternates between a real bump (0.4,
    detected) and no bump (0.1, missed)."""
    class _AlternatingSampler:
        def __init__(self, interval_s=0.5):
            self._hwm = 0.0
            self._next_bump = True

        def reset_hwm(self):
            self._hwm = 0.1

        def _sample_once(self):
            bump = 0.4 if self._next_bump else 0.1
            self._next_bump = not self._next_bump
            self._hwm = max(self._hwm, bump)

        @property
        def device_used_gb(self):
            return self._hwm

        def start(self):
            return self

        def stop(self):
            return self

    results = ts.run_sampler_probe(
        spike_durations=(0.1,), n_repeats=4, size_gb=0.25,
        sampler_cls=_AlternatingSampler, spike_fn_factory=lambda sz, dur: (lambda: None))
    r = results["0.1"]
    assert r["n_detected"] == 2
    assert r["detection_rate"] == pytest.approx(0.5)
    assert r["miss_rate"] == pytest.approx(0.5)


def test_run_sampler_probe_calls_start_and_stop_on_sampler():
    calls = []

    class _TrackingSampler(_FakeSampler):
        def start(self):
            calls.append("start")
            return super().start()

        def stop(self):
            calls.append("stop")
            return super().stop()

    import itertools
    ts.run_sampler_probe(
        spike_durations=(0.05,), n_repeats=2, size_gb=0.1,
        sampler_cls=lambda interval_s=0.5: _TrackingSampler(itertools.cycle([0.0, 0.2])),
        spike_fn_factory=lambda sz, dur: (lambda: None))
    assert calls == ["start", "stop"]


def test_run_sampler_probe_stops_sampler_even_if_trial_raises():
    stop_calls = []

    class _RaisingSampler(_FakeSampler):
        def reset_hwm(self):
            raise RuntimeError("boom")

        def stop(self):
            stop_calls.append(True)
            return super().stop()

    with pytest.raises(RuntimeError):
        ts.run_sampler_probe(
            spike_durations=(0.05,), n_repeats=1, size_gb=0.1,
            sampler_cls=lambda interval_s=0.5: _RaisingSampler([]),
            spike_fn_factory=lambda sz, dur: (lambda: None))
    assert stop_calls == [True]


# ---------------------------------------------------------------------------
# run() / artifact writing
# ---------------------------------------------------------------------------

def test_run_writes_result_gate_shaped_json(tmp_path, monkeypatch):
    def fake_run_sampler_probe(spike_durations, n_repeats, size_gb, interval_s):
        return {
            str(d): {"spike_s": d, "n_repeats": n_repeats, "n_detected": n_repeats,
                     "detection_rate": 1.0, "miss_rate": 0.0, "trials": []}
            for d in spike_durations
        }

    monkeypatch.setattr(ts, "run_sampler_probe", fake_run_sampler_probe)
    out_json = str(tmp_path / "t1b_sampler.json")
    record = ts.run(interval_s=0.5, size_gb=0.1, n_repeats=3,
                    spike_durations=(0.05, 0.1, 0.5), out_json=out_json)

    assert record["record_kind"] == "t1b_sampler"
    assert record["spike_durations_s"] == [0.05, 0.1, 0.5]
    assert set(record["results"].keys()) == {"0.05", "0.1", "0.5"}
    assert os.path.exists(out_json)
    with open(out_json) as f:
        on_disk = json.load(f)
    assert on_disk["record_kind"] == "t1b_sampler"


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = ts._parse_args([])
    assert args.interval_s == 0.5
    assert args.size_gb == ts.SPIKE_SIZE_GB
    assert args.n_repeats == ts.N_REPEATS
    assert tuple(args.spike_durations_s) == ts.SPIKE_DURATIONS_S


def test_parse_args_overrides():
    args = ts._parse_args(["--interval-s", "0.25", "--size-gb", "0.5",
                          "--n-repeats", "5", "--spike-durations-s", "0.05", "0.2"])
    assert args.interval_s == 0.25
    assert args.size_gb == 0.5
    assert args.n_repeats == 5
    assert args.spike_durations_s == [0.05, 0.2]
