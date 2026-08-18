# T1b measurement-validity protocol

Design draft: `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md`
sec 1.4 B1 / T1b row (§7.1). This is the short runbook for T1b's (b)/(c)
protocol rules -- the software they depend on is (a)
`ioplace/diagnostics/probes_m4/probe_t1b_arms.py`, (d)
`ioplace/diagnostics/probes_m4/gpu_exclusivity.py`, (e)
`ioplace/diagnostics/probes_m4/probe_t1b_sampler.py`.

## (b) formal ablation: one subprocess per arm

Sec 1.4 B1's M2 finding (`peak_mem_mb` contaminated across arms sharing a
process, `run_ablation_m2.py:86`'s pattern) is not fixed by
`reset_peak_memory_stats()` alone -- v2's adjudicated cause is retained
tensors surviving arm-to-arm, not just an unreset HWM counter (see
`probe_t1b_arms.py`'s module docstring for the full cause-vs-hypothesis
argument).

**Default rule for every M4 ablation/sweep driver from here on: one arm,
one subprocess.** This is the ONLY protocol with an unconditional isolation
guarantee -- a fresh process starts with nothing retained from any
previous arm, by construction, independent of whatever `probe_t1b_arms.py`
measures on any given host/PyTorch version.

**Conditional exemption:** a driver MAY run multiple arms in one process
IFF it can point at a `t1b_arms.json` run (or an equivalent run of its own,
same protocol) whose `judgment.teardown_within_2pct_tolerance` is `true`
for its own workload shape, AND the driver actually performs the same
`teardown_gc` between-arm hook that run exercised (explicit `del` of every
arm-owned object + `gc.collect()` + `torch.cuda.empty_cache()`) before
starting the next arm. `judgment.reset_only_fixes_baseline` being `true`
is NOT sufficient on its own to claim the exemption -- sec 1.4 B1's whole
point is that reset alone was already tried and is not the adjudicated
cause; if `probe_t1b_arms.py`'s own `cause` field comes back
`"cumulative_hwm"` on a real run, that finding should be re-argued
explicitly before anyone treats `reset_only` as a green light, not treated
as implicitly authorizing it via this runbook.

If a driver is not prepared to do the exemption's own teardown protocol
faithfully, use one-subprocess-per-arm. This is the same posture
`spike_30m.py`/`spike_30m_child.py` (T9) already take (workload isolated
into a child process, parent never touches CUDA) and `probe_t1b_arms.py`'s
own `fresh_subprocess`/`--single-arm` plumbing.

## (c) per-phase attribution needing true isolation: child process

`host_rss_hwm_at_phase_end` (`ioplace/profile.py`'s `host_rss_gb`) is a
**process-lifetime monotonic high-water mark** -- `ru_maxrss` has no
reset analogue, so a same-process reading after phase B can only be >=
what it was after phase A, even if phase B itself freed host memory. This
is fine for "the run's overall peak RSS" (sec 6.1's intended use) but is
NOT a valid per-phase host-RSS measurement on its own.

**Rule:** any M4 experiment that needs to attribute host RSS to ONE
specific phase in isolation (not "the run's peak as of that phase's end")
must run that phase in its own **child process** and read that child's own
`ru_maxrss` after it exits -- not infer it from a same-process delta
against a monotonic HWM sequence. `ioplace/bench/spike_30m.py`
(parent)/`spike_30m_child.py` (child) is the existing parent/child split
to follow: the parent never touches CUDA and only observes the child from
the outside (`nvidia-smi` polling + the child's own exit code/output file);
a phase-isolated host-RSS child process should follow the same shape --
launch, let it do exactly one phase's work, read its artifact/exit state,
never share Python state with the parent.

GPU-peak attribution does not need this -- `torch.cuda.reset_peak_memory_
stats()` (T1's per-phase reset, `ioplace/profile.py`'s `PhaseTimer`) DOES
correctly zero the counter (necessary), it just is not SUFFICIENT for
arm-to-arm isolation across multiple arms in one process (that's (b)'s
concern, above) -- within a single arm's own multi-phase run, per-phase
reset remains valid for GPU peaks specifically.

## (d)/(a)/(e) cross-references

- (d) exclusive-GPU preflight/postflight: `gpu_exclusivity.
  preflight_exclusivity`/`postflight_exclusivity` -- any run that fails
  either must record `experiment_status="contaminated"`, matching
  `spike_30m.py`'s existing state machine.
- (a) the four-arm A/B probe itself, and its `judgment.cause` field, are
  what this runbook's (b) exemption clause is gated on.
- (e) `probe_t1b_sampler.py`'s measured miss rate at 0.05s/0.1s/0.5s spike
  widths is the honest caveat on `DeviceMemSampler`'s `device_used_gb`:
  any M4 report citing that field should also cite the miss rate at the
  narrowest spike width relevant to the workload being profiled, not treat
  `device_used_gb` as an exact peak.
