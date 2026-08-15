"""M4 T2b integrated memory gate (design draft `docs/superpowers/specs/
2026-08-13-m4-scale-up-design-draft.md` sec 2.2/7.1 T2b): full-lifetime
buffer tracking + per-phase resident/transient/device-used accounting for
`run_placement_io.run_io`'s own driver process.

sec 2.2's corrected accounting replaces v1's "sum of standalone peaks"
(neither a feasibility proof nor a disproof -- Codex #2) with `peak_device
~= resident(t) + max_p transient_p`, where `resident(t)` is every
still-alive static buffer at time t and `transient_p` is phase p's own
extra allocation above that baseline. This module supplies the two halves
of that formula:

  - `scan_cuda_tensors`: walks an arbitrary Python object graph (a
    `GpuEvalContext`, an `IoTerm`, a `NonLinearPlace` instance, ...) and
    returns every distinct CUDA-tensor storage reachable from it --
    `resident(t)` is this, unioned over every object the driver has
    registered as a "root" and still holds a reference to.
  - `LifetimeRecorder`: drives `scan_cuda_tensors` at named checkpoints
    (`mark()`, for buffer create/destroy bookkeeping) and named phase
    spans (`phase_begin`/`phase_end`, for the resident/transient/
    device-used-peak/host-RSS accounting), and assembles both into the
    `results/m4/profile/lifetime_*.json` record (`to_record()`).

Every call site in `run_placement_io.run_io` is guarded by `if rec is not
None:` -- when `lifetime_out` is not requested (the default, and every
existing caller/test), this module is never touched and the driver's
control flow, tensor ops, and RNG draws are completely unchanged (the
bit-exactness guarantee sec 4.2 already applies to io_count/ft_count/hpwl/
tree_wl extended here to "lifetime_out on vs off").
"""
import time

import torch


def _looks_like_tensor(obj):
    """Duck-typed tensor check, not `isinstance(obj, torch.Tensor)` --
    lets pytest exercise `scan_cuda_tensors`'s traversal/dedup logic with
    plain CPU-only fake objects (see `tests/test_profile_lifetime.py`)
    instead of requiring a live CUDA device to test determinism and
    dedup."""
    return (hasattr(obj, "is_cuda") and hasattr(obj, "untyped_storage")
            and hasattr(obj, "dtype") and hasattr(obj, "shape"))


_PRIMITIVE_TYPES = (int, float, str, bytes, bool, type(None))


def scan_cuda_tensors(root, prefix, *, max_depth=6, node_budget=200_000):
    """Deterministic walk of `root`'s object graph (`__dict__` /
    `_buffers` / `_parameters` / `_modules` / list|tuple|dict values),
    collecting every distinct CUDA-tensor storage reachable from it.

    - Cycle-safe: an `id()` set stops re-descending into an already-visited
      object (handles reference cycles and diamond-shaped sharing without
      infinite recursion).
    - Dedup key is `(storage.data_ptr(), storage.nbytes())`, not tensor
      identity or `id()` -- two different Python tensor *objects* that
      view the same underlying allocation (a `.detach()`, a view, a second
      `nn.Parameter` wrapping the same storage) are the same resident byte
      range and must only be counted once.
    - `name` is the traversal path at which a given storage is *first*
      reached, under a **key-sorted** traversal order -- sorting container
      keys (dict keys, `_parameters`/`_buffers`/`_modules` keys) before
      descending makes "first reached" deterministic across runs/Python
      hash-seed variation, rather than depending on dict insertion order
      or set iteration order.
    - `max_depth` bounds descent into non-tensor containers (a tensor
      itself is always recorded regardless of depth -- the depth check
      only gates whether *its container* is explored further).
    - `node_budget` bounds the total number of graph nodes visited
      (tensor or container) -- a hard stop against a pathologically large
      or self-referential object graph; traversal beyond the budget is
      silently truncated (no exception), since this is a diagnostic scan,
      not a correctness-critical path.

    Returns `list[{"name", "root", "dtype", "shape", "bytes",
    "storage_ptr"}]`, one entry per distinct storage. `bytes` is
    `storage.untyped_storage().nbytes()` (the actual allocation size, not
    `numel() * element_size()`, which can undercount for a storage shared
    by overlapping/strided views).
    """
    found = {}          # (data_ptr, nbytes) -> record dict, first-reached wins
    seen_ids = set()
    budget = [node_budget]

    def visit_tensor(t, name):
        if not getattr(t, "is_cuda", False):
            return
        try:
            storage = t.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
        except Exception:
            return
        if key in found:
            return
        found[key] = {
            "name": name, "root": prefix,
            "dtype": str(t.dtype), "shape": tuple(t.shape),
            "bytes": int(storage.nbytes()), "storage_ptr": int(key[0]),
        }

    def walk(obj, name, depth):
        if budget[0] <= 0:
            return
        budget[0] -= 1

        if _looks_like_tensor(obj):
            # Tensors are always recorded (even past max_depth) -- only
            # their *container*'s further exploration is depth-limited.
            oid = id(obj)
            if oid in seen_ids:
                return
            seen_ids.add(oid)
            visit_tensor(obj, name)
            return

        if isinstance(obj, _PRIMITIVE_TYPES) or depth > max_depth:
            return

        oid = id(obj)
        if oid in seen_ids:
            return
        seen_ids.add(oid)

        if isinstance(obj, dict):
            for k in sorted(obj.keys(), key=str):
                walk(obj[k], f"{name}.{k}", depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{name}[{i}]", depth + 1)
            return

        # Generic object: nn.Module-style containers first (explicit, in
        # case a subclass overrides __getattr__ in a way that hides them
        # from a plain __dict__ walk), then the rest of __dict__ (which,
        # for a real nn.Module, already *contains* _parameters/_buffers/
        # _modules as literal entries -- skipped below to avoid walking
        # the same dict object twice).
        walked_attrs = ("_parameters", "_buffers", "_modules")
        for attr in walked_attrs:
            d = getattr(obj, attr, None)
            if isinstance(d, dict):
                for k in sorted(d.keys(), key=str):
                    walk(d[k], f"{name}.{k}", depth + 1)

        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for k in sorted(d.keys(), key=str):
                if k in walked_attrs:
                    continue
                walk(d[k], f"{name}.{k}", depth + 1)

    walk(root, prefix, 0)
    return list(found.values())


def _alloc_gb():
    return (torch.cuda.memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0


def _mem_get_info_used_gb(device):
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 2**30


def _host_rss_gb():
    from ioplace.profile import host_rss_gb
    return host_rss_gb()


class LifetimeRecorder:
    """Drives `scan_cuda_tensors` over a set of registered "root" objects
    (`add_root`) at named checkpoints (`mark`) and named phase spans
    (`phase_begin`/`phase_end`), and assembles the T2b lifetime JSON
    record (`to_record`).

    Phases are non-nested (a single open span at a time) -- `phase_begin`
    asserts no phase is already open, `phase_end` asserts the name
    matches. Fine-grained tracking *within* a long-running phase (e.g. the
    GP loop) uses `mark()` checkpoints instead of nested phases.
    """

    def __init__(self, device=None, sample_interval_s=0.5, scan_iters=(0, 1, 2, -1)):
        from ioplace.profile import DeviceMemSampler
        self.device = device
        self.scan_iters = frozenset(scan_iters) if scan_iters is not None else frozenset()
        self._roots = {}
        self._checkpoints = []
        self._buffers = {}          # (data_ptr, nbytes) -> buffer dict
        self._phases = []
        self._open_phase = None
        self.sampler = DeviceMemSampler(interval_s=sample_interval_s, device=device)
        self.sampler.start()
        self.device_baseline_gb = _mem_get_info_used_gb(device)

    def add_root(self, name, obj):
        """Register `obj` under `name` -- every subsequent `mark()`/
        `phase_begin()` resident scan walks every currently-registered
        root. Roots are never unregistered (a buffer that becomes
        unreachable because its root itself is torn down is still caught
        by `mark()`'s "absent from the live scan" destroy detection, since
        the root's own `__dict__` no longer references it)."""
        self._roots[name] = obj

    def wants_iter(self, iteration, is_last=False):
        """Whether a GP-loop callback at `iteration` should call `mark()`
        -- true iff `iteration` is one of `scan_iters`, or `is_last` and
        `-1` (the "final iteration" sentinel) is in `scan_iters`."""
        return iteration in self.scan_iters or (is_last and -1 in self.scan_iters)

    def _live_scan(self):
        """One deterministic scan_cuda_tensors pass over every registered
        root, deduped across roots (a storage shared by two roots is one
        entry, keyed by the alphabetically-first root/name it's reached
        through -- same key-sorted-traversal determinism as
        `scan_cuda_tensors` itself)."""
        live = {}
        for name in sorted(self._roots):
            for t in scan_cuda_tensors(self._roots[name], name):
                key = (t["storage_ptr"], t["bytes"])
                live.setdefault(key, t)
        return live

    def _resident_gb(self):
        live = self._live_scan()
        return sum(t["bytes"] for t in live.values()) / 2**30

    def mark(self, checkpoint, *, scan=True):
        """Records `checkpoint` in the checkpoint order; if `scan` (the
        default), also runs a resident-buffer scan and updates every
        buffer's `create_checkpoint`/`destroy_checkpoint`:
          - `create_checkpoint` = the first checkpoint (in call order) at
            which a given (data_ptr, nbytes) key was seen live.
          - `destroy_checkpoint` = the first checkpoint at which a
            previously-seen key is no longer live. A buffer still live at
            the final `mark()` call keeps `destroy_checkpoint=None` ("survived
            to teardown").
        `scan=False` is for checkpoints where no roots are registered yet
        (or none have changed) -- records the checkpoint's position in the
        ordering without paying for a scan that would find nothing new.
        """
        self._checkpoints.append(checkpoint)
        if not scan:
            return
        live = self._live_scan()
        for key, t in live.items():
            if key not in self._buffers:
                self._buffers[key] = {
                    "name": t["name"], "root": t["root"], "dtype": t["dtype"],
                    "shape": list(t["shape"]), "bytes": t["bytes"],
                    "storage_ptr": t["storage_ptr"],
                    "create_checkpoint": checkpoint, "destroy_checkpoint": None,
                }
        for key, rec in self._buffers.items():
            if rec["destroy_checkpoint"] is None and key not in live:
                rec["destroy_checkpoint"] = checkpoint

    def phase_begin(self, name):
        """Opens phase `name`: resets the CUDA allocator's peak-tracking
        counters and the internal `DeviceMemSampler`'s high-water mark
        (sec 1.4 B1's per-phase-reset fix, applied here instead of by
        `PhaseTimer` -- the driver passes `PhaseTimer(reset_peak=False)`
        so the two trackers never fight over the same process-wide CUDA
        counter), then snapshots `alloc_start_gb` (unconditionally saved,
        never skipped) and `resident_gb` (this phase's baseline, from a
        fresh scan of every registered root)."""
        assert self._open_phase is None, (
            f"LifetimeRecorder.phase_begin({name!r}): phase "
            f"{self._open_phase['name']!r} is still open -- phases are not "
            "nested, use mark() for fine-grained checkpoints within a phase")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.sampler.reset_hwm()
        alloc_start_gb = _alloc_gb()
        resident_gb = self._resident_gb()
        self._open_phase = {
            "name": name, "t0": time.perf_counter(),
            "alloc_start_gb": alloc_start_gb, "resident_gb": resident_gb,
            "unattributed_gb": alloc_start_gb - resident_gb,
        }

    def phase_end(self, name):
        """Closes phase `name`, computing:
          - `transient_gb = peak_alloc_gb - alloc_start_gb` (the phase's
            *own* extra allocation above its entry baseline).
          - `peak_alloc_gb`/`peak_reserved_gb`: the CUDA allocator's
            reset-then-read high-water marks (absolute values, not
            deltas -- `max_memory_allocated()` after a reset already
            reflects the true absolute peak observed since that reset,
            including whatever was already resident at reset time).
          - `device_used_peak_gb`: this phase's `DeviceMemSampler`
            high-water mark, with `device_baseline_gb` subtracted (sec 1's
            "量測時點定義": device_used comparisons are always relative to
            the pre-M4-allocation baseline, not the raw whole-device
            reading).
          - `host_rss_hwm_at_phase_end`: same field name/semantics as
            `profile.PhaseTimer`'s (a process-lifetime monotonic HWM *as
            of* this phase's end, not this phase's own contribution --
            see `profile.host_rss_gb`'s docstring)."""
        assert self._open_phase is not None and self._open_phase["name"] == name, (
            f"LifetimeRecorder.phase_end({name!r}): no matching open phase "
            f"(open={self._open_phase!r})")
        op = self._open_phase
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc_gb = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved_gb = torch.cuda.max_memory_reserved() / 2**30
        else:
            peak_alloc_gb = 0.0
            peak_reserved_gb = 0.0
        device_used_peak_gb = max(0.0, self.sampler.device_used_gb - self.device_baseline_gb)
        phase = {
            "name": name, "t_s": time.perf_counter() - op["t0"],
            "resident_gb": op["resident_gb"],
            "alloc_start_gb": op["alloc_start_gb"],
            "unattributed_gb": op["unattributed_gb"],
            "transient_gb": peak_alloc_gb - op["alloc_start_gb"],
            "peak_alloc_gb": peak_alloc_gb, "peak_reserved_gb": peak_reserved_gb,
            "device_used_peak_gb": device_used_peak_gb,
            "host_rss_hwm_at_phase_end": _host_rss_gb(),
        }
        self._phases.append(phase)
        self._open_phase = None
        return phase

    def stop(self):
        """Stops the internal DeviceMemSampler's background thread --
        call once, after the last `phase_end()`, before `to_record()`."""
        self.sampler.stop()

    def to_record(self):
        """Assembles the T2b lifetime record: `buffers` (every distinct
        storage ever seen, with its create/destroy checkpoint), `phases`
        (every closed phase span), `checkpoints` (mark() call order), and
        `totals` (run-wide reconstructions):

          - `peak_alloc_gb_run`/`peak_reserved_gb_run`/`device_used_gb_run`:
            the true run-wide peak, reconstructed as the max across every
            phase's own (per-phase-reset) peak -- the same "max over
            recorded phases" technique `run_placement._phase_summary`
            already uses for exactly the same reason (a single end-of-run
            read of a repeatedly-reset counter would only reflect the
            *last* phase). This is sec 2.2's B1 anti-contamination rule:
            each HWM counter is reset per-phase (so no phase's reading is
            polluted by an earlier phase's high-water mark) while the
            run-global equivalent is still recoverable, not lost.
          - `resident_max_gb`/`max_phase_transient_gb`: the two terms of
            sec 2.2's corrected accounting formula, `peak_device ~=
            resident(t) + max_p transient_p`.
          - `identity_residual_gb = peak_alloc_gb_run - (resident_max_gb +
            max_phase_transient_gb)`: how far the true measured peak
            deviates from that formula -- should be small and is reported,
            not asserted, since the formula is explicitly *not* claimed
            exact (sec 2.2: "既非可行性證明亦非不可行證明").
          - `standalone_upper_bound_gb`: sum of every distinct buffer's
            own `bytes` (each counted once, at its own peak size) --
            this repo's convention for the ledger sec 2.2 calls a
            "分項保守上界" (per-component conservative upper bound),
            extended from "3 standalone spikes" to "every buffer this
            probe ever saw", i.e. "if every distinct buffer we've ever
            observed were simultaneously resident forever". Deliberately
            not a tight bound.
          - `upper_bound_over_measured = standalone_upper_bound_gb /
            peak_alloc_gb_run`: **non-predictive, diagnostic only** (design
            draft sec 7.1 T2b row) -- reports how conservative the
            standalone-sum upper bound was on THIS run, not something
            that generalizes to other scales.
        """
        buffers = [
            {"name": r["name"], "root": r["root"], "dtype": r["dtype"],
             "shape": list(r["shape"]), "bytes": r["bytes"],
             "storage_ptr": r["storage_ptr"],
             "create_phase": r["create_checkpoint"],
             "destroy_phase": r["destroy_checkpoint"]}
            for r in self._buffers.values()
        ]
        peak_alloc_gb_run = max((p["peak_alloc_gb"] for p in self._phases), default=0.0)
        peak_reserved_gb_run = max((p["peak_reserved_gb"] for p in self._phases), default=0.0)
        device_used_gb_run = max((p["device_used_peak_gb"] for p in self._phases), default=0.0)
        resident_max_gb = max((p["resident_gb"] for p in self._phases), default=0.0)
        max_phase_transient_gb = max((p["transient_gb"] for p in self._phases), default=0.0)
        identity_residual_gb = peak_alloc_gb_run - (resident_max_gb + max_phase_transient_gb)
        standalone_upper_bound_gb = sum(b["bytes"] for b in buffers) / 2**30
        upper_bound_over_measured = (
            standalone_upper_bound_gb / peak_alloc_gb_run if peak_alloc_gb_run else None)
        return {
            "buffers": buffers,
            "phases": self._phases,
            "checkpoints": list(self._checkpoints),
            "device_baseline_gb": self.device_baseline_gb,
            "totals": {
                "peak_alloc_gb_run": peak_alloc_gb_run,
                "peak_reserved_gb_run": peak_reserved_gb_run,
                "device_used_gb_run": device_used_gb_run,
                "resident_max_gb": resident_max_gb,
                "max_phase_transient_gb": max_phase_transient_gb,
                "identity_residual_gb": identity_residual_gb,
                "standalone_upper_bound_gb": standalone_upper_bound_gb,
                "upper_bound_over_measured": upper_bound_over_measured,
            },
        }
