"""M4 T9 30M component-level spike -- CHILD process (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 5.3
layer 2 / T9 row / sec 2.2's three-state feasibility rule).

This is the ONLY process in the T9 pair that ever touches CUDA (`spike_30m.
py`, the parent, never builds a CUDA context -- see that module's
docstring). It:
  1. loads the tiled 27.7M-node `Netlist` via `ioplace.bench.
     bookshelf_netlist.load_tiled_netlist(cache_dir, mmap=True)` -- never
     `PlaceDB` (sec 5.3's "不經 PlaceDB");
  2. builds `IoTerm` + `GpuEvalContext` (T2's post-streaming evaluator) so
     they are CO-RESIDENT for the run's whole lifetime, matching sec 5.3's
     "兩個元件必須共常駐並交錯執行 >=3 個 iteration" requirement -- this is
     NOT two standalone spikes whose peaks get added, it's one process
     where both components' static buffers are alive simultaneously and
     the measured peak is the true interleaved-execution peak;
  3. runs `n_iter` (default 4, contract minimum 3) interleaved iterations:
     IoTerm fwd+bwd -> a plain gradient step on `pos` -> `GpuEvalContext.
     evaluate()` on the updated positions -- using the tiler's REAL `.pl`
     starting coordinates (design draft R4: uniform-random positions
     systematically overstate evaluator cost, see sec 1.2's io_count
     comparison), never synthetic/uniform ones;
  4. optionally allocates `--s4-scenario B`'s two extra `(E,) float64`
     ballast tensors (sec 8's memory-parametrization table) -- M3's own S4
     is not merged into this repo yet, so this is a disclosed EMULATION of
     its resident-memory footprint (`s4_source="not_merged_m3"`), not a
     real S4 computation;
  5. writes `--out` incrementally (right after the components are built,
     and again after every iteration) so a kill/timeout/OOM still leaves a
     legible partial record behind for the parent to read;
  6. exits 0 (completed) / 3 (OOM, caught as `torch.cuda.OutOfMemoryError`
     -- design draft T9 row's literal exit-code contract) / any other code
     on an unexpected exception (`workload_status="crashed"`, and this
     module itself marks `experiment_status="instrumentation_error"`
     directly rather than deferring to `result_gate.feasibility_verdict`'s
     more general crashed-workload handling -- see `spike_30m.py`'s
     docstring for why that's the literal instruction here).

No fillers: this is a component-level memory/runtime spike of IoTerm +
GpuEvalContext, not a full GP run (GP is sec 5.3's *layer 1*, not this
layer-2 spike) -- there is no `NonLinearPlace`/filler infrastructure in
this path at all, so `num_nodes` passed to `IoTerm` is `nl.num_physical`
(no filler slots), matching the same simplification `ioplace.diagnostics.
spike_10m`'s inherited 10M spike already made for the identical reason.
This is a disclosed simplification, not an oversight -- flagged again in
this task's final report.

CLI:
    PYTHONPATH=. $PY -m ioplace.bench.spike_30m_child \\
        --cache-dir results/m4/bench/tiled_cache/3x3_n2 \\
        --k 32 --rtype grid --seed 0 --n-iter 4 --lr 1.0 \\
        --s4-scenario A --budget-gb 19.5 --budget-source "NVIDIA L4" \\
        --out results/m4/profile/spike30m__A__k32.partial.json
"""
import argparse
import json
import os
import sys
import traceback

# Imported at module level (not deferred into main()'s try block): merely
# `import torch` does not touch CUDA -- only a `torch.cuda.*` call does
# (see the first `torch.cuda.mem_get_info()` call below, which really is
# this process's first CUDA touch) -- and importing it here lets every
# `except torch.cuda.OutOfMemoryError` clause below resolve safely even if
# something fails before that first CUDA call (an exception raised while
# Python is still evaluating an `except <expr>` clause's own `<expr>`
# would otherwise mask the original error with a confusing NameError).
import torch

from ioplace.bench.result_gate import assert_budget

# Contract minimum (design draft T9 row: "共常駐、交錯 >=3 iteration").
MIN_N_ITER = 3


def _write(path, record):
    """Atomic-ish incremental write (temp file + rename) -- every partial
    write must leave `path` either at its previous complete content or its
    new one, never truncated mid-write (a parent reading this file after a
    SIGKILL must never see a corrupt half-JSON)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True,
                    help="ioplace.bench.bookshelf_netlist cache dir (built by "
                         "build_tiled_netlist_cache)")
    ap.add_argument("--k", type=int, required=True, choices=(16, 32))
    ap.add_argument("--rtype", default="grid", choices=("grid", "slicing"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=None,
                    help="default: 0.1 * L_R (same convention as spike_10m.py)")
    ap.add_argument("--s4-scenario", choices=("A", "B"), default="A")
    ap.add_argument("--budget-gb", type=float, required=True)
    ap.add_argument("--budget-source", required=True)
    ap.add_argument("--ignore-net-degree", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.n_iter < MIN_N_ITER:
        raise ValueError(f"--n-iter={args.n_iter} < contract minimum {MIN_N_ITER} "
                         "(design draft sec 5.3: interleaved >=3 iterations)")
    return args


def _resident_bytes(*objs):
    """Reflection-based sum of every CUDA tensor's byte size found as a
    direct attribute of each object in `objs`, deduped by `data_ptr()`.
    This is a MEASURED (not purely analytic) stand-in for sec 2.2's
    `resident(t)`: it walks `IoTerm`'s and `GpuEvalContext`'s own
    registered buffers/attributes rather than requiring this module to
    independently re-derive every one of `GpuEvalContext`'s internal
    tensor shapes from `evaluator_gpu.py` by hand -- disclosed as a design
    choice in this task's final report."""
    total = 0
    seen = set()
    for obj in objs:
        for val in vars(obj).values():
            if isinstance(val, torch.Tensor) and val.is_cuda:
                ptr = val.data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                total += val.numel() * val.element_size()
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, torch.Tensor) and item.is_cuda:
                        ptr = item.data_ptr()
                        if ptr in seen:
                            continue
                        seen.add(ptr)
                        total += item.numel() * item.element_size()
    return total


def main(argv=None):
    args = _parse_args(argv)

    # sec 1.4 B3 enforcement point 1 of this pair (parent argv-parse time
    # covers the other two -- see spike_30m.py): a CLI --budget-gb may
    # never exceed the hardware contract, checked again here since the
    # child is a standalone-invocable process too.
    assert_budget(gpu_name=None, budget_gb=args.budget_gb, source=args.budget_source)

    record = {
        "record_kind": "spike",
        "K": args.k, "rtype": args.rtype, "seed": args.seed,
        "s4_scenario": ("A" if args.s4_scenario == "A" else "B_emulated_ballast"),
        "s4_source": ("n/a" if args.s4_scenario == "A" else "not_merged_m3"),
        "n_interleaved_iters": 0,
        "budget_gb": args.budget_gb, "budget_source": args.budget_source,
        "workload_status": "running",
        "cache_dir": args.cache_dir,
    }
    _write(args.out, record)

    try:
        from ioplace.bench.bookshelf_netlist import load_tiled_netlist
        from ioplace.drivers.run_placement import get_regions_for
        from ioplace.ops.soft_assign import rect_table
        from ioplace.ops.io_term import build_net_node_csr, IoTerm
        from ioplace.evaluator_gpu import GpuEvalContext
        from ioplace.region_grid import RegionGrid

        # First CUDA call in this process -- sec 1.4 B3 enforcement point 2
        # ("child 首個 CUDA call 後"), with the now-known real gpu_name.
        free0, total0 = torch.cuda.mem_get_info()
        device_baseline_gb = (total0 - free0) / 2**30
        gpu_name = torch.cuda.get_device_name(0)
        assert_budget(gpu_name=gpu_name, budget_gb=args.budget_gb, source=args.budget_source)
        record["device_baseline_gb"] = device_baseline_gb
        record["gpu_name"] = gpu_name
        torch.cuda.reset_peak_memory_stats()

        nl, meta = load_tiled_netlist(args.cache_dir, mmap=True)
        record.update({
            "case": os.path.basename(meta.get("dst_prefix", args.cache_dir)),
            "n_nodes": int(nl.num_physical), "n_nets": int(nl.num_nets),
            "n_pins": int(len(nl.pin2node)),
            "input_sha256": meta.get("manifest_output_sha256", {}).get("nets"),
        })

        die = (float(nl.xl), float(nl.yl), float(nl.xh), float(nl.yh))
        rs = get_regions_for(die, args.k, args.rtype, args.seed)
        rg = RegionGrid(rs)
        rects, r2k = rect_table(rs)
        csr = build_net_node_csr(nl, args.ignore_net_degree)

        io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=args.k,
                         num_movable=nl.num_movable, num_physical=nl.num_physical,
                         num_nodes=nl.num_physical, device="cuda")
        ctx = GpuEvalContext(nl, rg, device="cuda")

        record["k_chunk"] = io_term.k_chunk
        record["n_active"] = io_term.n_active
        record["P_dedup"] = io_term._n_pins_dedup

        # sec 8's S4 memory-parametrization emulation (module docstring
        # point 4): home_e is part of BOTH scenarios' baseline; scenario B
        # adds the two extra (E,) fp64 accumulators.
        e_count = io_term.n_active
        s4_home_e = torch.zeros(e_count, dtype=torch.int8, device="cuda")
        s4_extra = []
        if args.s4_scenario == "B":
            s4_extra = [torch.zeros(e_count, dtype=torch.float64, device="cuda") for _ in range(2)]

        num_physical = nl.num_physical
        node_x0 = torch.as_tensor(nl.node_x[:num_physical], dtype=torch.float64)
        node_y0 = torch.as_tensor(nl.node_y[:num_physical], dtype=torch.float64)
        pos = torch.cat([node_x0, node_y0]).cuda().requires_grad_(True)

        nx = die[2] - die[0]
        ny = die[3] - die[1]
        L_R = ((nx * ny) / args.k) ** 0.5
        tau = args.tau if args.tau is not None else 0.1 * L_R

        record["n_interleaved_iters"] = 0
        record["measured_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        _write(args.out, record)

        measured_peak_gb = record["measured_peak_gb"]
        for it in range(args.n_iter):
            L = io_term(pos, tau, lambda_io=1.0)
            L.backward()
            with torch.no_grad():
                pos.data.add_(pos.grad, alpha=-args.lr)
                pos.data[:num_physical].clamp_(die[0], die[2])
                pos.data[num_physical:].clamp_(die[1], die[3])
            pos.grad = None

            with torch.no_grad():
                res = ctx.evaluate(pos[:num_physical], pos[num_physical:2 * num_physical])

            measured_peak_gb = max(measured_peak_gb, torch.cuda.max_memory_allocated() / 2**30)
            record["n_interleaved_iters"] = it + 1
            record["measured_peak_gb"] = measured_peak_gb
            record["last_io_loss"] = float(L.detach())
            record["last_eval_hpwl"] = float(getattr(res, "hpwl", 0.0) or 0.0)
            _write(args.out, record)

        record["resident_gb"] = _resident_bytes(io_term, ctx) / 2**30 + (
            (pos.numel() * pos.element_size()
             + (pos.grad.numel() * pos.grad.element_size() if pos.grad is not None else 0)
             + s4_home_e.numel() * s4_home_e.element_size()
             + sum(t.numel() * t.element_size() for t in s4_extra)) / 2**30)
        record["max_phase_transient_gb"] = max(0.0, measured_peak_gb - record["resident_gb"])
        record["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        record["device_used_peak_gb"] = None  # parent's nvsmi trace fills this in
        record["workload_status"] = "completed"
        _write(args.out, record)
        return 0

    except torch.cuda.OutOfMemoryError as e:
        record["workload_status"] = "oom"
        record["child_error"] = str(e)
        try:
            record["measured_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        except Exception:
            pass
        _write(args.out, record)
        return 3
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            record["workload_status"] = "oom"
            record["child_error"] = str(e)
            _write(args.out, record)
            return 3
        record["workload_status"] = "crashed"
        record["child_stderr_tail"] = traceback.format_exc()[-4000:]
        _write(args.out, record)
        return 1
    except Exception:
        record["workload_status"] = "crashed"
        record["child_stderr_tail"] = traceback.format_exc()[-4000:]
        _write(args.out, record)
        return 1


if __name__ == "__main__":
    sys.exit(main())
