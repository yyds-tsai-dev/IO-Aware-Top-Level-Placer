"""Posthoc schema4 GPU-only validation on the completed frozen 27.7M coordinates.

This does not replace the first-run timing or refit its prospective forecast.
The exact/tolerant aggregate comparisons are declared before the GPU work.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = REPO / "results/h100_nvl_followup_20260906"
CACHE = REPO / "results/recovery_visible_20260906/cache_schema4/3x3_n2"
OUT = ROOT / "ordered_gpu_validation.json"
PROTOCOL = ROOT / "ordered_gpu_validation.protocol.json"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def main():
    if OUT.exists() or PROTOCOL.exists():
        raise FileExistsError("preserve previous engineering validation")
    import numpy as np
    import torch
    from ioplace.bench.bookshelf_netlist import load_tiled_netlist
    from ioplace.bench.spike_30m import device_snapshot
    from ioplace.drivers.run_placement import get_regions_for, _pack_eval_metrics
    from ioplace.evaluator_gpu import GpuEvalContext
    from ioplace.export.evaluation import array_digest
    from ioplace.region_grid import RegionGrid
    torch.set_num_threads(2)
    reference_path = ROOT / "full_3x3_flat_k16.json"
    reference = json.loads(reference_path.read_text())
    registered = json.loads((ROOT / "forecast/protocol.json").read_text())
    receipt = json.loads((ROOT / "full_3x3_flat_k16.execution.json").read_text())
    if receipt["status"] != "completed" or receipt["output_sha256"] != sha(reference_path):
        raise ValueError("original run receipt mismatch")
    verification = json.loads((CACHE / "verification.json").read_text())["verify"]
    if not verification["ok"] or verification["mode"] != "full":
        raise ValueError("requires full cache verification")
    meta = json.loads((CACHE / "meta.json").read_text())
    if (meta["schema_version"] != 4 or meta["n_nodes"] != 27804699 or meta["n_nets"] != 31595252
            or meta["n_pins"] != 106683202 or meta.get("net_pin_order") != "native_sequential_output_front_swap"):
        raise ValueError("cache/count contract mismatch")
    legacy_meta = json.loads((REPO / "results/recovery_visible_20260906/cache/3x3_n2/meta.json").read_text())
    for key in ("manifest_source_sha256", "manifest_output_sha256", "n_pins_raw", "n_nodes", "n_nets", "n_pins"):
        if meta[key] != legacy_meta[key]:
            raise ValueError(f"schema4 source lineage differs: {key}")
    if not verification.get("native_pin_order_verified"):
        raise ValueError("schema4 native ordered verification required")
    exact = {key:reference[key] for key in ("io_count", "ft_count", "hard_lambda_sum", "io_rg", "ft_rg")}
    relative = {"hpwl":1e-9, "tree_wl":1e-5}
    hashes = {str(path):sha(path) for path in [reference_path, Path(str(reference_path)+".npz"),
        CACHE / "meta.json", CACHE / "verification.json", Path(__file__), *sorted(CACHE.glob("*.npy"))]}
    hashes.update({str(path):sha(path) for path in sorted((REPO / "ioplace").rglob("*.py"))})
    if sha(REPO / "ioplace/evaluator_gpu.py") != registered["source_sha256"]["ioplace/evaluator_gpu.py"]:
        raise ValueError("GPU evaluator differs from frozen first run")
    protocol = dict(kind="posthoc_schema4_engineering_validation", expected_exact=exact,
        expected_relative={key:dict(value=reference[key],rtol=rtol) for key,rtol in relative.items()},
        shift=[10260.,10080.], scale=1/380., dtype_order="raw -> float32; in-place transform; -> float64",
        geometry_basis="PlaceDB.initialize overrides scale when site_width!=1; verified SCL site width380",
        K=16, lattice=512, max_degree=256, n_iterations=3, cpu_threads=2,
        timeout_s=3600, hashes=hashes, registered_at=time.time(),
        first_run_forecast_and_timing_unchanged=True, historical_per_net_parity_claim=False)
    write(PROTOCOL, protocol)
    result = dict(status="running", pid=os.getpid(), started=time.time(), protocol_sha256=sha(PROTOCOL),
                  measurement_mode="shared", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), iterations=[])
    write(OUT,result)
    try:
        before = device_snapshot(); result["device_before"] = before
        if before["free_gib"] < 32:
            raise RuntimeError("less than32GiB free for bounded engineering check")
        start = time.perf_counter()
        raw, _ = load_tiled_netlist(str(CACHE), mmap=True)
        with np.load(str(reference_path)+".npz", allow_pickle=False) as saved:
            x, y = saved["node_x"], saved["node_y"]
        if x.shape != (raw.num_physical,) or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("coordinate count/finite check failed")
        for positions, cache_positions, shift in ((x,raw.node_x,10260.),(y,raw.node_y,10080.)):
            fixed = np.array(cache_positions[raw.num_movable:], dtype=np.float32)
            fixed -= shift
            fixed *= 1/380.
            if not np.array_equal(positions[raw.num_movable:],fixed.astype(np.float64)):
                raise ValueError("fixed/NI tail geometry or identity mismatch")
        def scale(values):
            value = np.array(values, dtype=np.float32, copy=True)
            value *= 1/380.
            return value.astype(np.float64)
        nl = replace(raw,node_x=x,node_y=y,node_size_x=scale(raw.node_size_x),node_size_y=scale(raw.node_size_y),
            pin_offset_x=scale(raw.pin_offset_x),pin_offset_y=scale(raw.pin_offset_y),
            xl=0.,yl=0.,xh=(raw.xh-10260.)/380.,yh=(raw.yh-10080.)/380.)
        result.update(fixed_and_ni_tail_exact=True, die=[nl.xl,nl.yl,nl.xh,nl.yh],
                      load_convert_s_after_hash_reads=time.perf_counter()-start)
        regions = RegionGrid(get_regions_for((nl.xl,nl.yl,nl.xh,nl.yh),16,"grid",1000,lattice=512))
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.perf_counter()
        context = GpuEvalContext(nl,regions,device="cuda",max_degree=256)
        torch.cuda.synchronize(); result["context_build_s"] = time.perf_counter()-start
        for iteration in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = context.evaluate(x,y)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            metrics = _pack_eval_metrics(value)
            checks = {key:metrics[key] == expected for key,expected in exact.items()}
            checks.update({key:bool(np.isclose(metrics[key],reference[key],rtol=rtol,atol=0)) for key,rtol in relative.items()})
            row = dict(iteration=iteration,evaluate_s=seconds,metrics=metrics,checks=checks,
                per_net_digests={key:array_digest(getattr(value,key)) for key in
                    ("per_net_crossings","per_net_ft","per_net_lambda","per_net_steiner","per_net_home")})
            if iteration == 0:
                baseline = ROOT / "parity_diagnosis_v1"
                old_io = np.load(baseline / "per_net_crossings.npy", mmap_mode="r")
                old_ft = np.load(baseline / "per_net_ft.npy", mmap_mode="r")
                changed = np.flatnonzero((value.per_net_crossings != old_io) | (value.per_net_ft != old_ft))
                row["changed_order_sensitive_nets"] = [dict(net=int(i), degree=int(nl.flat_net2pin_start[i+1]-nl.flat_net2pin_start[i]),
                    legacy_io=int(old_io[i]), legacy_ft=int(old_ft[i]),
                    native_io=int(value.per_net_crossings[i]), native_ft=int(value.per_net_ft[i])) for i in changed]
                # Preserve a small CPU oracle fixture for every changed IO/FT net.
                from ioplace.netlist import Netlist
                from ioplace.evaluator_ref import evaluate as reference_evaluate
                degree = nl.net_degrees[changed]
                starts = np.r_[0, np.cumsum(degree)]
                index = np.repeat(nl.flat_net2pin_start[changed]-starts[:-1], degree)+np.arange(starts[-1])
                pins = nl.flat_net2pin[index]
                px = x[nl.pin2node[pins]]+nl.pin_offset_x[pins]
                py = y[nl.pin2node[pins]]+nl.pin_offset_y[pins]
                tiny = Netlist(px,py,np.zeros(len(px)),np.zeros(len(px)),len(px),0,0,
                    np.zeros(len(px)),np.zeros(len(px)),np.arange(len(px),dtype=np.int32),
                    np.repeat(np.arange(len(changed),dtype=np.int32),degree),
                    np.arange(len(px),dtype=np.int32),starts.astype(np.int32),*regions.die)
                cpu = reference_evaluate(tiny,px,py,regions)
                row["changed_nets_cpu_parity"] = bool(np.array_equal(cpu.per_net_crossings,value.per_net_crossings[changed]) and np.array_equal(cpu.per_net_ft,value.per_net_ft[changed]))
                if not row["changed_nets_cpu_parity"]:
                    raise ValueError("changed net CPU/GPU parity failed")
                np.savez(ROOT / "ordered_gpu_validation_changed_nets.npz", net_ids=changed, degrees=degree,
                    px=px,py=py,die=np.array(regions.die),expected_io=cpu.per_net_crossings,expected_ft=cpu.per_net_ft)
            result["iterations"].append(row)
            write(OUT,result)
            if not all(checks.values()):
                raise ValueError(f"aggregate parity failed: {checks}")
        result.update(status="completed",aggregate_parity_pass=True,
            warm_mean_evaluate_s=float(np.mean([row["evaluate_s"] for row in result["iterations"][1:]])),
            process_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            process_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            device_after=device_snapshot())
    except BaseException as error:
        result.update(status="failed",error=repr(error))
        raise
    finally:
        result.update(ended=time.time(),host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        write(OUT,result)
    print(json.dumps({key:result[key] for key in ("status","aggregate_parity_pass","context_build_s","warm_mean_evaluate_s","process_peak_allocated_gib")},indent=2))


if __name__ == "__main__":
    main()
