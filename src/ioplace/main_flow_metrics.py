"""Pure metric functions for the v2 main flow's result.json (design v2 sec 7).

No torch, no DREAMPlace: every function here takes numpy arrays or plain
records so it can be unit-tested and re-run over saved artefacts.
"""
import numpy as np

# Every phase the v2 main flow can open, in run order. phase_summary emits a
# `t_<name>` key for each one whether or not it ran, so result.json's field
# contract (artifacts.MAIN_FLOW_RESULT_FIELDS) holds for `--phase fence` too.
MAIN_FLOW_PHASES = ("read_soft", "gp_soft", "freeze", "read_fence", "gp_fence",
                    "lg", "eval")


def io_accounting(io_soft, io_fence_gp, io_final):
    """The design's closing identity:
    `io(final) = io(soft, last GP) + io_delta_at_freeze + lg_loss`.

    `lg_loss` keeps run_placement_io.py:691-692's definition (post-LG minus
    the last GP evaluation); `io_delta_at_freeze` is everything the fence
    phase changed, measured between the freeze evaluation and the last
    fence-GP evaluation.

    `io_identity_residual` is `io_final - (io_soft + delta + lg_loss)`, which
    is 0 for *any* three inputs -- both summands were just defined as
    differences of them. It is kept as a result.json field because the schema
    documents the identity, but it detects nothing (pre-flight amendment D-1);
    the field that does is `io_fence_gp_source`, which run_fence_gp sets from
    whether the legalize_op wrapper actually ran.
    """
    io_soft, io_fence_gp, io_final = int(io_soft), int(io_fence_gp), int(io_final)
    delta = io_fence_gp - io_soft
    lg_loss = io_final - io_fence_gp
    return {"io_soft": io_soft, "io_fence_gp": io_fence_gp, "io_count": io_final,
            "io_delta_at_freeze": delta, "lg_loss": lg_loss,
            "io_identity_residual": io_final - (io_soft + delta + lg_loss)}


def region_area_balance(part, node_size_x, node_size_y, rs):
    """Per-region cell count/area/utilisation plus the max-min summaries.

    The four per-region arrays come from `freeze.region_cell_stats` -- one
    implementation, two consumers (pre-flight amendment D-2); this function
    adds only the summaries result.json quotes. The import is local because
    `freeze` pulls in `ops/soft_assign`, which imports torch, and this module
    must stay loadable in a bare CPU report process.

    Utilisation is `cell area / region area`, which is invariant under
    PlaceDB's shift+scale (both terms carry `scale_factor**2`).
    `region_cell_area`/`region_area` are **not**, so call this with
    native-unit sizes and the native `RegionSet`: that is the frame
    `freeze.json` is written in, and result.json must quote the same numbers.
    """
    from ioplace.freeze import region_cell_stats
    stats = region_cell_stats(part, node_size_x, node_size_y, rs)
    counts = np.asarray(stats["region_cell_count"], dtype=np.float64)
    utilization = np.asarray(stats["region_utilization"], dtype=np.float64)
    k = rs.k
    mean_count = counts.mean() if k else 0.0
    out = dict(stats)
    out.update({
        "k": int(k),
        "utilization_max": float(utilization.max()),
        "utilization_min": float(utilization.min()),
        "utilization_ratio": (float(utilization.max() / utilization.min())
                              if utilization.min() > 0 else None),
        "cell_count_max": int(counts.max()), "cell_count_min": int(counts.min()),
        "cell_count_deviation": (float(np.max(np.abs(counts - mean_count)) / mean_count)
                                 if mean_count > 0 else None),
        "empty_regions": np.flatnonzero(counts == 0).astype(int).tolist()})
    return out


def fence_compliance(rg, node_x, node_y, part, node_size_x=None, node_size_y=None):
    """Fraction of movable cells that landed in their assigned region.

    `lower_left` is run_placement_two_stage.py:252-255's definition, kept so
    the number stays comparable with the legacy two-stage arm. `center` uses
    the same anchor as the freeze membership (design v2 sec 3 phase 2) and is
    the one to quote for the v2 flow; it is None when sizes are not supplied.
    """
    part = np.asarray(part, dtype=np.int64)
    m = len(part)
    x = np.asarray(node_x, dtype=np.float64)[:m]
    y = np.asarray(node_y, dtype=np.float64)[:m]
    out = {"lower_left": float((rg.region_of_points(x, y) == part).mean()),
           "center": None}
    if node_size_x is not None and node_size_y is not None:
        cx = x + 0.5 * np.asarray(node_size_x, dtype=np.float64)[:m]
        cy = y + 0.5 * np.asarray(node_size_y, dtype=np.float64)[:m]
        out["center"] = float((rg.region_of_points(cx, cy) == part).mean())
    return out


def phase_summary(timer, sampler, *, names=MAIN_FLOW_PHASES, host_rss=None):
    """Assemble the per-phase runtime/GPU-peak block from a `profile.PhaseTimer`
    and a `profile.DeviceMemSampler`.

    `peak_mem_mb` is the max over phase peaks, not a single end-of-run read:
    each phase calls `reset_peak_memory_stats()` on entry, so the global
    counter holds only the last phase's peak by the end (same reasoning as
    run_placement._phase_summary). Phases recorded with `peak_alloc_gb=None`
    (PhaseTimer(reset_peak=False)) are skipped rather than counted as zero.
    """
    phases = dict(timer.phases)
    out = {"phases": phases, "peak_mem_mb_by_phase": {}}
    for name in names:
        out[f"t_{name}"] = 0.0
    for name, record in phases.items():
        out[f"t_{name}"] = float(record.get("t_s", 0.0))
        peak = record.get("peak_alloc_gb")
        out["peak_mem_mb_by_phase"][name] = None if peak is None else peak * 1024.0
    measured = [value for value in out["peak_mem_mb_by_phase"].values()
                if value is not None]
    out["peak_mem_mb"] = max(measured) if measured else 0.0
    if host_rss is None:
        from ioplace.profile import host_rss_gb
        host_rss = host_rss_gb()
    host_peaks = [record.get("host_rss_hwm_at_phase_end") or 0.0
                  for record in phases.values()] + [host_rss]
    out["device_used_gb"] = sampler.device_used_gb
    out["host_peak_rss_gb"] = max(host_peaks)
    return out
