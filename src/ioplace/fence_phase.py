"""Phase-3 (fence GP) PlaceDB construction and density-weight clamp
(design v2 sec 3 phase 3).

Fence data may only be injected between `read()` and `initialize()`
(fence_inject.py:9-16), so phase 3 always builds a *fresh* PlaceDB; that is
what forces the two-instance split in the first place.
"""
import numpy as np

from ioplace.artifacts import placedb_identity_sha256
from ioplace.drivers.run_placement import _load_dreamplace
from ioplace.drivers.run_placement_io import _install_attribute
from ioplace.drivers.run_placement_two_stage import _pick_escape_cell
from ioplace.fence_inject import inject_fence_regions
from ioplace.init_pos import apply_init

DENSITY_CLAMP_LO = 0.25
DENSITY_CLAMP_HI = 4.0


def clamp_density_weight(model, reference, lo=DENSITY_CLAMP_LO, hi=DENSITY_CLAMP_HI):
    """Clamp `model.density_weight` to `[lo*reference, hi*reference]`.

    In fence mode `density_weight == density_weight_u * s` and the
    overflow-based update recomputes it from `density_weight_u`
    (PlaceObj.py:862-875), so the clamp scales `u` by the same elementwise
    ratio and recomputes the step size exactly as PlaceObj.py:817 does --
    otherwise the very first update would undo the clamp.

    The returned `lo_abs`/`hi_abs` are the **absolute** bounds
    (`lo*reference`, `hi*reference`), not the multipliers; they coincide only
    when `reference == 1` (pre-flight amendment D-14).
    """
    import torch
    reference = float(reference)
    if not reference > 0.0:
        raise ValueError(f"reference density weight must be positive, got {reference}")
    low, high = lo * reference, hi * reference
    with torch.no_grad():
        before = model.density_weight.detach().clone()
        after = before.clamp(min=low, max=high)
        # dtype-safe floor: GCD's placedb.dtype is float32, where 1e-300
        # underflows to 0.0 and the guard would rest entirely on torch.where's
        # mask (pre-flight amendment D-10).
        tiny = torch.finfo(before.dtype).tiny
        ratio = torch.where(before > 0, after / before.clamp_min(tiny),
                            torch.ones_like(before))
        model.density_weight.copy_(after)
        u = getattr(model, "density_weight_u", None)
        if u is not None:
            u.mul_(ratio)
            model.density_weight_step_size = (
                model.density_weight_step_size_inc_low - 1.0) * float(u.norm(p=2))
    return {"reference": reference, "lo_abs": float(low), "hi_abs": float(high),
            "before": [float(v) for v in before.reshape(-1)],
            "after": [float(v) for v in after.reshape(-1)],
            "bound": bool(torch.any(after != before))}


def install_density_weight_clamp(cleanup, reference, log, *, lo=DENSITY_CLAMP_LO,
                                 hi=DENSITY_CLAMP_HI):
    """Scope a clamp around every `PlaceObj.initialize_density_weight` call.

    The class, not the instance, is patched: `placer.model` only exists once
    `NonLinearPlace.__call__` assigns it (NonLinearPlace.py:236), the density
    weight is initialised at NonLinearPlace.py:399-400, and the iteration
    callback only fires at NonLinearPlace.py:521 -- after
    `make_parameter_update()` (NonLinearPlace.py:454/462) has already taken a
    step at the unclamped weight. `_install_attribute` restores the original
    function when `cleanup` closes, including on an exception.

    The patch is **process-global**: every `PlaceObj` built in this interpreter
    sees it while it is installed, so it must only ever be installed inside an
    `_io_cleanup()` scope (which is what guarantees the restore) and never
    around code that runs a second, unrelated placement concurrently.
    `initialize_density_weight` has two call sites (NonLinearPlace.py:400 and
    :783), so `log` may collect more than one entry per run (pre-flight
    amendment D-4).
    """
    import PlaceObj
    original = PlaceObj.PlaceObj.initialize_density_weight

    def wrapped(self, params, placedb):
        weight = original(self, params, placedb)
        log.append(clamp_density_weight(self, reference, lo, hi))
        return weight

    _install_attribute(cleanup, PlaceObj.PlaceObj, "initialize_density_weight", wrapped)


def build_fence_placedb(config_json, region_set, part, positions, *,
                        dp_seed=None, deterministic=None):
    """Read a fresh PlaceDB, inject fences from the frozen membership, apply the
    escape-cell workaround, warm start from `positions`, then initialize()."""
    params, placedb = _load_dreamplace(config_json)
    assert params.enable_fillers == 1, "fence mode requires enable_fillers"
    if dp_seed is not None:
        params.random_seed = dp_seed
    if deterministic is not None:
        params.deterministic_flag = deterministic

    k = region_set.k
    m = placedb.num_movable_nodes
    part = np.asarray(part, dtype=np.int32)
    if len(part) != m:
        raise ValueError(f"membership has {len(part)} entries, expected {m}")
    die_native = (float(placedb.xl), float(placedb.yl),
                  float(placedb.xh), float(placedb.yh))
    span = max(die_native[2] - die_native[0], die_native[3] - die_native[1])
    if not np.allclose(region_set.die, die_native, atol=1e-6 * span, rtol=0.0):
        raise ValueError(f"region set die {region_set.die} != placedb die {die_native}; "
                         "regions.json must be in native post-read units")
    sha = placedb_identity_sha256(placedb)
    if positions.placedb_sha256 != sha:
        raise ValueError("warm-start positions were produced for a different netlist "
                         f"({positions.placedb_sha256} != {sha})")

    inject_fence_regions(placedb, region_set, part)
    # DREAMPlace always allocates an implicit "no fence" bucket (region_id == k)
    # and calls calc_num_filler_for_fence_region on it; our regions tile the whole
    # die, so that bucket is empty and np.percentile of the empty slice raises
    # IndexError at PlaceDB.py:687 under numpy 1.26.4 (pre-flight E-3; the older
    # "int(round(nan)) -> ValueError at :729" wording is wrong). Same workaround,
    # same helper, as run_placement_two_stage.py:192-233.
    escape = _pick_escape_cell(placedb.node2fence_region_map, part,
                               placedb.node_size_x, placedb.node_size_y, k)
    escape_from = int(placedb.node2fence_region_map[escape])
    placedb.node2fence_region_map[escape] = k

    init = apply_init(placedb, params, "seed", positions=positions)
    placedb.initialize(params)

    info = {"escape_cell": int(escape), "escape_from": escape_from,
            "placedb_sha256": sha,
            "shift_factor": (float(params.shift_factor[0]), float(params.shift_factor[1])),
            "scale_factor": float(params.scale_factor),
            "die_native": die_native,
            "die_scaled": (float(placedb.xl), float(placedb.yl),
                           float(placedb.xh), float(placedb.yh)),
            "init": init}
    return params, placedb, info
