"""v2 artefact I/O and the native/scaled coordinate contract (design v2 sec 1).

Shared by BOTH v2 drivers: the main flow (P-B, drivers/run_main_flow.py) and the
region producer (P-C, drivers/run_region_producer.py). Every name the producer
needs -- placedb_identity_sha256, save/load_positions, save/load_membership,
save/load_producer_json -- lives here; P-C adds nothing of its own.

Every array artefact here is written in **native post-read PlaceDB units**:
the coordinate frame of `placedb` after `placedb.read(params)` and before
`placedb.initialize(params)`. `initialize()` calls `PlaceDB.scale()`
($DREAMPLACE_ROOT/install/dreamplace/PlaceDB.py:151-196), which rescales
node positions, node sizes, the die box and `regions`/`flat_region_boxes`
together with `shift_factor = (xl, yl)` and `scale_factor = 1/site_width`
fixed at PlaceDB.py:759-767 -- native units are therefore the only frame in
which a seed, a region set and a membership vector produced by different
processes can be combined.

This module deliberately imports neither torch nor DREAMPlace: it must stay
loadable in a bare CPU process (report tooling, tests).
"""
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from ioplace.regions import RegionSet, RegionSpec

POSITIONS_SCHEMA_VERSION = 1
MEMBERSHIP_SCHEMA_VERSION = 1
FREEZE_SCHEMA_VERSION = 1
MAIN_FLOW_RESULT_SCHEMA_VERSION = 1
PRODUCER_SCHEMA_VERSION = 1

POSITION_KINDS = ("seed", "soft")

FREEZE_FIELDS = (
    "schema_version", "iteration", "reason", "overflow", "tau", "tau_rel",
    "churn", "k", "region_cell_count", "region_cell_area", "region_area",
    "region_utilization", "io_soft", "membership_npz", "soft_npz",
    "repaired_empty_regions", "gp_iterations_soft", "density_weight_soft",
)

MAIN_FLOW_RESULT_FIELDS = (
    # identity / provenance
    "mode", "schema_version", "config", "k", "rtype", "seed", "init",
    "norm_policy", "phase", "regions_json", "run_id", "status", "repo_commit",
    "input_sha256", "env", "command", "hostname", "benchmark_kind",
    "device_baseline_gb", "dp_seed", "det", "runtime_s",
    # IO accounting (design sec 7 diagnostics 4/5 + the closing identity)
    "io_soft", "io_fence_gp", "io_count", "io_delta_at_freeze", "lg_loss",
    "io_identity_residual", "io_fence_gp_source",
    # v2 P-F (design sec 7): the anchor the soft phase ran with, and the three
    # evaluator-side diagnostics plus their two components. Together with
    # io_delta_at_freeze and fence_compliance above, these are F's five.
    "node_anchor", "straddle_cells", "straddle_area_fraction",
    "straddle_pin_split_nets", "straddle_out_area", "straddle_movable_area",
    "straddle_wide_cells",
    # evaluator metrics
    "ft_count", "hard_lambda_sum", "tree_wl", "hpwl", "io_rg", "ft_rg",
    "large_net_lb", "hpwl_gp", "hpwl_lg",
    # fence + geometry diagnostics
    "fence_compliance", "fence_compliance_center", "region_area_balance",
    "freeze", "density_weight_clamp", "escape_cell",
    # runtime / memory / legalization
    "phases", "t_read_soft", "t_gp_soft", "t_freeze", "t_read_fence",
    "t_gp_fence", "t_lg", "t_eval", "peak_mem_mb", "peak_mem_mb_by_phase",
    "device_used_gb", "host_peak_rss_gb",
    "gp_iterations_soft", "gp_iterations_fence", "gp_iteration_budget",
    "final_overflow", "stop_overflow_reached", "legalization_status",
    "num_unplaced_cells", "effective_target_density", "num_filler_nodes",
    "num_bins_x", "num_bins_y",
    # soft-phase provenance (the run_soft_phase record minus its arrays; None
    # for a --phase fence run, which never opens the soft phase)
    "soft_summary",
    # artefacts
    "artifacts",
)

# producer.json -- the region producer's run record (P-C Task 10 builds it).
PRODUCER_FIELDS = (
    # identity / provenance
    "config", "out_dir", "placedb_sha256", "command", "hostname", "env",
    # knobs
    "k", "membership_source", "membership_seed", "epsilon", "hierarchy_depth",
    "extract_bins", "fine_bins", "lattice", "rect_max", "t_hull",
    "probe_every", "alpha_pull", "alpha_push", "sa_seed",
    # geometry and the coordinate contract
    "die_native", "die_scaled", "shift_factor", "scale_factor",
    "num_movable", "num_physical", "num_nodes", "num_nets", "target_density",
    # grouping-term telemetry
    "n_hull_rebuilds", "wt_final", "lambda_group_final", "ratio_ema_final",
    "probes",
    # placement outcome
    "gp_iterations_run", "final_overflow", "hpwl_gp", "hpwl_lg",
    # shapes
    "sa", "rects_per_region", "rect_max_observed", "region_bins",
    "region_area", "region_cell_area", "region_utilisation", "area_balance",
    # runtime
    "runtime_s", "peak_mem_mb",
)


@dataclass
class Positions:
    node_x: np.ndarray
    node_y: np.ndarray
    die: tuple
    shift_factor: tuple
    scale_factor: float
    placedb_sha256: str
    kind: str

    @property
    def num_physical(self):
        return len(self.node_x)


@dataclass
class Membership:
    part: np.ndarray
    source: str
    k: int
    seed: int
    epsilon: float

    @property
    def num_movable(self):
        return len(self.part)


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def placedb_identity_sha256(placedb):
    """Fingerprint of the netlist structure a seed/membership must match.

    Must be computed after read() and BEFORE initialize(): node sizes are
    multiplied by scale_factor inside initialize() (PlaceDB.py:160-161), so
    the same design would otherwise fingerprint differently in phase 1 and
    phase 3 and every cross-phase check would spuriously fail. The region
    producer (P-C) hashes in its own `read` phase for the same reason.

    Covers counts, node sizes and full pin connectivity -- never a coordinate.
    pin2node_map/pin2net_map come from P-C's `placedb_fingerprint`, which this
    function replaces: the start map alone does not notice a permutation of the
    pins inside a net, which is exactly the "seed written for a different
    design" case the guard exists for.
    """
    return _digest_arrays(
        np.asarray([placedb.num_movable_nodes, placedb.num_physical_nodes,
                    placedb.num_nets, len(placedb.pin2node_map)], dtype=np.int64),
        np.asarray(placedb.node_size_x[:placedb.num_physical_nodes], dtype=np.float64),
        np.asarray(placedb.node_size_y[:placedb.num_physical_nodes], dtype=np.float64),
        np.asarray(placedb.pin2node_map, dtype=np.int64),
        np.asarray(placedb.pin2net_map, dtype=np.int64),
        np.asarray(placedb.flat_net2pin_start_map, dtype=np.int64))


def _restore_umask_permissions(path):
    """mkstemp() creates the temporary file at 0600; os.replace() preserves
    that mode across the rename, so every artefact would otherwise end up
    unreadable by anyone but its writer regardless of the process umask.
    Read the umask (os.umask() has no read-only form) and immediately put it
    back, then apply the same 0666-minus-umask a normal open()/write() would
    have produced."""
    mask = os.umask(0)
    os.umask(mask)
    os.chmod(path, 0o666 & ~mask)


def _atomic_savez(path, **arrays):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", suffix=".npz", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
        _restore_umask_permissions(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_default(obj):
    """json.dump's `default=` hook: coerce the numpy/torch scalars that
    naturally fall out of driver code (np.float32 overflow ratios, np.int64
    counters, 0-d torch losses) into native JSON types. Anything else --
    including a multi-element array/tensor, which would silently lose shape
    under a bare `.tolist()`/`.item()` -- is refused rather than guessed at.
    Duck-typed on `.detach()` so this module still need not import torch."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach"):
        try:
            if obj.numel() == 1:
                return obj.item()
        except Exception:
            pass
    raise ValueError(f"value of type {type(obj).__name__} is not JSON-serialisable")


def _atomic_write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".artifact-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=1, sort_keys=True, default=_json_default)
        os.replace(temporary, path)
        _restore_umask_permissions(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_positions(path, node_x, node_y, *, die, shift_factor, scale_factor,
                   placedb_sha256, kind):
    if kind not in POSITION_KINDS:
        raise ValueError(f"kind must be one of {POSITION_KINDS}, got {kind!r}")
    x = np.asarray(node_x, dtype=np.float64).reshape(-1)
    y = np.asarray(node_y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("node_x and node_y must have the same length")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("positions must be finite")
    die = tuple(float(v) for v in die)
    shift_factor = tuple(float(v) for v in shift_factor)
    if len(die) != 4:
        raise ValueError("die must be (xl, yl, xh, yh)")
    if len(shift_factor) != 2:
        raise ValueError("shift_factor must be (dx, dy)")
    if not float(scale_factor) > 0.0:
        raise ValueError("scale_factor must be positive")
    _atomic_savez(path, node_x=x, node_y=y,
                  die=np.asarray(die, dtype=np.float64),
                  shift_factor=np.asarray(shift_factor, dtype=np.float64),
                  scale_factor=np.asarray(float(scale_factor), dtype=np.float64),
                  placedb_sha256=np.asarray(str(placedb_sha256)),
                  kind=np.asarray(str(kind)),
                  schema_version=np.asarray(POSITIONS_SCHEMA_VERSION, dtype=np.int64))


def load_positions(path, *, expect_num_physical=None, expect_sha256=None):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if int(data["schema_version"]) != POSITIONS_SCHEMA_VERSION:
        raise ValueError(f"unsupported positions schema in {path}")
    out = Positions(
        node_x=np.asarray(data["node_x"], dtype=np.float64),
        node_y=np.asarray(data["node_y"], dtype=np.float64),
        die=tuple(float(v) for v in data["die"]),
        shift_factor=tuple(float(v) for v in data["shift_factor"]),
        scale_factor=float(data["scale_factor"]),
        placedb_sha256=str(data["placedb_sha256"]),
        kind=str(data["kind"]))
    if expect_num_physical is not None and out.num_physical != int(expect_num_physical):
        raise ValueError(f"{path}: num_physical {out.num_physical} != {expect_num_physical}")
    if expect_sha256 is not None and out.placedb_sha256 != expect_sha256:
        raise ValueError(f"{path}: placedb fingerprint mismatch "
                         f"({out.placedb_sha256} != {expect_sha256})")
    return out


def save_membership(path, part, *, source, k, seed=0, epsilon=0.0):
    arr = np.asarray(part).reshape(-1).astype(np.int32)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    if arr.size and (arr.min() < 0 or arr.max() >= k):
        raise ValueError(f"membership out of range [0, {k})")
    _atomic_savez(path, part=arr,
                  source=np.asarray(str(source)),
                  k=np.asarray(k, dtype=np.int64),
                  seed=np.asarray(int(seed), dtype=np.int64),
                  epsilon=np.asarray(float(epsilon), dtype=np.float64),
                  schema_version=np.asarray(MEMBERSHIP_SCHEMA_VERSION, dtype=np.int64))


def load_membership(path, *, expect_num_movable=None, expect_k=None,
                    require_nonempty=False):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if int(data["schema_version"]) != MEMBERSHIP_SCHEMA_VERSION:
        raise ValueError(f"unsupported membership schema in {path}")
    out = Membership(part=np.asarray(data["part"], dtype=np.int32),
                     source=str(data["source"]), k=int(data["k"]),
                     seed=int(data["seed"]), epsilon=float(data["epsilon"]))
    if expect_num_movable is not None and out.num_movable != int(expect_num_movable):
        raise ValueError(f"{path}: num_movable {out.num_movable} != {expect_num_movable}")
    if expect_k is not None and out.k != int(expect_k):
        raise ValueError(f"{path}: k {out.k} != {expect_k}")
    if require_nonempty:
        counts = np.bincount(out.part, minlength=out.k)
        empty = np.flatnonzero(counts == 0).tolist()
        if empty:
            # PlaceDB.calc_num_filler_for_fence_region takes np.percentile of
            # an empty movable-size slice (PlaceDB.py:687); under the installed
            # numpy (1.26.4) that raises `IndexError: index -1 is out of bounds
            # for axis 0 with size 0` right there, inside initialize() -- it
            # never reaches the int(round(nan)) at PlaceDB.py:728. Verified on
            # this host 2026-09-19 (pre-flight E-3); the older "silently yields
            # NaN then ValueError at :729" wording came from
            # run_placement_two_stage.py's comment block and is wrong. Refuse
            # the membership here with a readable message either way.
            raise ValueError(f"{path}: empty regions {empty} would crash "
                             "PlaceDB.calc_num_filler_for_fence_region")
    return out


def _require_exact_fields(record, fields, what, extra_allowed=()):
    """Require record's keys to match `fields` exactly (order-independent),
    rejecting both a missing declared field and a key that was never
    declared -- a stray/renamed key must not round-trip silently. `fields`
    grows in later plans; `extra_allowed` carries the handful of keys a
    writer stamps itself after this check runs (schema_version for
    producer.json, per the C-2 asymmetry) so a load of that same file does
    not then reject its own stamp."""
    allowed = set(fields) | set(extra_allowed)
    record_set = set(record)
    missing = [name for name in fields if name not in record_set]
    unexpected = sorted(record_set - allowed)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing field(s): {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected field(s): {', '.join(unexpected)}")
        raise ValueError(f"{what}: {'; '.join(parts)}")


def save_freeze(path, record):
    _require_exact_fields(record, FREEZE_FIELDS, "freeze.json")
    if int(record["schema_version"]) != FREEZE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {FREEZE_SCHEMA_VERSION}")
    _atomic_write_json(path, record)


def load_freeze(path):
    with open(path) as stream:
        record = json.load(stream)
    _require_exact_fields(record, FREEZE_FIELDS, f"{path}")
    if int(record["schema_version"]) != FREEZE_SCHEMA_VERSION:
        raise ValueError(f"unsupported freeze schema in {path}")
    return record


def save_result(path, record):
    _require_exact_fields(record, MAIN_FLOW_RESULT_FIELDS, "result.json")
    _atomic_write_json(path, record)


def save_producer_json(path, payload):
    """producer.json -- the region producer's own run record (P-C Task 10).

    Unlike save_freeze/save_result this writer stamps schema_version itself:
    run_region_producer.run_producer builds the payload as one dict literal and
    also returns it to its caller, so the version belongs to the writer.
    """
    _require_exact_fields(payload, PRODUCER_FIELDS, "producer.json")
    record = dict(payload)
    record["schema_version"] = PRODUCER_SCHEMA_VERSION
    _atomic_write_json(path, record)


def load_producer_json(path):
    with open(path) as stream:
        record = json.load(stream)
    _require_exact_fields(record, PRODUCER_FIELDS, f"{path}",
                          extra_allowed=("schema_version",))
    if int(record["schema_version"]) != PRODUCER_SCHEMA_VERSION:
        raise ValueError(f"unsupported producer schema in {path}")
    return record


def scaled_region_set(rs, shift_factor, scale_factor):
    """Native -> scaled RegionSet, using PlaceDB.scale()'s own transform
    (PlaceDB.py:184-196: subtract the shift, multiply by the scale, applied to
    both corners of every rect). The lattice count is unchanged, so an input
    that passes validate() still passes it afterwards."""
    shift = np.asarray([shift_factor[0], shift_factor[1],
                        shift_factor[0], shift_factor[1]], dtype=np.float64)
    scale = float(scale_factor)
    regions = [RegionSpec(r.name,
                          (np.asarray(r.rects, dtype=np.float64).reshape(-1, 4) - shift) * scale)
               for r in rs.regions]
    xl, yl, xh, yh = rs.die
    die = ((xl - shift_factor[0]) * scale, (yl - shift_factor[1]) * scale,
           (xh - shift_factor[0]) * scale, (yh - shift_factor[1]) * scale)
    return RegionSet(die=die, lattice=rs.lattice, regions=regions)
