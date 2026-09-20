"""Portable evaluator evidence, aligned by net name and exact region geometry.

The NPZ is deliberately separate from the placement snapshot: an old snapshot
containing only x/y is not evidence of per-net evaluation. No pickle is used.
"""
import hashlib
import json
import os
import tempfile

import numpy as np

from ioplace.straddle import STRADDLE_SCALARS


SCHEMA_VERSION = 2
# v2 P-F: schema 2 adds the sec 7 straddle block. Schema 1 archives stay
# readable -- results/ holds historical evidence the route-calibration tooling
# still pairs against, and a diagnostics addition is no reason to orphan it.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
PER_NET_FIELDS = (
    "per_net_crossings", "per_net_ft", "per_net_lambda",
    "per_net_steiner", "per_net_home",
)


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def region_digest(rg):
    return array_digest(np.asarray(rg.die, dtype=np.float64), rg.grid)


def normalized_names(names):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v)
                       for v in names], dtype=np.str_)


def boundary_statistics(rg, demand):
    lengths = {}
    for a, b, unit in ((rg.grid[:, :-1], rg.grid[:, 1:], rg.cell_h),
                        (rg.grid[:-1, :], rg.grid[1:, :], rg.cell_w)):
        different = a != b
        keys = np.minimum(a[different], b[different]).astype(np.int64)*rg.k + np.maximum(a[different], b[different])
        unique, counts = np.unique(keys, return_counts=True)
        for key, count in zip(unique, counts):
            pair = (int(key//rg.k), int(key%rg.k))
            lengths[pair] = lengths.get(pair, 0.) + int(count)*unit
    pairs = sorted(set(lengths) | set(demand))
    values = np.asarray([demand.get(pair, 0) for pair in pairs], dtype=float)
    lens = np.asarray([lengths.get(pair, 0.) for pair in pairs], dtype=float)
    if np.any((values > 0) & (lens <= 0)):
        raise ValueError("positive demand on a non-adjacent region pair")
    def stats(v):
        n = len(v)
        total = float(v.sum())
        ordered = np.sort(v)
        gini = float(np.sum((2*np.arange(1,n+1)-n-1)*ordered)/(n*total)) if n and total else 0.
        return dict(n=n, total=total, max=float(v.max()) if n else 0.,
                    mean=float(v.mean()) if n else 0., p90=float(np.percentile(v, 90)) if n else 0., gini=gini)
    per_length = np.divide(values, lens, out=np.zeros_like(values), where=lens > 0)
    return pairs, values, lens, dict(boundary_demand=stats(values),
                                    boundary_demand_per_length=stats(per_length),
                                    boundary_length_units="evaluator_coordinates")


def save_evaluation(path, nl, rg, result, node_x, node_y, net_names, *,
                    max_degree=256, provenance=None):
    """Persist an already computed result and its pin-region membership.

    Coordinates are in evaluator units (after PlaceDB scaling, when used by a
    driver). The net order and lattice fingerprints prevent accidental pairing
    across K, netlist versions, or independent placements.
    """
    names = normalized_names(net_names)
    if names.shape != (nl.num_nets,) or len(set(names)) != nl.num_nets:
        raise ValueError("net names must be unique and aligned to all nets")
    x = np.asarray(node_x[:nl.num_physical], dtype=np.float64)
    y = np.asarray(node_y[:nl.num_physical], dtype=np.float64)
    arrays = {key: np.asarray(getattr(result, key)) for key in PER_NET_FIELDS}
    if any(value.shape != (nl.num_nets,) for value in arrays.values()):
        raise ValueError("evaluator per-net arrays must cover the entire netlist")
    pairs, demand, lengths, boundary_stats = boundary_statistics(rg, result.boundary_pair_demand)
    arrays.update(
        net_names=names,
        net_degrees=np.asarray(nl.net_degrees, dtype=np.int32),
        pin_region_mask=rg.pin_region_bitmask(nl, x, y),
        node_region=rg.region_of_points(x, y),
        boundary_pairs=np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        boundary_demand=demand.astype(np.int64), boundary_length=lengths,
    )
    # v2 P-F (design sec 7). Present iff the evaluator actually computed them,
    # so "not measured" and "measured as zero" stay distinguishable.
    straddle = None
    if getattr(result, "per_node_straddle", None) is not None:
        node_straddle = np.asarray(result.per_node_straddle, dtype=np.uint8)
        pin_split = np.asarray(result.per_net_pin_split, dtype=np.int32)
        if node_straddle.shape != (nl.num_physical,):
            raise ValueError("per_node_straddle must cover every physical node")
        if pin_split.shape != (nl.num_nets,):
            raise ValueError("per_net_pin_split must cover the entire netlist")
        arrays.update(per_node_straddle=node_straddle, per_net_pin_split=pin_split)
        straddle = {key: getattr(result, key) for key in STRADDLE_SCALARS}
        straddle = {key: (int(value) if isinstance(value, (int, np.integer))
                          else float(value)) for key, value in straddle.items()}
        straddle["anchor"] = "center"
        straddle["box"] = "closed_four_corner"
    metadata = dict(
        schema_version=SCHEMA_VERSION, num_nets=nl.num_nets,
        num_physical=nl.num_physical, num_movable=nl.num_movable,
        k=rg.k, max_degree=int(max_degree), region_sha256=region_digest(rg),
        placement_sha256=array_digest(x, y),
        net_order_sha256=array_digest(names),
        node_region_convention="lower_left_position",
        totals={key: getattr(result, key) for key in
                ("io_count", "ft_count", "hard_lambda_sum", "io_rg", "ft_rg",
                 "tree_wl", "hpwl", "large_net_lb")},
        provenance=provenance or {},
        straddle=straddle,
        **boundary_stats,
    )
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".evaluation-", suffix=".npz", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def load_evaluation(path, *, net_names=None, rg=None):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data.pop("metadata")))
    if metadata.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported evaluator evidence schema")
    n = metadata["num_nets"]
    for key in (*PER_NET_FIELDS, "net_degrees", "pin_region_mask", "net_names"):
        if key not in data or data[key].shape != (n,):
            raise ValueError(f"invalid evaluator evidence array: {key}")
    if array_digest(data["net_names"]) != metadata["net_order_sha256"]:
        raise ValueError("evaluator net-order fingerprint mismatch")
    for key, total in (("per_net_crossings", "io_count"),
                       ("per_net_ft", "ft_count"), ("per_net_steiner", "io_rg")):
        if int(data[key].sum(dtype=np.int64)) != metadata["totals"][total]:
            raise ValueError(f"evaluator total mismatch: {key}")
    if net_names is not None and not np.array_equal(normalized_names(net_names), data["net_names"]):
        raise ValueError("evaluator and route net-name orders differ")
    if rg is not None and region_digest(rg) != metadata["region_sha256"]:
        raise ValueError("evaluator and route region geometries differ")
    straddle = metadata.get("straddle")
    if straddle is not None:
        for key, shape in (("per_node_straddle", (metadata["num_physical"],)),
                           ("per_net_pin_split", (n,))):
            if key not in data or data[key].shape != shape:
                raise ValueError(f"invalid evaluator evidence array: {key}")
        if int(data["per_node_straddle"].sum(dtype=np.int64)) != straddle["straddle_cells"]:
            raise ValueError("evaluator total mismatch: per_node_straddle")
        if int((data["per_net_pin_split"] > 0).sum()) != straddle["straddle_pin_split_nets"]:
            raise ValueError("evaluator total mismatch: per_net_pin_split")
    data["metadata"] = metadata
    return data


def pin_regions_from_evaluation(data):
    """Return all nets, including empty sets, so route FT has no missing sentinel."""
    k = data["metadata"]["k"]
    return {i: [r for r in range(k) if int(mask) & (1 << r)]
            for i, mask in enumerate(data["pin_region_mask"])}
