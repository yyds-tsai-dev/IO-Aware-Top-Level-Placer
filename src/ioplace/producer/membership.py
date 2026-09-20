"""The producer's membership prior.

GrandPlan takes RTL partition membership as an input and never re-partitions
(digest section 1). spec section 2 substitutes Mt-KaHyPar (K=16, epsilon=0.03)
or RTL hierarchy prefixes, run BEFORE the flat GP. The labels are fixed for the
whole producer run: the grouping loss, the extraction and membership.npz all
read the same array, which is also what arm (e) needs.

No block->region matching: spec section 2 explicitly rules out
run_placement_two_stage.assign_blocks_to_regions here, because the hulls supply
geometry rather than a permutation.
"""
import numpy as np

from ioplace.partition.mtkahypar_runner import partition_netlist


def mtkahypar_membership(nl, k, epsilon=0.03, seed=0, threads=8):
    """Mt-KaHyPar KM1 partition, truncated to the movable prefix.

    `threads` is only a request: mtkahypar_runtime honours
    IOPLACE_MTKAHYPAR_THREADS (src/scripts/env.sh defaults it to 1 because the
    installed 1.6.2 wheel crashes in parallel coarsening).
    """
    part = partition_netlist(nl, int(k), epsilon=float(epsilon), seed=int(seed),
                             threads=int(threads))
    return np.asarray(part[:nl.num_movable], dtype=np.int32)


def _prefix(name, depth):
    s = name.decode() if isinstance(name, (bytes, np.bytes_)) else str(name)
    parts = s.split("/")
    return "/".join(parts[:depth]) if len(parts) > depth else (
        "/".join(parts[:-1]) if len(parts) > 1 else "")


def hierarchy_membership(node_names, num_movable, k, depth=1):
    """Group movable cells by the first `depth` slash-separated name
    components, then pack the groups into k buckets longest-processing-time
    first (largest group into the currently lightest bucket; ties on the lowest
    bucket index). Fully deterministic."""
    m = int(num_movable)
    names = list(node_names)[:m]
    keys = [_prefix(n, depth) for n in names]
    uniq, inv = np.unique(np.array(keys, dtype=object), return_inverse=True)
    if len(uniq) < k:
        raise ValueError(
            f"hierarchy membership found {len(uniq)} prefix groups at depth "
            f"{depth}, fewer than k={k}; raise --hierarchy-depth or use "
            "--membership mtkahypar")
    sizes = np.bincount(inv, minlength=len(uniq))
    order = sorted(range(len(uniq)), key=lambda g: (-int(sizes[g]), str(uniq[g])))
    load = np.zeros(k, dtype=np.int64)
    group_bucket = np.zeros(len(uniq), dtype=np.int32)
    for g in order:
        b = int(np.argmin(load))            # np.argmin -> lowest index on ties
        group_bucket[g] = b
        load[b] += int(sizes[g])
    return group_bucket[inv].astype(np.int32)


def build_membership(source, *, nl, node_names, num_movable, k, epsilon=0.03,
                     seed=0, depth=1, threads=8):
    """Dispatch for the driver's --membership {mtkahypar,hierarchy}."""
    if source == "mtkahypar":
        return mtkahypar_membership(nl, k, epsilon=epsilon, seed=seed,
                                    threads=threads)
    if source == "hierarchy":
        return hierarchy_membership(node_names, num_movable, k, depth=depth)
    raise ValueError(
        f"unknown membership source {source!r}; expected 'mtkahypar' or 'hierarchy'")
