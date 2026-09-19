"""M4 T4 glue-net generator (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 3.2,
v2.1's power-law kernel + closed-form method-of-moments rewrite of Codex
finding #5).

Distance kernel (single parameter, defined for every integer distance):

    phi(d; alpha) = d^(-alpha),  phi(1) = 1
    E[glue nets between tiles u,v] = lambda_0 * phi(dist(u,v); alpha)

`(lambda_0, alpha)` are solved by **closed-form** method-of-moments from the
observed per-distance-level average glue-net counts -- no grid search:

  2x2 arrangement (d=1 four adjacent pairs, d=sqrt(2) two diagonal pairs):
      alpha = log(lambda_adj / lambda_diag) / log(sqrt(2))
      lambda_0 = lambda_adj                              (phi(1) = 1)

  1x4 arrangement (d=1 three pairs, d=2 two pairs, d=3 one pair -- held out):
      alpha = log(lambda_1 / lambda_2) / log(2)
      lambda_0 = lambda_1
      d=3 is a *hold-out*: z = (lambda_3_obs - lambda_1 * 3**(-alpha))
                               / sqrt(lambda_1 * 3**(-alpha))   (Poisson scale)
      pre-registered threshold |z| <= 2 (sec 3.2's "identifiability ledger")

Degenerate cases (sec 3.2, pre-registered, no ad hoc handling at call
time):
  (a) a zero-count distance level -> can't take a log -> `truncated_at_d`
  (b) alpha < 0 (farther pairs denser than nearer ones) -> contradicts the
      power-law assumption -> `kernel_rejected`

This module is the *mechanism* only (sec 3.2's own scope note for this
task): the statistical inputs (`lambda_adj`, `lambda_diag`, `deg_hist`,
`pinshare`, `iface_dist`) are products of T3a/`cluster_stats.py`, which do
not exist yet -- every test here is against synthetic, hand-constructed
statistics, not a real `mempool_cluster`.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
import shutil

import numpy as np

from ioplace.bench import tile_bookshelf as tb


def phi(d, alpha):
    """d^(-alpha), defined (== 1.0) at d=1 for any alpha, and finite for
    every d > 0 -- this is the whole point of the v2.1 kernel rewrite
    (the old gamma_far kernel left phi(sqrt(5))/phi(sqrt(8)) undefined)."""
    return float(d) ** (-float(alpha))


@dataclass
class MomResult:
    lambda_0: float
    alpha: float
    status: str  # "ok" | "truncated_at_d" | "kernel_rejected"
    detail: dict


def solve_mom_2x2(lambda_adj, lambda_diag):
    """2x2 arrangement: d=1 (4 adjacent pairs), d=sqrt(2) (2 diagonal
    pairs). Exactly identified (2 moments, 2 parameters) -> 0 residual DoF;
    sec 3.2 is explicit this cannot validate the kernel *shape*, only
    reproduce the two numbers it was fed."""
    if lambda_adj <= 0 or lambda_diag <= 0:
        # can't take a log of a non-positive count -> truncate, don't guess
        d_max = math.sqrt(2) if lambda_adj > 0 else 1.0
        lam_max = lambda_adj if lambda_adj > 0 else 0.0
        return MomResult(
            lambda_0=float("nan"), alpha=float("nan"), status="truncated_at_d",
            detail={"truncated_at_d": d_max, "lambda_adj": lambda_adj, "lambda_diag": lambda_diag,
                    "alpha_lower_bound": None if lam_max <= 0 else None})
    alpha = math.log(lambda_adj / lambda_diag) / math.log(math.sqrt(2))
    if alpha < 0:
        return MomResult(
            lambda_0=lambda_adj, alpha=alpha, status="kernel_rejected",
            detail={"lambda_adj": lambda_adj, "lambda_diag": lambda_diag,
                    "reason": "alpha<0: farther pairs denser than nearer ones"})
    return MomResult(lambda_0=lambda_adj, alpha=alpha, status="ok",
                      detail={"lambda_adj": lambda_adj, "lambda_diag": lambda_diag})


def solve_mom_1x4(lambda_1, lambda_2, lambda_3_obs=None):
    """1x4 arrangement: fit on d=1 (3 pairs) and d=2 (2 pairs); d=3 (1
    pair) is a hold-out lack-of-fit check, not part of the fit. Returns the
    fitted MomResult; if `lambda_3_obs` is given, `detail["holdout"]`
    carries the pre-registered z-test (|z| <= 2)."""
    if lambda_1 <= 0 or lambda_2 <= 0:
        return MomResult(
            lambda_0=float("nan"), alpha=float("nan"), status="truncated_at_d",
            detail={"truncated_at_d": 2.0 if lambda_1 > 0 else 1.0,
                    "lambda_1": lambda_1, "lambda_2": lambda_2})
    alpha = math.log(lambda_1 / lambda_2) / math.log(2.0)
    detail = {"lambda_1": lambda_1, "lambda_2": lambda_2}
    status = "ok"
    if alpha < 0:
        status = "kernel_rejected"
        detail["reason"] = "alpha<0: farther pairs denser than nearer ones"
    if lambda_3_obs is not None:
        lambda_3_hat = lambda_1 * phi(3, alpha) if status == "ok" else None
        z = None
        if lambda_3_hat is not None and lambda_3_hat > 0:
            z = (lambda_3_obs - lambda_3_hat) / math.sqrt(lambda_3_hat)
        detail["holdout"] = {
            "lambda_3_obs": lambda_3_obs, "lambda_3_hat": lambda_3_hat,
            "z": z, "z_threshold": 2.0,
            "pass": (z is not None and abs(z) <= 2.0),
        }
    return MomResult(lambda_0=lambda_1, alpha=alpha, status=status, detail=detail)


# ---------------------------------------------------------------------------
# Bootstrap CI over seeds (sec 3.3 fit metrics: "lambda_0, alpha (>=5 seeds
# bootstrap 95% CI)")
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_resamples=2000, seed=0, alpha_level=0.05):
    """Percentile bootstrap 95% CI (default) for the mean of `values`."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo, hi = np.quantile(means, [alpha_level / 2, 1 - alpha_level / 2])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


# ---------------------------------------------------------------------------
# Cross-tile pair distances / expected counts (arithmetic core reused by
# H5'/`verify_bench.check_h5prime_self_consistency`)
# ---------------------------------------------------------------------------

def tile_pair_distances(R, C):
    """{(tile_a, tile_b): chebyshev-free euclidean grid distance} for every
    unordered pair of distinct tiles in an R x C array (tile = (i, j))."""
    tiles = [(i, j) for i in range(R) for j in range(C)]
    out = {}
    for a in range(len(tiles)):
        for b in range(a + 1, len(tiles)):
            (i1, j1), (i2, j2) = tiles[a], tiles[b]
            d = math.hypot(i1 - i2, j1 - j2)
            out[(tiles[a], tiles[b])] = d
    return out


def expected_glue_total(lambda_0, alpha, R, C):
    """C(m) = sum_{u<v} lambda_0 * phi(dist(u,v), alpha) -- the closed-form
    total this module's sampler is built to reproduce (sec 3.3 H5')."""
    dists = tile_pair_distances(R, C)
    return sum(lambda_0 * phi(d, alpha) for d in dists.values())


# ---------------------------------------------------------------------------
# N2 normalization (2026-08-14 T6 adjudication doc sec 2d): fixed per-tile
# terminal budget, redistributed by phi(d), instead of N1's raw per-pair
# lambda_0*phi(d) (sec 2d: N1 "會讓每 tile 端子隨 m 成長、違反 Rent" as the
# array grows -- a tile's total glue-terminal count sums over every other
# tile in the array, so it scales with the number of tiles under N1 but is
# held fixed at `budget_pairs * lambda_0` under N2).
# ---------------------------------------------------------------------------

def n2_pair_counts(lambda_0, alpha, R, C, budget_pairs=3.0):
    """N2 expected glue-net count per unordered tile pair: every tile gets
    the *same fixed* budget `B = budget_pairs * lambda_0` (sec 2d's "B=3
    lambda_0", anchored to the 2x2 arrangement's 3-neighbors-per-tile, used
    as a constant reference budget for every array shape -- not
    re-derived from each shape's own neighbor count, which is the whole
    point: N1 and N2 are meant to disagree more as shapes get more/fewer
    neighbors than 2x2's 3), split among that tile's neighbors
    proportional to `phi(dist, alpha)`:

        c(u,v) = B * phi(dist(u,v), alpha) / Z_u,   Z_u = sum_{w!=u} phi(dist(u,w), alpha)

    Only defined here for arrangements where every tile has the *same*
    neighbor-distance multiset, so `Z_u` doesn't depend on `u` and `c(u,v)`
    computed from either endpoint's budget agrees (unambiguous). True of
    every shape M4 T6 uses (1xC/Rx1, 2x2); a general asymmetric R*C array
    (e.g. 3x3, where corner and center tiles have different neighbor-
    distance multisets) needs a different rule this function does not
    implement -- `NotImplementedError` rather than silently picking one."""
    dists = tile_pair_distances(R, C)
    weights = {pair: phi(d, alpha) for pair, d in dists.items()}
    z = {}
    for (a, b), w in weights.items():
        z[a] = z.get(a, 0.0) + w
        z[b] = z.get(b, 0.0) + w
    distinct_z = {round(v, 9) for v in z.values()}
    if len(distinct_z) != 1:
        raise NotImplementedError(
            f"n2_pair_counts only supports arrangements where every tile has an identical "
            f"neighbor-distance multiset (T6 scope: 1xC/Rx1 and 2x2); R={R} C={C} has "
            f"{len(distinct_z)} distinct per-tile normalizers")
    z_u = next(iter(distinct_z))
    budget = budget_pairs * lambda_0
    return {pair: budget * w / z_u for pair, w in weights.items()}


# ---------------------------------------------------------------------------
# N2 generalized to asymmetric arrangements (2026-08-15 T6 holdout
# adjudication doc sec 5.2/8-2): `n2_pair_counts` only works where every
# tile has the *same* neighbor-distance multiset (Z_u identical for all u),
# because it derives `a_u` in closed form from that single shared Z. A 3x3
# array's corner/edge/center tiles each have a different multiset -> no
# closed form. Symmetric Sinkhorn scaling solves the same per-tile-budget
# constraint for the general case: find `a_u > 0` such that
#
#     c(u,v) = a_u * a_v * phi(dist(u,v), alpha)   satisfies
#     sum_{v!=u} c(u,v) = B = budget_pairs * lambda_0   for every tile u.
#
# (Adjudication sec 5.2's rejected alternative (A), "two-endpoint-average"
# closed form `c(u,v) = (B/2)(phi/Z_u + phi/Z_v)`, does *not* hold every
# tile's terminal count to exactly B -- it drifts up to +-21% across
# corner/edge/center at alpha=1.915 -- which defeats N2's entire reason for
# existing over N1 (sec 2d): "a tile's glue-terminal count is a property of
# the tile, not of its shape/neighbor count". Sinkhorn holds every tile to
# *exactly* B by construction, so it's the decided rule; (A) is not
# implemented here.)
# ---------------------------------------------------------------------------

def _phi_power_law(d, alpha):
    """K1 (design draft sec 3.2 line 301, main line): same as `phi` above."""
    return phi(d, alpha)


def _phi_exp(d, alpha):
    """K2 (design draft sec 3.2 line 302): exp(-alpha*(d-1)) -- phi(1)=1
    like every other kernel here; `alpha` here plays the role of the
    design draft's separate `alpha'` (this module's single-alpha interface
    reuses the same parameter slot for whichever kernel is selected)."""
    return math.exp(-float(alpha) * (float(d) - 1.0))


def _phi_truncated(d, alpha):
    """K3 (design draft sec 3.2 line 303): "explicit non-simulation of
    long-range connectivity" -- the same power-law shape as K1 at the only
    distances any tile in T6's shapes (1xC/Rx1/2x2/3x3) has at
    d<=sqrt(2) (its immediate 4- and 8-neighbors), zero beyond."""
    if d > math.sqrt(2) + 1e-9:
        return 0.0
    return phi(d, alpha)


_N2_SINKHORN_KERNELS = {
    "power_law": _phi_power_law,
    "exp": _phi_exp,
    "truncated": _phi_truncated,
}


def n2_pair_counts_sinkhorn(lambda_0, alpha, R, C, budget_pairs=3.0, tol=1e-9,
                             max_iter=1000, kernel="power_law"):
    """General-shape N2: symmetric Sinkhorn scaling (adjudication sec 5.2,
    decided over the closed-form-average alternative -- see module note
    above). Solves for per-tile scale factors `a_u > 0` with Gauss-Seidel,
    updated **in place** (each `a_u` update immediately sees the previous
    tile's *new* value, not last iteration's) in fixed row-major tile order
    `(i, j) for i in range(R) for j in range(C)` (sec 5.2's "執行細節"):

        a_u <- B / sum_{v!=u} a_v * w_uv,   w_uv = kernel(dist(u,v), alpha)

    Convergence: `max_u |sum_v c(u,v) - B| / B < tol` (checked once per
    full sweep over all tiles), capped at `max_iter` sweeps. `a_u` is
    initialized to 1.0 for every tile (the plain Gauss-Seidel starting
    point -- nothing in sec 5.2 specifies a particular init, and this is
    the least-assuming choice); note that on every shape this module's N2
    caller actually uses (1xC/Rx1/2x2 -- uniform Z_u), the fixed point this
    converges to is *identical* to `n2_pair_counts`'s closed form (sec
    5.2's "在 1×C/R×1/2×2 上與現行公式逐位元等價" -- provable directly: with
    Z_u equal for every tile, `a_u = sqrt(B/Z)` for all u is already a
    fixed point of the update rule above).

    Returns `(pair_counts, diagnostics)`: `pair_counts` is
    `{(tile_a, tile_b): count}` exactly like `n2_pair_counts`;
    `diagnostics = {"n2_iters": n, "n2_max_rel_dev": dev}` (sec 5.2:
    manifest fields `n2_rule="sinkhorn"`/`n2_iters`/`n2_max_rel_dev`)."""
    try:
        weight_fn = _N2_SINKHORN_KERNELS[kernel]
    except KeyError:
        raise ValueError(f"unknown kernel {kernel!r}; choices are {sorted(_N2_SINKHORN_KERNELS)}")

    dists = tile_pair_distances(R, C)
    tiles = [(i, j) for i in range(R) for j in range(C)]
    neighbors = {t: [] for t in tiles}
    for (tile_a, tile_b), d in dists.items():
        w = weight_fn(d, alpha)
        neighbors[tile_a].append((tile_b, w))
        neighbors[tile_b].append((tile_a, w))
    for u in tiles:
        if sum(w for _, w in neighbors[u]) <= 0:
            raise ValueError(f"tile {u} has no positive-weight neighbor under kernel={kernel!r}, "
                              f"alpha={alpha} -- Sinkhorn budget cannot be met")

    budget = budget_pairs * lambda_0
    a = {u: 1.0 for u in tiles}
    n_iters, max_rel_dev = 0, float("inf")
    for it in range(1, max_iter + 1):
        for u in tiles:
            s = sum(a[v] * w for v, w in neighbors[u])
            a[u] = budget / s
        max_rel_dev = 0.0
        for u in tiles:
            s = sum(a[u] * a[v] * w for v, w in neighbors[u])
            max_rel_dev = max(max_rel_dev, abs(s - budget) / budget)
        n_iters = it
        if max_rel_dev < tol:
            break

    pair_counts = {pair: a[pair[0]] * a[pair[1]] * weight_fn(d, alpha) for pair, d in dists.items()}
    return pair_counts, {"n2_iters": n_iters, "n2_max_rel_dev": max_rel_dev}


def sample_glue_net_count_from_expected(rng, expected_per_pair, mode="poisson"):
    """Same sampling step as `sample_glue_net_count`, generalized to accept
    a precomputed `{(tile_a,tile_b): expected_count}` map -- N1 keeps using
    `sample_glue_net_count` (its expected counts are `lambda_0*phi(d)`
    directly); N2 goes through `n2_pair_counts` then this."""
    if mode not in ("poisson", "expected"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "poisson":
        return {pair: int(rng.poisson(lam)) for pair, lam in expected_per_pair.items()}
    return {pair: int(round(lam)) for pair, lam in expected_per_pair.items()}


# ---------------------------------------------------------------------------
# Sampling actual glue nets (mechanism only -- deg_hist/pinshare/iface_dist
# are toy/synthetic in this task's tests, real ones are T3a/T6 products)
# ---------------------------------------------------------------------------

def sample_glue_net_count(rng, lambda_0, alpha, R, C, mode="poisson"):
    """Per-unordered-tile-pair glue-net count from its expected value
    lambda_0 * phi(dist). Returns {(tile_a, tile_b): k}.

    mode="poisson" (default): a stochastic Poisson draw per pair -- this
        is what preserves the per-pair anisotropy/overdispersion sec 3.2
        says must be kept and reported (a deterministic count would hide
        exactly that variation), and what >=5-seed bootstrap CIs are run
        over.
    mode="expected": the deterministic round(lambda_0 * phi(dist)) per
        pair, with no rng draw. This is the mode `verify_bench.py`'s H5'
        self-check is defined against (sec 3.3: "this is the construction
        method's arithmetic identity" -- an aggregate of independent
        Poisson draws is only approximately equal to its own expectation,
        not exact to the spec's 0.1% self-check tolerance, but the
        rounded expectation is deterministic and matches by construction).
    """
    if mode not in ("poisson", "expected"):
        raise ValueError(f"unknown mode {mode!r}")
    dists = tile_pair_distances(R, C)
    if mode == "poisson":
        return {pair: int(rng.poisson(lambda_0 * phi(d, alpha))) for pair, d in dists.items()}
    return {pair: int(round(lambda_0 * phi(d, alpha))) for pair, d in dists.items()}


def sample_glue_nets(rng, counts_per_pair, candidate_nodes, deg_hist=(2,), deg_weights=None):
    """Build concrete 2-tile glue net specs.

    candidate_nodes: {(i, j): [node_name, ...]} -- eligible interface-cell
        node names *within that tile's own bookshelf, without the t{i}_{j}/
        prefix* (in production these come from the cluster's iface_dist
        sample; here the tests supply a toy pool directly). Each glue net
        picks one candidate from each of its two tiles (with replacement
        across nets -- a node fans out to more than one glue net exactly
        as a real interface cell would).
    deg_hist / deg_weights: degree distribution glue nets are drawn from;
        every degree in `deg_hist` must be exactly 2 in this task's scope
        (bipartite tile-pair nets) -- higher-degree multi-tile glue nets
        are a T6/T7 concern once real deg_hist data exists.

    Returns a list of dicts: {"pins": [(tile, node_name), (tile, node_name)]}.
    """
    if any(d != 2 for d in deg_hist):
        raise NotImplementedError("only degree-2 (bipartite tile-pair) glue nets are implemented; "
                                   "deg_hist with degree != 2 needs real T6/T7 pinshare data")
    nets = []
    for (tile_a, tile_b), k in counts_per_pair.items():
        pool_a, pool_b = candidate_nodes.get(tile_a), candidate_nodes.get(tile_b)
        if k > 0 and (not pool_a or not pool_b):
            raise ValueError(f"no candidate interface nodes supplied for pair {(tile_a, tile_b)}")
        for _ in range(k):
            na = pool_a[rng.integers(0, len(pool_a))]
            nb = pool_b[rng.integers(0, len(pool_b))]
            nets.append({"pins": [(tile_a, na), (tile_b, nb)]})
    return nets


def append_glue_nets(dst_prefix, glue_nets, lambda_0, alpha, kernel="power_law", seed=None, net_name_prefix="glue"):
    """Append `glue_nets` (as returned by `sample_glue_nets`) to
    `<dst_prefix>.nets`, rewrite its NumNets/NumPins header, and update
    `<dst_prefix>.manifest.json`'s "glue" section with the *actual* counts
    (never left at 0/unset once this is called). Returns (n_nets, n_pins).

    **M4 T6 scale fix (2026-08-15):** this used to `readlines()`/
    `writelines()` the whole `.nets` file ("at the toy/mechanism scale this
    task tests, the file fits trivially in memory" -- true then, not at
    T6's actual scale: a 2x2 array's `.nets` is multiple GB, and holding it
    as a list of ~10^7-10^8 Python `str` objects multiplies its on-disk
    footprint several-fold in RAM). Streams instead: the header is fixed as
    it's copied (`NumNets`/`NumPins` lines rewritten in place, everything
    else passed through line by line), the unchanged body is moved with
    `shutil.copyfileobj` (no per-line Python object churn), and the new
    glue-net blocks are appended last -- peak memory is O(1) in file size,
    same shape as `export_bookshelf.py`'s `_fix_nets_file`.
    """
    nets_path = dst_prefix + ".nets"
    tmp_path = nets_path + ".tmp"
    n_pins_added = sum(len(net["pins"]) for net in glue_nets)

    with open(nets_path, "r") as fin, open(tmp_path, "w") as fout:
        n_nets_old = n_pins_old = None
        while n_pins_old is None:
            line = fin.readline()
            s = line.strip()
            if s.startswith("NumNets"):
                n_nets_old = int(s.split(":")[1])
                continue  # rewritten below, once n_nets_old+len(glue_nets) is known
            if s.startswith("NumPins"):
                n_pins_old = int(s.split(":")[1])
                fout.write(f"NumNets : {n_nets_old + len(glue_nets)}\n")
                fout.write(f"NumPins : {n_pins_old + n_pins_added}\n")
                continue
            fout.write(line)
        shutil.copyfileobj(fin, fout)

        for k, net in enumerate(glue_nets):
            pins = net["pins"]
            name = f"{net_name_prefix}{k}"
            fout.write(f"NetDegree : {len(pins)} {name}\n")
            for tile, node in pins:
                i, j = tile
                fout.write(f"    t{i}_{j}/{node} I : 0 0\n")

    os.replace(tmp_path, nets_path)

    manifest_path = dst_prefix + ".manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["glue"] = {
        "n_nets": len(glue_nets), "n_pins": n_pins_added,
        "kernel": kernel, "lambda_0": lambda_0, "alpha": alpha, "seed": seed,
    }
    manifest["output_sha256"]["nets"] = tb.sha256_file(nets_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)

    return len(glue_nets), n_pins_added
