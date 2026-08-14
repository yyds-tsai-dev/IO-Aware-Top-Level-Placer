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

    This rewrites `.nets` in full (read all lines, append, rewrite) rather
    than streaming -- unlike `tile_bookshelf.tile`, this module is not
    required to be O(1)-memory (sec 3.2 attributes the streaming budget to
    the tiler specifically); at the toy/mechanism scale this task tests,
    the file fits trivially in memory.
    """
    nets_path = dst_prefix + ".nets"
    with open(nets_path) as f:
        lines = f.readlines()

    header_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumNets"))
    pins_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("NumPins"))
    n_nets_old = int(lines[header_idx].split(":")[1])
    n_pins_old = int(lines[pins_idx].split(":")[1])

    new_blocks = []
    n_pins_added = 0
    for k, net in enumerate(glue_nets):
        pins = net["pins"]
        name = f"{net_name_prefix}{k}"
        new_blocks.append(f"NetDegree : {len(pins)} {name}\n")
        for tile, node in pins:
            i, j = tile
            new_blocks.append(f"    t{i}_{j}/{node} I : 0 0\n")
            n_pins_added += 1
        new_blocks.append("")

    n_nets_new = n_nets_old + len(glue_nets)
    n_pins_new = n_pins_old + n_pins_added
    lines[header_idx] = f"NumNets : {n_nets_new}\n"
    lines[pins_idx] = f"NumPins : {n_pins_new}\n"
    lines.extend(l + "\n" if l and not l.endswith("\n") else l for l in new_blocks if l)

    with open(nets_path, "w") as f:
        f.writelines(lines)

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
