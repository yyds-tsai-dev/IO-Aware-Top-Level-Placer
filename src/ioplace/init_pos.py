"""Initial movable-cell positions for the v2 main flow (design v2 sec 3, arms (a)-(d)).

All three modes write `placedb.node_x`/`node_y` in **native post-read units**
and must run after `placedb.read(params)` and before
`placedb.initialize(params)`: `initialize()`'s `scale()` converts whatever is
in those arrays into the internal frame (PlaceDB.py:151-196), and
`BasicPlace.__init__` copies them into `init_pos` (BasicPlace.py:269-288).

`random_center_init_flag` is the switch that decides whether DREAMPlace
overwrites the movable slice with its own die-centre Gaussian
(BasicPlace.py:272-277/283-288): `die_center` leaves it at 1, the two
explicit modes set it to 0.
"""
import numpy as np

INIT_MODES = ("die_center", "region_center", "seed")

# BasicPlace.py:276/287 -- DREAMPlace's own centre-init noise, as a fraction of
# the die span. Reused verbatim so `region_center` differs from `die_center`
# only in the mean, never in the dispersion.
NOISE_FRACTION = 0.001


def region_centers(rs):
    """(k, 2) area-weighted centre of each region's rects, native units.

    Same formula as run_placement_two_stage.assign_blocks_to_regions'
    geometry step (run_placement_two_stage.py:112-119) -- a RegionSpec may
    hold several rects once the producer (P-C) emits rectilinear regions.
    """
    centers = np.zeros((rs.k, 2), dtype=np.float64)
    for rid, region in enumerate(rs.regions):
        rects = np.asarray(region.rects, dtype=np.float64).reshape(-1, 4)
        area = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
        cx = (rects[:, 0] + rects[:, 2]) * 0.5
        cy = (rects[:, 1] + rects[:, 3]) * 0.5
        total = area.sum()
        if total <= 0:
            raise ValueError(f"region {region.name!r} has zero total rect area")
        centers[rid] = (np.sum(cx * area) / total, np.sum(cy * area) / total)
    return centers


def apply_init(placedb, params, mode, *, region_set=None, part=None,
               positions=None, rng_seed=0):
    """Install the initial positions for `mode`; return a JSON-able record."""
    if mode not in INIT_MODES:
        raise ValueError(f"mode must be one of {INIT_MODES}, got {mode!r}")
    m = placedb.num_movable_nodes
    n_phys = placedb.num_physical_nodes

    if mode == "die_center":
        params.random_center_init_flag = 1
        return {"mode": mode}

    params.random_center_init_flag = 0

    if mode == "region_center":
        if region_set is None or part is None:
            raise ValueError("region_center needs region_set and part")
        part = np.asarray(part, dtype=np.int64).reshape(-1)
        if len(part) != m:
            raise ValueError(f"part has {len(part)} entries, expected {m}")
        if part.min() < 0 or part.max() >= region_set.k:
            raise ValueError(f"part out of range [0, {region_set.k})")
        centers = region_centers(region_set)
        width = float(placedb.xh) - float(placedb.xl)
        height = float(placedb.yh) - float(placedb.yl)
        scale_x, scale_y = width * NOISE_FRACTION, height * NOISE_FRACTION
        rng = np.random.default_rng(int(rng_seed))
        size_x = np.asarray(placedb.node_size_x[:m], dtype=np.float64)
        size_y = np.asarray(placedb.node_size_y[:m], dtype=np.float64)
        if (size_x <= 0).any() or (size_y <= 0).any():
            raise ValueError("region_center needs positive node_size_x/node_size_y "
                             "for every movable node")
        # The *cell centre* lands on the region centre: the freeze membership
        # is the argmax at the cell centre (design v2 sec 3 phase 2), so
        # initialising the lower-left there -- what BasicPlace.py:272-277 does
        # for the die centre -- would start wide cells in a foreign region.
        x = centers[part, 0] - 0.5 * size_x + rng.normal(0.0, scale_x, size=m)
        y = centers[part, 1] - 0.5 * size_y + rng.normal(0.0, scale_y, size=m)
        x = np.clip(x, float(placedb.xl), float(placedb.xh) - size_x)
        y = np.clip(y, float(placedb.yl), float(placedb.yh) - size_y)
        placedb.node_x[:m] = x.astype(placedb.node_x.dtype)
        placedb.node_y[:m] = y.astype(placedb.node_y.dtype)
        return {"mode": mode, "rng_seed": int(rng_seed),
                "noise_scale_x": scale_x, "noise_scale_y": scale_y}

    if positions is None:
        raise ValueError("seed mode needs positions")
    if positions.num_physical != n_phys:
        raise ValueError(f"seed num_physical {positions.num_physical} != {n_phys}")
    fixed_x = np.asarray(placedb.node_x[m:n_phys], dtype=np.float64)
    fixed_y = np.asarray(placedb.node_y[m:n_phys], dtype=np.float64)
    span = max(float(placedb.xh) - float(placedb.xl),
               float(placedb.yh) - float(placedb.yl))
    tol = 1e-6 * span
    if (not np.allclose(positions.node_x[m:n_phys], fixed_x, atol=tol, rtol=0.0)
            or not np.allclose(positions.node_y[m:n_phys], fixed_y, atol=tol, rtol=0.0)):
        raise ValueError("seed disagrees with this design's fixed nodes "
                         "(terminals/macros must not move between phases)")
    placedb.node_x[:m] = np.asarray(positions.node_x[:m]).astype(placedb.node_x.dtype)
    placedb.node_y[:m] = np.asarray(positions.node_y[:m]).astype(placedb.node_y.dtype)
    return {"mode": mode, "placedb_sha256": positions.placedb_sha256,
            "source_kind": positions.kind}
