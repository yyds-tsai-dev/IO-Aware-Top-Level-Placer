import json
import os
import numpy as np
import pytest
from ioplace import artifacts
from ioplace.regions import RegionSet

DP = os.environ.get("DREAMPLACE_ROOT", "/ldaphome/yyds-tsai-dev/DREAMPlace")
CFG = os.path.join(DP, "install", "test", "simple.json")


class _StubTerm:
    """A one-line stand-in for GroupingTerm's forward(pos, lam) contract."""

    def forward(self, pos, lam):
        return float(lam) * 0.5 * (pos[:2] ** 2).sum()


def test_group_adapter_exposes_the_unweighted_term():
    """Ruling D5: TermNormalizer's term protocol is value(pos, ctx) returning
    the UNWEIGHTED objective, i.e. forward(pos, 1.0)."""
    import torch
    from ioplace.drivers.run_region_producer import _GroupAdapter
    pos = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert float(_GroupAdapter(_StubTerm()).value(pos, {})) == pytest.approx(2.5)


def test_normalizer_wiring_derives_eq3_and_tracks_the_refresh_contract():
    """Rulings D5/D6. The driver holds no schedule state; TermNormalizer does.
    ||grad WL||_1 = 20 and ||grad Group||_1 = 3 on this input, so ratio = 20/3,
    and Eq.3's wt0 = 0.05 gives lambda = 0.05 * 20/3. A transaction leaves the
    version pair out of sync until mark_refreshed() -- exactly what
    install_version_invariant asserts on.

    DEVIATION from the Task 10 brief (see the report): the brief asserted
    `nz.lambdas["group"] == 0.0` on the activating transaction, "activation
    ramp at it_activate". That is not what the landed norm.py does.
    `self.lambdas` carries the COMMITTED, UN-RAMPED coefficient (norm.py's
    TermState docstring and `_compute`); the activation ramp lives only in
    `TermState.lam_applied` / `applied_lambda(name, iteration)`. Asserted both
    ways round here, because the difference is exactly why the driver
    registers with n_ramp = 0 (next test).
    """
    import torch
    from ioplace.norm import TermNormalizer
    from ioplace.drivers.run_region_producer import _GroupAdapter
    nz = TermNormalizer(policy="grandplan", norm_p=1, probe_every=20,
                        num_movable=2, num_nodes=4)
    nz.register("group", _GroupAdapter(_StubTerm()), curvature=1.0,
                activate_overflow=1.0)
    pos = torch.zeros(8, dtype=torch.float64)
    pos[0], pos[1] = 1.0, 2.0
    wl = lambda p: 10.0 * p.abs().sum()
    ctx = {"iteration": 0, "overflow": 0.0, "tau": 0.0, "gamma": 0.0}
    assert not nz.needs_refresh()
    assert nz.probe(0, pos, wl, ctx) == {"wl": 20.0, "group": 3.0}
    nz.transaction(0, 0.0, 0.0, 0.0)
    assert nz.needs_refresh()
    nz.mark_refreshed()
    assert not nz.needs_refresh()
    assert nz.states["group"].ratio_ema == pytest.approx(20.0 / 3.0)
    # committed (un-ramped) vs applied (ramped) at the activating transaction
    assert nz.lambdas["group"] == pytest.approx(0.05 * 20.0 / 3.0)
    assert nz.states["group"].lam_applied == 0.0   # n_ramp=20 ramp at it_activate
    assert nz.applied_lambda("group", 0) == 0.0
    nz.probe(20, pos, wl, dict(ctx, iteration=20))
    nz.transaction(20, 0.0, 0.0, 0.0)
    nz.mark_refreshed()
    assert nz.states["group"].wt == pytest.approx(0.05)
    assert nz.lambdas["group"] == pytest.approx(0.05 * 20.0 / 3.0)
    assert nz.applied_lambda("group", 20) == pytest.approx(0.05 * 20.0 / 3.0)


def test_the_driver_registers_the_group_term_with_no_activation_ramp():
    """Controller ruling of 2026-09-20, on top of D5/D6: the group term is
    registered with n_ramp = 0, so the grouping force is live from the
    activating transaction (GrandPlan applies grouping from iteration 0 and
    Eq.3's own `wt` ramp already supplies the gentle onset).

    This is load-bearing, not cosmetic: the driver's attach_terms closure reads
    `normalizer.lambdas`, which is the UN-ramped coefficient. With register's
    default n_ramp = 20 the closure would apply the un-ramped lambda while the
    normalizer recorded a ramped `lam_applied` -- the trace and the force would
    disagree silently. At n_ramp = 0 the two are the same number.
    """
    import torch
    from ioplace.norm import TermNormalizer
    from ioplace.drivers.run_region_producer import GROUP_N_RAMP, _GroupAdapter
    assert GROUP_N_RAMP == 0
    nz = TermNormalizer(policy="grandplan", norm_p=1, probe_every=20,
                        num_movable=2, num_nodes=4)
    nz.register("group", _GroupAdapter(_StubTerm()), curvature=1.0,
                activate_overflow=1.0, n_ramp=GROUP_N_RAMP)
    pos = torch.zeros(8, dtype=torch.float64)
    pos[0], pos[1] = 1.0, 2.0
    nz.probe(0, pos, lambda p: 10.0 * p.abs().sum(),
             {"iteration": 0, "overflow": 0.0, "tau": 0.0, "gamma": 0.0})
    nz.transaction(0, 0.0, 0.0, 0.0)
    nz.mark_refreshed()
    lam = nz.lambdas["group"]
    assert lam == pytest.approx(0.05 * 20.0 / 3.0)
    assert nz.applied_lambda("group", 0) == pytest.approx(lam)
    assert nz.states["group"].lam_applied == pytest.approx(lam)


def test_rectify_falls_back_to_32_bins_and_records_the_path():
    """Ruling D3: exhausting the rect budget at 64 bins is a driver fallback,
    not an abort and not an operator instruction."""
    from ioplace.drivers.run_region_producer import rectify_with_fallback
    lab32 = np.zeros((32, 32), dtype=np.int16)
    lab32[:, 16:] = 1
    seen = []

    def build(bins):
        seen.append(bins)
        if bins == 64:
            raise RuntimeError("region 8 has 18 rects (> 8) and neither a "
                               "feasible notch fill nor a feasible shed remains")
        return lab32, {"levels_run": 3}

    bins_used, labels, report, path, reason = rectify_with_fallback(
        build, lambda lab: lab, 64)
    assert seen == [64, 32]
    assert bins_used == 32 and path == "fallback_32"
    assert "18 rects" in reason
    assert labels.shape == (32, 32) and report["levels_run"] == 3


def test_rectify_does_not_fall_back_at_or_below_32_bins():
    from ioplace.drivers.run_region_producer import rectify_with_fallback
    seen = []

    def build(bins):
        seen.append(bins)
        raise RuntimeError("no feasible move remains")

    with pytest.raises(RuntimeError, match="no feasible move"):
        rectify_with_fallback(build, lambda lab: lab, 32)
    assert seen == [32]


def _grad_l1_fixture():
    """A 2-movable-node, K=1 GroupingTerm with real anchor tables on a tiny
    lattice: one cell outside the hull (pull active), one inside (pull off,
    and with a single hull there is no foreign hull, so push is off too).

    num_nodes == num_movable, so TermNormalizer.probe's fixed/filler mask is a
    no-op and its ||grad Group||_1 must equal GroupingTerm.grad_l1 exactly.
    """
    import torch
    from ioplace.producer import hull as hull_mod
    from ioplace.producer.grouping_term import GroupingTerm
    die = (0.0, 0.0, 16.0, 16.0)
    hull = np.array([[4.0, 4.0], [12.0, 4.0], [12.0, 12.0], [4.0, 12.0]])
    tables = hull_mod.anchor_tables([hull], die, 8, device="cpu")
    size = np.array([2.0, 2.0])
    term = GroupingTerm(part=np.zeros(2, dtype=np.int64), node_size_x=size,
                        node_size_y=size, num_movable=2, num_nodes=2,
                        device="cpu")
    term.set_tables(tables)
    # pos is [x_0, x_1, y_0, y_1] of lower-left corners; centres are +1.0.
    pos = torch.tensor([0.0, 7.0, 0.0, 7.0], dtype=torch.float64)
    return term, pos


def test_grouping_term_grad_l1_matches_the_frozen_anchor_gradient():
    """Task 4 review gap: grad_l1 is what feeds TermNormalizer's ratio, and a
    wrong value there silently mis-scales Eq.3's coefficient. With a frozen
    anchor the gradient is the exact spring force
    alpha*w*(x - anchor), so the L1 norm is closed-form off the tables."""
    import torch
    term, pos = _grad_l1_fixture()
    x, y = term._centres(pos)
    cx, cy, pull_off, pull_on, push_off, push_n = term._lookup(x, y)
    gx = (term.alpha_pull * pull_on.double() * (x - (cx + pull_off[:, 0]))
          + term.alpha_push * push_n * (x - (cx + push_off[:, 0])))
    gy = (term.alpha_pull * pull_on.double() * (y - (cy + pull_off[:, 1]))
          + term.alpha_push * push_n * (y - (cy + push_off[:, 1])))
    expected = float(gx.abs().sum() + gy.abs().sum())
    assert expected > 0.0, "fixture must exercise a live pull"
    assert term.grad_l1(pos) == pytest.approx(expected, rel=1e-12)
    # and it is genuinely the outside cell that carries the force
    assert bool(pull_on[0]) and not bool(pull_on[1])


def test_grad_l1_agrees_with_the_normalizer_probe_the_driver_uses():
    """The driver never calls grad_l1 -- it registers _GroupAdapter and lets
    TermNormalizer.probe measure the same quantity. The two must agree, or the
    standalone diagnostic and the production coefficient disagree silently."""
    import torch
    from ioplace.norm import TermNormalizer
    from ioplace.drivers.run_region_producer import _GroupAdapter
    term, pos = _grad_l1_fixture()
    nz = TermNormalizer(policy="grandplan", norm_p=1, probe_every=1,
                        num_movable=2, num_nodes=2)
    nz.register("group", _GroupAdapter(term), curvature=1.0,
                activate_overflow=1.0)
    norms = nz.probe(0, pos, lambda p: p.abs().sum(),
                     {"iteration": 0, "overflow": 0.0, "tau": 0.0, "gamma": 0.0})
    assert norms["group"] == pytest.approx(term.grad_l1(pos), rel=1e-12)


def test_grad_l1_is_zero_without_tables():
    from ioplace.producer.grouping_term import GroupingTerm
    import torch
    size = np.array([2.0])
    term = GroupingTerm(part=np.zeros(1, dtype=np.int64), node_size_x=size,
                        node_size_y=size, num_movable=1, num_nodes=1,
                        device="cpu")
    assert term.grad_l1(torch.zeros(2, dtype=torch.float64)) == 0.0


@pytest.mark.slow
def test_producer_emits_all_four_artefacts_on_simple(tmp_path):
    from ioplace.drivers.run_region_producer import run_producer
    out = str(tmp_path / "run")
    res = run_producer(CFG, out, k=2, membership_source="mtkahypar",
                       extract_bins=32, rect_max=8, seed=0, t_hull=20,
                       probe_every=20, sa_seed=0, gp_iterations=200)
    for name in ("regions.json", "seed.npz", "membership.npz", "producer.json"):
        assert os.path.exists(os.path.join(out, name)), name

    rs = RegionSet.from_json(os.path.join(out, "regions.json"))
    rs.validate()
    assert rs.k == 2 and rs.lattice == 512
    for r in rs.regions:
        assert 1 <= len(np.asarray(r.rects)) <= 8

    seed = artifacts.load_positions(os.path.join(out, "seed.npz"),
                                    expect_num_physical=res["num_physical"],
                                    expect_sha256=res["placedb_sha256"])
    assert seed.kind == "seed"
    assert seed.node_x.shape == seed.node_y.shape
    assert seed.die == tuple(res["die_native"])

    mem = artifacts.load_membership(os.path.join(out, "membership.npz"),
                                    expect_num_movable=res["num_movable"],
                                    expect_k=2)
    assert mem.source == "mtkahypar"

    doc = artifacts.load_producer_json(os.path.join(out, "producer.json"))
    assert doc["k"] == 2 and doc["extract_bins"] == 32 and doc["rect_max"] == 8
    assert doc["sa"]["rect_max_path"] == "direct"
    assert doc["n_hull_rebuilds"] >= 1
    assert max(doc["rects_per_region"]) <= 8
    assert doc["runtime_s"]["total"] > 0.0
    assert set(doc["runtime_s"]) >= {"read", "prior", "gp", "extract", "sa",
                                     "rectify", "total"}


@pytest.mark.slow
def test_seed_is_in_native_units_and_inside_the_die(tmp_path):
    """The coordinate contract: seed.npz must be in the post-read die box, not
    the post-initialize scaled one."""
    from ioplace.drivers.run_region_producer import run_producer
    out = str(tmp_path / "run")
    res = run_producer(CFG, out, k=2, extract_bins=32, t_hull=20,
                       probe_every=20, gp_iterations=200)
    xl, yl, xh, yh = res["die_native"]
    assert (xl, yl) != (0.0, 0.0), "simple.json's die does not start at the origin"
    s = artifacts.load_positions(os.path.join(out, "seed.npz"))
    assert s.shift_factor == (xl, yl)
    assert (s.node_x >= xl - 1e-6).all() and (s.node_x <= xh + 1e-6).all()
    assert (s.node_y >= yl - 1e-6).all() and (s.node_y <= yh + 1e-6).all()


@pytest.mark.slow
def test_extract_bins_32_and_64_both_produce_valid_geometry(tmp_path):
    """spec section 8: arm (e) needs faithful 32^2 extraction, so the producer
    must expose --extract-bins {32,64} and both must validate."""
    from ioplace.drivers.run_region_producer import run_producer
    for bins in (32, 64):
        out = str(tmp_path / f"b{bins}")
        res = run_producer(CFG, out, k=2, extract_bins=bins, t_hull=20,
                           probe_every=20, gp_iterations=200)
        rs = RegionSet.from_json(os.path.join(out, "regions.json"))
        rs.validate()
        assert res["extract_bins"] == bins
        assert rs.lattice == 512


@pytest.mark.slow
def test_producer_is_reproducible_with_the_same_seeds(tmp_path):
    from ioplace.drivers.run_region_producer import run_producer
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    for out in (a, b):
        run_producer(CFG, out, k=2, extract_bins=32, seed=0, sa_seed=0,
                     t_hull=20, probe_every=20, dp_seed=1000, deterministic=1,
                     gp_iterations=200)
    pa = artifacts.load_membership(os.path.join(a, "membership.npz")).part
    pb = artifacts.load_membership(os.path.join(b, "membership.npz")).part
    assert np.array_equal(pa, pb)
    ja = json.load(open(os.path.join(a, "regions.json")))
    jb = json.load(open(os.path.join(b, "regions.json")))
    assert ja["regions"] == jb["regions"]


@pytest.mark.slow
def test_version_invariant_is_live_on_the_real_optimizer(tmp_path, monkeypatch):
    """Ruling D6, negative control. install_version_invariant is installed on
    `placer.optimizer` from the first iteration callback, so breaking the
    refresh half of the transaction contract must abort the run instead of
    silently stepping Nesterov's secant cache against a stale objective.

    Without this the D6 wiring is untestable from outside: every positive test
    passes just as well with the invariant never installed at all.
    """
    from ioplace.norm import TermNormalizer
    from ioplace.drivers.run_region_producer import run_producer
    orig = TermNormalizer.mark_refreshed
    calls = {"n": 0}

    def broken(self):
        calls["n"] += 1
        if calls["n"] >= 2:      # let the installing callback finish normally
            return
        return orig(self)

    monkeypatch.setattr(TermNormalizer, "mark_refreshed", broken)
    with pytest.raises(AssertionError, match="refresh_nesterov_secant"):
        run_producer(CFG, str(tmp_path / "neg"), k=2, extract_bins=32,
                     t_hull=20, probe_every=20, gp_iterations=60)
    assert calls["n"] >= 2, "the invariant must fire after a missed refresh"
