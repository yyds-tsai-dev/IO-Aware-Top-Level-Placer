"""M3 Phase B: dynamic per-net ft_rg reweight (docs/results/
2026-08-14-m3-s4-adjudication.md sec C item 2 -- the preregistered fallback
route: reuse the M1 "w = 1 + alpha*min(signal, cap)" formula family
(ioplace/reweight.py::update_net_weights), signal switched from IO crossings
to per-net ft_rg).

Four things this file locks down (task's own checklist):
  1. formula + net-index alignment (test_ft_rg_formula_and_index_alignment) --
     mirrors tests/test_reweight.py's test_reweight_end_to_end_index_alignment
     fixture exactly (same 5-net/2x2-region geometry) so the hand-derived
     per_net_crossings numbers already proven correct there are reused as a
     cross-check, alongside newly hand-derived per_net_lambda/per_net_steiner.
  2. event ordering / version invariant (test_ft_reweight_on_respects_version_invariant)
  3. --ft-reweight off is a true no-op (test_ft_reweight_off_is_bit_identical_to_no_flag)
  4. off/on validation, incl. mutual exclusion with the pre-existing alpha_io
     IO-crossings reweight (test_ft_reweight_rejects_bad_flag /
     test_ft_reweight_mutually_exclusive_with_alpha_io)
"""
import os
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.drivers.run_placement_io import run_io, RESULT_FIELDS

DP = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
SIMPLE = os.path.join(DP, "install", "test", "simple.json")


def test_result_fields_declare_the_new_columns():
    for f in ("io_rg", "ft_rg", "ft_reweight", "alpha_ft"):
        assert f in RESULT_FIELDS


def test_ft_reweight_rejects_bad_flag():
    with pytest.raises(ValueError, match="ft_reweight"):
        run_io(SIMPLE, 4, "grid", 0, "/tmp/unused_ft_reweight_bad_flag.json",
              ft_reweight="maybe")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ft_reweight_mutually_exclusive_with_alpha_io(tmp_path):
    """Both alpha_io>0 and ft_reweight='on' write io_term.w from a different
    per-net signal -- the M1 formula family reused here writes one w per
    net, not a sum of two, so running both at once is undefined and must
    fail fast (before the placement loop, not silently pick a winner)."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "unused.json"),
              rho_max=0.05, alpha_io=0.3, ft_reweight="on")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ft_rg_formula_and_index_alignment():
    """Same 5-net/2x2-region fixture as tests/test_reweight.py's
    test_reweight_end_to_end_index_alignment (region ids/adjacency/D exactly
    as documented there): region0=[0,50)x[0,50) region1=[50,100)x[0,50)
    region2=[0,50)x[50,100) region3=[50,100)x[50,100), so D[0,1]=D[0,2]=
    D[1,3]=D[2,3]=1 (share a lattice edge) and D[0,3]=D[1,2]=2 (diagonal,
    routed through either neighbour).

    Per-net (Lambda_e, ST_e) hand-derived from the same pin geometry:
      net0 (d=2): both pins in region0 -> touched={0}, Lambda=1 (popcount of
        a single-region bitmask); per_net_steiner stays at its zero-init
        (Lambda==1 matches none of the lam2/lam3/lam_ge4 branches) -> ST=0.
      net1 (d=2): region0->region1 -> touched={0,1}, Lambda=2,
        ST=D[0,1]=1.
      net2 (d=2): region0->region3 -> touched={0,3}, Lambda=2,
        ST=D[0,3]=2.
      net3 (d=3): A=region0(49.9,10) B=region1(99,10) C=region3(50.1,90) ->
        touched={0,1,3}, Lambda=3; Steiner = min_v D[v,0]+D[v,1]+D[v,3] over
        v in {0,1,2,3} = {3,2,4,3} -> 2 (at v=1).
      net4 (d=3): P0=region0(1,49) P1=region0(49,1) P2=region3(50.5,50.5) ->
        P0,P1 collapse to the same region0 -> touched={0,3}, Lambda=2,
        ST=D[0,3]=2.

    FT_rg_e = ST_e - max(Lambda_e-1, 0):
      net0: 0-0=0   net1: 1-1=0   net2: 2-1=1   net3: 2-2=0   net4: 2-1=1
      -> [0, 0, 1, 0, 1], sum=2.

    Cross-checked against the scalar aggregate fields evaluate_gpu already
    returns: hard_lambda_sum = sum(max(Lambda_e-1,0)) = 0+1+1+2+1 = 5,
    io_rg = sum(ST_e) = 0+1+2+2+2 = 7, ft_rg = io_rg - hard_lambda_sum = 2 --
    matches the per-net sum, so the per-net decomposition used by the
    driver's ft-reweight block is not just locally plausible but exactly
    reconstructs the evaluator's own scalar ft_rg.

    update_net_weights (ioplace/reweight.py, reused unchanged by
    run_placement_io.py's new ft_reweight='on' block) with alpha=0.5,
    cap=10.0 (all FT_rg_e well under cap) -> w = 1 + 0.5*FT_rg_e =
    [1.0, 1.0, 1.5, 1.0, 1.5].
    """
    from ioplace.netlist import Netlist
    from ioplace.region_grid import RegionGrid
    from ioplace.regions import make_grid_regions
    from ioplace.evaluator_gpu import GpuEvalContext
    from ioplace.reweight import update_net_weights

    DIE = (0., 0., 100., 100.)
    coords = [
        (10, 10), (30, 30),                  # net0
        (10, 10), (90, 10),                  # net1
        (10, 10), (90, 90),                  # net2
        (49.9, 10), (99, 10), (50.1, 90),    # net3: A, B, C
        (1, 49), (49, 1), (50.5, 50.5),      # net4: P0, P1, P2
    ]
    n = len(coords)
    node_x = np.array([c[0] for c in coords], dtype=np.float64)
    node_y = np.array([c[1] for c in coords], dtype=np.float64)
    flat_net2pin_start = np.array([0, 2, 4, 6, 9, 12], dtype=np.int32)
    flat_net2pin = np.arange(n, dtype=np.int32)
    pin2net = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.int32)
    pin2node = np.arange(n, dtype=np.int32)  # one pin per node, zero offset

    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=np.ones(n), node_size_y=np.ones(n),
                 num_movable=n, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(n), pin_offset_y=np.zeros(n),
                 pin2node=pin2node, pin2net=pin2net,
                 flat_net2pin=flat_net2pin, flat_net2pin_start=flat_net2pin_start,
                 xl=DIE[0], yl=DIE[1], xh=DIE[2], yh=DIE[3])
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=2))

    ctx = GpuEvalContext(nl, rg)
    res = ctx.evaluate(nl.node_x, nl.node_y)

    expected_lambda = np.array([1, 2, 2, 3, 2])
    expected_steiner = np.array([0, 1, 2, 2, 2])
    assert np.array_equal(res.per_net_lambda, expected_lambda)
    assert np.array_equal(res.per_net_steiner, expected_steiner)

    ft_rg_e = res.per_net_steiner.astype(np.float64) - np.maximum(
        res.per_net_lambda.astype(np.float64) - 1.0, 0.0)
    expected_ft_rg_e = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    assert np.array_equal(ft_rg_e, expected_ft_rg_e)

    # scalar aggregates the evaluator itself returns must reconstruct the
    # same per-net sum -- not just self-consistent with this test's own
    # by-hand numbers.
    assert res.hard_lambda_sum == 5
    assert res.io_rg == 7
    assert res.ft_rg == 2
    assert res.ft_rg == int(ft_rg_e.sum())

    w = torch.ones(nl.num_nets, device="cuda")
    update_net_weights(w, ft_rg_e, alpha=0.5, cap=10.0)
    w_cpu = w.cpu().numpy()
    expected_w = np.array([1.0, 1.0, 1.5, 1.0, 1.5])
    assert np.allclose(w_cpu, expected_w)


@pytest.mark.slow
def test_ft_reweight_off_is_bit_identical_to_no_flag(tmp_path):
    """Regression lock: --ft-reweight off (the default) must be a true
    no-op, even when alpha_ft is set to a non-default value -- the gate is
    ft_reweight itself, not alpha_ft. rho_max>0 so io_term.w is actually
    live (not the trivial rho_max=0 observer-mode case)."""
    a = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "a.json"), rho_max=0.05,
              every=10, dp_seed=1000, deterministic=1)
    b = run_io(SIMPLE, 4, "grid", 0, str(tmp_path / "b.json"), rho_max=0.05,
              every=10, dp_seed=1000, deterministic=1,
              ft_reweight="off", alpha_ft=2.0)
    xa = np.load(str(tmp_path / "a.json") + ".npz")["node_x"]
    xb = np.load(str(tmp_path / "b.json") + ".npz")["node_x"]
    ya = np.load(str(tmp_path / "a.json") + ".npz")["node_y"]
    yb = np.load(str(tmp_path / "b.json") + ".npz")["node_y"]
    assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    assert a["io_count"] == b["io_count"] and a["ft_count"] == b["ft_count"]
    assert b["ft_reweight"] == "off" and b["alpha_ft"] == 2.0


@pytest.mark.slow
def test_ft_reweight_on_respects_version_invariant(tmp_path):
    """Event-order contract (design draft sec 5.5's atomic-callback
    discipline, reused here for the reweight coefficient update, same as
    the pre-existing alpha_io block): the ft_rg-signalled w update and its
    obj_version bump happen inside the same N=every evaluator-gated branch
    as the refresh check immediately below it ("順序不可換:先讓新
    tau/lambda/w 生效,再 refresh"), so a live version-invariant guard
    (install_version_invariant) must never fire across the whole run."""
    res = run_io(SIMPLE, k=4, rtype="grid", seed=0,
                 out_json=str(tmp_path / "io.json"), rho_max=0.05, every=10,
                 ft_reweight="on", alpha_ft=0.5, check_invariant=True)
    assert res["ft_reweight"] == "on" and res["alpha_ft"] == 0.5
    assert res["num_refreshes"] >= 1
    assert res["io_rg"] >= 0 and res["ft_rg"] >= 0
