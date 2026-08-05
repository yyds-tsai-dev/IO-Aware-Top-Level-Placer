import numpy as np
import pytest
torch = pytest.importorskip("torch")
from ioplace.reweight import update_net_weights

def test_update_net_weights_formula_and_inplace():
    w = torch.ones(5)
    wid = id(w)
    update_net_weights(w, np.array([0, 1, 2, 20, 5]), alpha=0.5, cap=10.0)
    assert id(w) == wid
    assert torch.allclose(w, torch.tensor([1.0, 1.5, 2.0, 6.0, 3.5]))


# --- Task 12 review follow-up ------------------------------------------------
# The reweight closed loop (ioplace/drivers/run_placement_reweight.py's `cb`)
# feeds GpuEvalContext.evaluate(...).per_net_crossings[i] into net_weights[i],
# assuming index i means "the same net" on both sides. That assumption rests
# on two links that were never independently tested:
#   (1) netlist_from_placedb (ioplace/netlist.py) copies placedb's net-indexed
#       arrays into the Netlist without reordering them.
#   (2) GpuEvalContext.evaluate's per_net_crossings is indexed by that same
#       Netlist net index, and update_net_weights applies the documented
#       formula at that same index.
# The two tests below prove each link constructively (by hand-checkable
# fixtures), rather than arguing it from reading the source.

def test_netlist_preserves_placedb_net_order():
    """Link (1): netlist_from_placedb must not sort/filter/renumber nets or
    pins. Fake placedb has 3 nets of different degrees (2, 3, 1) with a
    deliberately unsorted pin layout (pin ids do not appear in ascending
    order within a net), so any reordering bug -- e.g. sorting nets by
    degree, or pins by id -- would show up as an elementwise mismatch below.
    """
    from types import SimpleNamespace
    from ioplace.netlist import netlist_from_placedb

    # net0 (flat[0:2]) -> pin ids [4, 1]
    # net1 (flat[2:5]) -> pin ids [0, 5, 2]
    # net2 (flat[5:6]) -> pin ids [3]
    flat_net2pin_map = np.array([4, 1, 0, 5, 2, 3], dtype=np.int32)
    flat_net2pin_start_map = np.array([0, 2, 5, 6], dtype=np.int32)
    # pin id -> owning net, authored independently of the flat map above so a
    # bug that reorders one array but not the other is still caught.
    pin2net_map = np.array([1, 0, 1, 2, 0, 1], dtype=np.int32)
    pin2node_map = np.array([2, 0, 1, 3, 0, 2], dtype=np.int32)

    fake_placedb = SimpleNamespace(
        num_physical_nodes=4,
        node_x=np.array([0.0, 10.0, 20.0, 30.0]),
        node_y=np.array([0.0, 5.0, 15.0, 25.0]),
        node_size_x=np.ones(4), node_size_y=np.ones(4),
        num_movable_nodes=3, num_terminals=1, num_terminal_NIs=0,
        pin_offset_x=np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.2]),
        pin_offset_y=np.array([0.0, 0.1, -0.1, 0.2, 0.0, -0.2]),
        pin2node_map=pin2node_map,
        pin2net_map=pin2net_map,
        flat_net2pin_map=flat_net2pin_map,
        flat_net2pin_start_map=flat_net2pin_start_map,
        xl=0.0, yl=0.0, xh=100.0, yh=100.0,
    )

    nl = netlist_from_placedb(fake_placedb)

    # constructive identity: no reordering anywhere in the pipeline
    assert np.array_equal(nl.flat_net2pin_start, flat_net2pin_start_map)
    assert np.array_equal(nl.flat_net2pin, flat_net2pin_map)
    assert np.array_equal(nl.pin2net, pin2net_map)
    assert np.array_equal(nl.pin2node, pin2node_map)
    # derived per-net degrees land where the shuffled layout implies
    assert list(nl.net_degrees) == [2, 3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reweight_end_to_end_index_alignment():
    """Link (2) + formula, end to end: GpuEvalContext(nl, rg).evaluate(...)
    -> update_net_weights must land each net's own crossing count on that
    same net's weight. Every expected value below is derived by hand from
    the algorithm (Prim's MST + region-grid crossing prefix sums, see
    ioplace/evaluator_gpu.py's module docstring) -- not copied from a
    program run.

    Region grid: 2x2 over die (0,0,100,100), lattice=2, so each of the 4
    regions is exactly one lattice cell = one quadrant:
      region0=[0,50)x[0,50) (ix=0,iy=0)   region1=[50,100)x[0,50) (ix=1,iy=0)
      region2=[0,50)x[50,100)(ix=0,iy=1)  region3=[50,100)x[50,100)(ix=1,iy=1)
    With only 2 cells per axis, Ph/Pv (the horizontal/vertical crossing
    prefix sums in GpuEvalContext) reduce to a shortcut used throughout this
    derivation: Ph[0,:]==Ph[1,:]==[0,1] and Pv[:,0]==Pv[:,1]==[0,1], so a
    straight MST edge between pins at grid indices (ax,ay) and (bx,by) in
    {0,1} crosses exactly (ax!=bx) + (ay!=by) region boundaries -- 1 per axis
    the two quadrants differ on -- regardless of which pin is the MST
    "source". (Cross-checked, not just asserted here, against
    ioplace.evaluator_ref.evaluate on this exact fixture while authoring
    this test.)

    5 nets, each hand-picked to a distinct crossing count -- so a shuffled
    index would visibly land a value on the wrong net:

      net0 (d=2): both pins in region0                         -> crossing 0
      net1 (d=2): region0 -> region1 (differ ix only)           -> crossing 1
      net2 (d=2): region0 -> region3 (differ ix and iy)         -> crossing 2
      net3 (d=3): A=region0(49.9,10) B=region1(99,10) C=region3(50.1,90)
                  Prim from A: dist(A,B)=49.1, dist(A,C)=80.2, dist(B,C)=128.9
                    step1: closer of B/C to A is B (49.1<80.2) -> edge A-B;
                    C's best-cost stays "via A" (80.2 < dist(B,C)=128.9, so
                    never beaten by routing through B) -> step2: edge A-C
                  crossing = (A-B: differ ix = 1) + (A-C: differ both = 2) = 3
      net4 (d=3): P0=region0(1,49) P1=region0(49,1) P2=region3(50.5,50.5)
                  Prim from P0: dist(P0,P1)=96, dist(P0,P2)=51, dist(P1,P2)=51
                    step1: P2 is closer to P0 (51<96) -> edge P0-P2;
                    P1's best-cost updates to 51 "via P2" (51<96 via P0)
                    -> step2: edge P2-P1
                  crossing = (P0-P2: differ both = 2) + (P2-P1: differ both = 2) = 4

    per_net_crossings = [0, 1, 2, 3, 4]; alpha=0.5, cap=10.0 (all crossings
    well under cap) => expected net_weights = 1 + 0.5*[0,1,2,3,4]
    = [1.0, 1.5, 2.0, 2.5, 3.0].
    """
    from ioplace.netlist import Netlist
    from ioplace.region_grid import RegionGrid
    from ioplace.regions import make_grid_regions
    from ioplace.evaluator_gpu import GpuEvalContext

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

    expected_crossings = [0, 1, 2, 3, 4]
    assert np.array_equal(res.per_net_crossings, np.array(expected_crossings))

    w = torch.ones(nl.num_nets, device="cuda")
    update_net_weights(w, res.per_net_crossings, alpha=0.5, cap=10.0)
    w_cpu = w.cpu().numpy()
    for j, crossing_j in enumerate(expected_crossings):
        expected_w_j = 1 + 0.5 * min(crossing_j, 10.0)
        assert w_cpu[j] == pytest.approx(expected_w_j), (
            f"net {j}: got {w_cpu[j]}, expected {expected_w_j} "
            f"(crossing={crossing_j})")


# NOTE on the benchmark choice below (coordinator override; brief Step 2
# originally specified install/test/simple.json):
#
# `simple` (8 movable cells) has a non-deterministic filler-sizing bug (see
# tests/test_fence_inject.py's note on the same benchmark) and is too small
# for a meaningful reweighting signal, so this integration test targets
# adaptec1 instead -- a full flat GP+LG run on adaptec1 takes ~1.5-3 minutes.
# `every=100` (vs. the brief's `every=20`, sized for `simple`'s handful of
# iterations) is scaled to adaptec1's iteration count instead. Assertions are
# otherwise unchanged from the brief.
@pytest.mark.slow
def test_reweight_adaptec1_runs_and_calls_back(tmp_path):
    import os
    from ioplace.drivers.run_placement_reweight import run_reweight
    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    res = run_reweight(os.path.join(root, "install/test/ispd2005/adaptec1.json"),
                       4, "grid", 0, str(tmp_path / "rw.json"), every=100, alpha=0.5)
    assert res["mode"] == "reweight"
    assert res["num_reweights"] >= 1
    # Task 12 review follow-up: the closed loop must have actually moved
    # net_weights off its all-ones default, and stayed within the documented
    # w = 1 + alpha*min(crossings, cap=10.0) range for this run's alpha.
    assert res["net_weights_min"] >= 1.0
    assert res["net_weights_max"] <= 1.0 + res["alpha"] * 10.0
    assert res["net_weights_max"] > res["net_weights_min"]  # non-uniform: really changed
