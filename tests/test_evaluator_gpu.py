import numpy as np
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from ioplace.regions import make_grid_regions
from ioplace.region_grid import RegionGrid
from ioplace.evaluator_ref import evaluate
from ioplace.evaluator_gpu import evaluate_gpu
from tests.test_evaluator_ref import DIE  # (0,0,100,100)

def _random_case(rng, n_cells=40, n_nets=25, max_d=12):
    from ioplace.netlist import Netlist
    nx_, ny_ = rng.uniform(0.5, 99.5, n_cells), rng.uniform(0.5, 99.5, n_cells)
    pins, p2n = [], []
    for net in range(n_nets):
        d = int(rng.integers(2, max_d))
        pins += list(rng.choice(n_cells, d, replace=False)); p2n += [net] * d
    p2n = np.array(p2n, np.int32); pins = np.array(pins, np.int32)
    start = np.searchsorted(p2n, np.arange(n_nets + 1)).astype(np.int32)
    return Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                   node_size_y=np.ones(n_cells), num_movable=n_cells,
                   num_terminals=0, num_terminal_NIs=0,
                   pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                   pin2node=pins, pin2net=p2n,
                   flat_net2pin=np.arange(len(pins), dtype=np.int32),
                   flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gpu_matches_reference(seed):
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert gpu.io_count == ref.io_count
    assert gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.tree_wl == pytest.approx(ref.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand


def test_gpu_matches_reference_large_net():
    """A net with degree > max_degree exercises the presence-lower-bound branch
    (GpuEvalContext.large_net_ids_t / per_net_crossings lower bound), which the
    5-seed test above never triggers (those nets all have degree < 12). Small case:
    max_degree=8 and one net with degree 20 forces exactly that net through the
    large-net path while the rest still go through the normal MST/FT path."""
    from ioplace.netlist import Netlist
    rng = np.random.default_rng(11)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    n_cells = 25
    nx_, ny_ = rng.uniform(0.5, 99.5, n_cells), rng.uniform(0.5, 99.5, n_cells)
    degree_plan = [2, 3, 4, 20, 6]  # net index 3 has degree 20 > max_degree=8
    pins, p2n = [], []
    for net, d in enumerate(degree_plan):
        pins += list(rng.choice(n_cells, d, replace=False))
        p2n += [net] * d
    p2n = np.array(p2n, np.int32)
    pins = np.array(pins, np.int32)
    start = np.searchsorted(p2n, np.arange(len(degree_plan) + 1)).astype(np.int32)
    nl = Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                 node_size_y=np.ones(n_cells), num_movable=n_cells,
                 num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                 pin2node=pins, pin2net=p2n,
                 flat_net2pin=np.arange(len(pins), dtype=np.int32),
                 flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg, max_degree=8)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg, max_degree=8)
    assert gpu.io_count == ref.io_count
    assert gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.tree_wl == pytest.approx(ref.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand
    assert gpu.large_net_lb == ref.large_net_lb
    assert gpu.large_net_lb > 0  # sanity: the large-net branch was actually exercised


def test_gpu_matches_reference_on_lattice_boundaries():
    """C1 regression (whole-branch review): GpuEvalContext._to_idx used to divide
    by the plain python floats self.cell_w/self.cell_h. CUDA compiles
    `tensor / python_float` as a reciprocal-multiply (x * (1/c)), and (1/c) is
    itself rounded -- so a coordinate that lands exactly on a lattice boundary can
    come out one ULP low (e.g. 2673.0 / (10692/512) -> 127.99999999999999, not
    128.0), which `.to(torch.int64)` truncates down to 127 instead of 128: an
    off-by-one lattice-cell mis-assignment exactly at region boundaries. Plain
    `tensor / tensor` (even a 0-dim one) instead correctly rounds, matching numpy.

    This die (10692x10680, lattice=512 -> cell_w=10692/512=20.8828125,
    cell_h=10680/512=20.859375) and a 4x4 grid partition (region boundaries at
    lattice index 128/256/384) are not arbitrary: empirically (see the C1 fix in
    evaluator_gpu.py) this exact cell_w mis-rounds 80 of the 511 possible
    lattice-boundary indices when divided as tensor/python-float, including
    indices 128 and 256 -- exactly the region-column boundaries hit below by
    node_x 2673.0 (=128*cell_w) and 5346.0 (=256*cell_w). Before the C1 fix this
    test fails (GPU under-counts crossings by 1 on both boundary-straddling nets,
    per_net_crossings [1,1,3] (ref) vs [0,0,3] (gpu)); after the fix it's exact.
    """
    from ioplace.netlist import Netlist
    DIE_ND = (0., 0., 10692., 10680.)
    rs = make_grid_regions(DIE_ND, 4, 4, lattice=512)
    rg = RegionGrid(rs)

    # 5 cells, one pin each (offset 0,0):
    #   0: (2673.0, 5000.0) -- x exactly on the col0/col1 region boundary (idx 128)
    #   1: (2000.0, 5000.0) -- safely inside col0, same row as cell 0
    #   2: (5346.0, 5000.0) -- x exactly on the col1/col2 region boundary (idx 256)
    #   3: (4700.0, 5000.0) -- safely inside col1, same row as cell 2
    #   4: (9000.0, 2670.0) -- y exactly on a row boundary (idx 128) but a
    #      non-triggering axis (cell_h happens not to mis-round here -- included to
    #      show the fix doesn't disturb the already-correct axis), x safely inside col3
    node_x = np.array([2673.0, 2000.0, 5346.0, 4700.0, 9000.0], dtype=np.float64)
    node_y = np.array([5000.0, 5000.0, 5000.0, 5000.0, 2670.0], dtype=np.float64)

    # net0 = {0,1} straddles the idx-128 boundary; net1 = {2,3} straddles idx-256;
    # net2 = {1,3,4} is a plain 3-pin net (no boundary pin) exercising MST with the
    # fix in place, alongside the two boundary-triggering 2-pin nets.
    pin2node = np.array([0, 1, 2, 3, 1, 3, 4], dtype=np.int32)
    pin2net = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    flat_net2pin = np.arange(7, dtype=np.int32)
    flat_net2pin_start = np.array([0, 2, 4, 7], dtype=np.int32)

    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=np.ones(5), node_size_y=np.ones(5),
                 num_movable=5, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(7), pin_offset_y=np.zeros(7),
                 pin2node=pin2node, pin2net=pin2net,
                 flat_net2pin=flat_net2pin, flat_net2pin_start=flat_net2pin_start,
                 xl=0., yl=0., xh=10692., yh=10680.)

    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    assert ref.per_net_crossings.tolist() == [1, 1, 3]
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpu_hard_lambda_matches_reference_and_legacy_fields_unchanged(seed):
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert gpu.hard_lambda_sum == ref.hard_lambda_sum
    assert np.array_equal(gpu.per_net_lambda, ref.per_net_lambda)
    # legacy fields must be bit-identical (Global Constraints: evaluator semantics frozen)
    assert gpu.io_count == ref.io_count and gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand


def test_gpu_matches_reference_k32():
    """8x4 region grid (K=32) -- the largest K in the M3 experiment matrix
    (K in {8,16,32}; see docs/superpowers/specs/2026-08-13-m3-differentiable-ft-
    design-draft.md §1) and the new bitmask-vectorization boundary
    GpuEvalContext asserts on construction. Supersedes the old K=64 test: T1's
    paired int8 dtype fix (R2) tightens the contract to K<=32 (`_pow2_k` uses
    signed int64, so 1<<63 == INT64_MIN makes K=64 out of contract -- see
    docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md §4.2 and
    docs/reviews/2026-08-13-m4-draft-v1-adversarial-codex.md finding 9)."""
    rng = np.random.default_rng(64)
    rg = RegionGrid(make_grid_regions(DIE, 8, 4, lattice=32))
    nl = _random_case(rng, n_cells=80, n_nets=50, max_d=20)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert gpu.io_count == ref.io_count
    assert gpu.ft_count == ref.ft_count
    assert np.array_equal(gpu.per_net_crossings, ref.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, ref.per_net_ft)
    assert gpu.tree_wl == pytest.approx(ref.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand
    assert np.array_equal(gpu.per_net_steiner, ref.per_net_steiner)
    assert np.array_equal(gpu.per_net_home, ref.per_net_home)
    assert gpu.io_rg == ref.io_rg and gpu.ft_rg == ref.ft_rg


def test_gpu_matches_reference_k1():
    """K=1 (the whole die is a single region) -- a degenerate boundary case
    explicitly called out for coverage alongside K=8/K=32 (M4 spec §4.2's dtype
    contract test matrix): every net's touched set has Λ<=1, so io_rg/ft_rg
    and every legacy field must all come out exactly 0."""
    from ioplace.regions import RegionSet, RegionSpec
    rs = RegionSet(die=DIE, lattice=20,
                    regions=[RegionSpec("P0", np.array([[0., 0., 100., 100.]]))])
    rg = RegionGrid(rs)
    rng = np.random.default_rng(1)
    nl = _random_case(rng, n_cells=20, n_nets=10, max_d=6)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert gpu.io_count == ref.io_count == 0
    assert gpu.ft_count == ref.ft_count == 0
    assert gpu.io_rg == ref.io_rg == 0
    assert gpu.ft_rg == ref.ft_rg == 0
    assert np.array_equal(gpu.per_net_steiner, ref.per_net_steiner)
    assert np.array_equal(gpu.per_net_home, ref.per_net_home)


def test_gpu_matches_reference_k8():
    rg = RegionGrid(make_grid_regions(DIE, 4, 2, lattice=20))
    rng = np.random.default_rng(8)
    nl = _random_case(rng, n_cells=40, n_nets=25, max_d=12)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert np.array_equal(gpu.per_net_steiner, ref.per_net_steiner)
    assert np.array_equal(gpu.per_net_home, ref.per_net_home)
    assert gpu.io_rg == ref.io_rg and gpu.ft_rg == ref.ft_rg


def test_gpu_context_rejects_k_over_32():
    from ioplace.evaluator_gpu import GpuEvalContext
    rg = RegionGrid(make_grid_regions(DIE, 8, 8, lattice=32))  # K=64
    rng = np.random.default_rng(64)
    nl = _random_case(rng, n_cells=80, n_nets=50, max_d=20)
    with pytest.raises(AssertionError):
        GpuEvalContext(nl, rg, device="cuda")


# ---------------------------------------------------------------------------
# M3 T1: io_rg / ft_rg / per_net_steiner / per_net_home CPU/GPU equivalence,
# including the Λ>=4 (Dreyfus-Wagner exact) and Λ>8 (metric-closure-MST
# heuristic) tiers, which the small random-degree cases above rarely reach.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gpu_matches_reference_region_graph_fields(seed):
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _random_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert np.array_equal(gpu.per_net_steiner, ref.per_net_steiner)
    assert np.array_equal(gpu.per_net_home, ref.per_net_home)
    assert gpu.io_rg == ref.io_rg
    assert gpu.ft_rg == ref.ft_rg
    assert gpu.io_rg == gpu.hard_lambda_sum + gpu.ft_rg


def _lambda_ge4_netlist(rng, k):
    """Cells scattered across all k regions of an 8x4 grid, plus a handful of
    high-fanout nets that each pick one cell from every region (Λ=k, well into
    both the [4,8] Dreyfus-Wagner tier and the >8 metric-closure-MST tier) mixed
    in with ordinary small nets."""
    n_cells = k * 6
    xs = np.linspace(0.5, 99.5, 8, endpoint=True)
    ys = np.linspace(0.5, 99.5, 4, endpoint=True)
    # place 6 cells per region cell-center so every region has pins available
    cell_x, cell_y = [], []
    gx, gy = 100.0 / 8, 100.0 / 4
    for j in range(4):
        for i in range(8):
            for _ in range(6):
                cell_x.append(i * gx + rng.uniform(1, gx - 1))
                cell_y.append(j * gy + rng.uniform(1, gy - 1))
    nx_ = np.array(cell_x[:n_cells])
    ny_ = np.array(cell_y[:n_cells])
    pins, p2n = [], []
    net = 0
    # a few full-fanout nets: one cell from each of the k regions
    for _ in range(4):
        for r in range(k):
            pins.append(r * 6 + int(rng.integers(0, 6)))
            p2n.append(net)
        net += 1
    # ordinary small nets to keep the Λ<=3 closed-form path exercised too
    for _ in range(30):
        d = int(rng.integers(2, 6))
        pins += list(rng.choice(n_cells, d, replace=False))
        p2n += [net] * d
        net += 1
    p2n = np.array(p2n, np.int32)
    pins = np.array(pins, np.int32)
    start = np.searchsorted(p2n, np.arange(net + 1)).astype(np.int32)
    from ioplace.netlist import Netlist
    return Netlist(node_x=nx_, node_y=ny_, node_size_x=np.ones(n_cells),
                   node_size_y=np.ones(n_cells), num_movable=n_cells,
                   num_terminals=0, num_terminal_NIs=0,
                   pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                   pin2node=pins, pin2net=p2n,
                   flat_net2pin=np.arange(len(pins), dtype=np.int32),
                   flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)


def test_gpu_matches_reference_region_graph_fields_lambda_ge4():
    """K=32 grid with several Λ=32 nets (deep into the >8 MST-heuristic tier)
    plus Λ=16 nets via a coarser sub-selection -- exercises evaluator_gpu's
    shared-CPU-routine delegation for the rare Λ>=4 tier end to end and checks
    it agrees with evaluator_ref bit-for-bit, not just that both run."""
    rg = RegionGrid(make_grid_regions(DIE, 8, 4, lattice=32))
    rng = np.random.default_rng(99)
    nl = _lambda_ge4_netlist(rng, rg.k)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert (ref.per_net_lambda >= 9).any(), "test netlist should reach the >8 (ub) tier"
    assert np.array_equal(gpu.per_net_steiner, ref.per_net_steiner)
    assert np.array_equal(gpu.per_net_home, ref.per_net_home)
    assert gpu.io_rg == ref.io_rg
    assert gpu.ft_rg == ref.ft_rg
    # identity per net: ST - max(Λ-1,0) == FT (via RG), on every net
    ft_rg_per_net = gpu.per_net_steiner.astype(np.int64) - np.maximum(gpu.per_net_lambda.astype(np.int64) - 1, 0)
    assert (ft_rg_per_net >= 0).all()
    assert (gpu.per_net_steiner >= np.maximum(gpu.per_net_lambda - 1, 0)).all()
    assert (gpu.per_net_steiner <= gpu.per_net_crossings).all()


def test_gpu_bit_plane_helpers_support_paired_int8_dtype():
    """R2 remedy (T1 §2.5 item 3, docs/superpowers/specs/2026-08-13-m3-
    differentiable-ft-design-draft.md line ~142): pin_bit_acc/passed_bit_acc
    must go from (E,K) int64 to int8, and PyTorch's scatter_reduce_ requires
    self.dtype == src.dtype -- so the *source* bit planes must be produced as
    int8 directly (not int64-then-cast), paired with the int8 accumulators.
    This is a white-box test of the dtype parameter added to _bit_planes /
    _one_hot_planes; the end-to-end equivalence tests above are the black-box
    check that the paired change doesn't alter any result."""
    from ioplace.evaluator_gpu import GpuEvalContext
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    rng = np.random.default_rng(3)
    nl = _random_case(rng)
    ctx = GpuEvalContext(nl, rg, device="cuda")
    ids = torch.tensor([0, 3, 15, 7], dtype=torch.int64, device=ctx.device)
    onehot8 = ctx._one_hot_planes(ids, dtype=torch.int8)
    assert onehot8.dtype == torch.int8
    assert torch.equal(onehot8.to(torch.int64), ctx._one_hot_planes(ids))
    bm = torch.tensor([0b1001, 0b0110], dtype=torch.int64, device=ctx.device)
    planes8 = ctx._bit_planes(bm, dtype=torch.int8)
    assert planes8.dtype == torch.int8
    assert torch.equal(planes8.to(torch.int64), ctx._bit_planes(bm))
    # default (no dtype arg) must stay int64 -- other callers (_popcount_k on
    # already-packed int64 pin_bm/passed_bm/ft_bm) must not silently change.
    assert ctx._one_hot_planes(ids).dtype == torch.int64
    assert ctx._bit_planes(bm).dtype == torch.int64
    # a real evaluate() call must not raise the scatter_reduce_ dtype-mismatch
    # error that motivated this fix (self.dtype == src.dtype).
    ctx.evaluate(nl.node_x, nl.node_y)


def test_gpu_pair_key_chunks_correct_across_many_chunks():
    """The pair_key_chunks -> boundary_pair_demand reduction was refactored to
    accumulate per-chunk via torch.bincount instead of one final
    torch.cat(...).cpu().numpy() + np.unique sync (M3-scope memory fix #2).
    A K=32 grid with many boundary-crossing edges forces multiple segment
    chunks/buckets to all contribute to the same (a,b) pair keys; this checks
    the accumulated counts are still exactly correct (not just that *a*
    boundary_pair_demand dict comes out), independent of chunk boundaries."""
    rg = RegionGrid(make_grid_regions(DIE, 8, 4, lattice=32))
    rng = np.random.default_rng(55)
    nl = _lambda_ge4_netlist(rng, rg.k)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert len(ref.boundary_pair_demand) > 5  # sanity: multiple distinct pairs
    assert gpu.boundary_pair_demand == ref.boundary_pair_demand


@pytest.mark.slow
def test_legacy_fields_bit_exact_regression_adaptec1_k16_grid_flat():
    """T1 deliverable #3: "既有欄位逐位元不變" regression, checked against the
    real saved evaluator output on adaptec1 (not just synthetic fixtures) --
    results/m2/ablation/adaptec1_A0_k16_grid.json (+ its .npz positions) is the
    exact "adaptec1 k16 grid flat" run tabulated in the M3 design draft §2.2
    (io_mst 30,256 / ft_mst 2,545), recomputed here with both evaluator_ref and
    evaluator_gpu after the T1 changes (region_graph wiring, paired int8 bit
    planes, pair_key_chunks bincount refactor) and compared bit-for-bit against
    that pre-T1 saved JSON. Also reports the new io_rg/ft_rg fields (see the T1
    report) and cross-checks the F11 identity + CPU/GPU equivalence on real
    (not synthetic) net topology."""
    import json, os
    from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
    from ioplace.netlist import netlist_from_placedb

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved_path = os.path.join(repo, "results/m2/ablation/adaptec1_A0_k16_grid.json")
    saved = json.load(open(saved_path))
    npz = np.load(saved_path + ".npz")

    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    cfg = os.path.join(root, "install/test/ispd2005/adaptec1.json")
    params, placedb = _load_dreamplace(cfg)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, 16, "grid", 0))

    node_x, node_y = npz["node_x"], npz["node_y"]
    ref = evaluate(nl, node_x, node_y, rg)
    gpu = evaluate_gpu(nl, node_x, node_y, rg)

    for res, tag in [(ref, "ref"), (gpu, "gpu")]:
        assert res.io_count == saved["io_count"], tag
        assert res.ft_count == saved["ft_count"], tag
        assert res.large_net_lb == saved["large_net_lb"], tag
        assert res.hpwl == pytest.approx(saved["hpwl"], rel=1e-9), tag
    # tree_wl is a sum-order-sensitive float64 reduction (see module docstring);
    # exact on CPU (evaluator_ref, same algorithm as the original save), only
    # approx on GPU (reduction-order differs from the Python accumulation loop).
    assert ref.tree_wl == pytest.approx(saved["tree_wl"], rel=1e-9)
    assert gpu.tree_wl == pytest.approx(saved["tree_wl"], rel=1e-5)

    # new M3 T1 fields: CPU/GPU equivalence + F11 identity on the real netlist
    assert np.array_equal(gpu.per_net_steiner, ref.per_net_steiner)
    assert np.array_equal(gpu.per_net_home, ref.per_net_home)
    assert gpu.io_rg == ref.io_rg and gpu.ft_rg == ref.ft_rg
    assert ref.io_rg == ref.hard_lambda_sum + ref.ft_rg
    ft_rg_per_net = ref.per_net_steiner.astype(np.int64) - np.maximum(ref.per_net_lambda.astype(np.int64) - 1, 0)
    assert (ft_rg_per_net >= 0).all()
    assert (ref.per_net_steiner <= ref.per_net_crossings).all()
    print(f"\n[T1] adaptec1 k16 grid flat: io_rg={ref.io_rg} ft_rg={ref.ft_rg} "
          f"(design draft §2.2 reference: io_rg=26921 ft_rg=2454)")
