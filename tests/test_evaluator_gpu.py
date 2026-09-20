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
    # The JSON is committed; its companion .npz of node positions is host-local
    # scratch that never was, so the golden is only checkable where that run's
    # artefacts still sit on disk.
    if not os.path.exists(saved_path + ".npz"):
        pytest.skip(f"{saved_path}.npz not present on this host "
                    "(re-run the M2 A0 ablation to regenerate it)")
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


# ---------------------------------------------------------------------------
# M4 T2: evaluator streaming (one-hot elimination, edge/segment batching,
# int32 indices with int64 composite keys). See
# docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md §4.2's
# field-specific acceptance table: integer fields (crossings/ft/lambda/
# pair_demand/steiner/home/edge counts) must be *bit-exact* across
# construction parameters (mst_chunk_budget/seg_chunk_budget/edge_batch_size);
# tree_wl/hpwl (sum-order-sensitive float64 reductions) only need rel<=1e-12.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v2 P-F (design sec 7 / spec sec 9): straddle diagnostics. The three integer
# fields and the two integer arrays are bit-exact ref-vs-GPU and across batch
# sizes / chunk budgets; the three float fields carry tree_wl/hpwl's rel<=1e-12
# contract, because they are float64 reductions whose order numpy and torch do
# not share.
# ---------------------------------------------------------------------------

_STRADDLE_INT_SCALARS = ("straddle_cells", "straddle_pin_split_nets",
                         "straddle_wide_cells")
_STRADDLE_FLOAT_SCALARS = ("straddle_area_fraction", "straddle_out_area",
                           "straddle_movable_area")


def _assert_straddle_equal(a, b, float_rel=1e-12):
    for field in _STRADDLE_INT_SCALARS:
        assert getattr(a, field) == getattr(b, field), field
    for field in _STRADDLE_FLOAT_SCALARS:
        assert getattr(a, field) == pytest.approx(getattr(b, field), rel=float_rel), field
    assert np.array_equal(a.per_node_straddle, b.per_node_straddle)
    assert np.array_equal(a.per_net_pin_split, b.per_net_pin_split)


def _assert_batch_invariant_fields(a, b, float_rel=1e-12, straddle=True):
    """a, b: two EvalResult from the same netlist/positions, different
    construction parameters (mst_chunk_budget/seg_chunk_budget/edge_batch_size).
    Every integer/structural field must be bit-exact; tree_wl/hpwl only need
    rel<=float_rel (spec §4.2's pre-registered tolerance, not bit-exact --
    see the module docstring and T2's empirical evidence for why)."""
    assert a.io_count == b.io_count
    assert a.ft_count == b.ft_count
    assert a.hard_lambda_sum == b.hard_lambda_sum
    assert a.large_net_lb == b.large_net_lb
    assert np.array_equal(a.per_net_crossings, b.per_net_crossings)
    assert np.array_equal(a.per_net_ft, b.per_net_ft)
    assert np.array_equal(a.per_net_lambda, b.per_net_lambda)
    assert np.array_equal(a.per_net_steiner, b.per_net_steiner)
    assert np.array_equal(a.per_net_home, b.per_net_home)
    assert a.io_rg == b.io_rg
    assert a.ft_rg == b.ft_rg
    assert a.boundary_pair_demand == b.boundary_pair_demand
    assert a.tree_wl == pytest.approx(b.tree_wl, rel=float_rel)
    assert a.hpwl == pytest.approx(b.hpwl, rel=float_rel)
    if straddle:
        # v2 P-F: same split as everything above -- integers bit-exact, the
        # three float64 reductions at float_rel. `straddle=False` is only for
        # the one test that deliberately compares a straddle-on run against a
        # straddle-off one.
        for field in _STRADDLE_INT_SCALARS:
            assert getattr(a, field) == getattr(b, field), field
        assert np.array_equal(a.per_node_straddle, b.per_node_straddle)
        assert np.array_equal(a.per_net_pin_split, b.per_net_pin_split)
        for field in _STRADDLE_FLOAT_SCALARS:
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=float_rel), field


def test_gpu_evaluate_batch_invariance_small_batches():
    """T2 field-specific acceptance, fast/synthetic form: construction
    parameters that force many small chunks (mst_chunk_budget/seg_chunk_budget/
    edge_batch_size all tiny, well below this case's edge/segment counts) must
    give bit-exact integer fields and rel<=1e-12 tree_wl/hpwl vs. a single
    huge-batch ("process everything in one shot") reference run -- i.e. the
    T2 batching refactor changes memory shape only, not results. Uses
    _lambda_ge4_netlist (K=32, several high-fanout Λ>=4 nets mixed with small
    nets) so multiple MST degree buckets and multiple segment-length buckets
    are all exercised by the chunking, not just the common-case small nets."""
    rg = RegionGrid(make_grid_regions(DIE, 8, 4, lattice=32))
    rng = np.random.default_rng(123)
    nl = _lambda_ge4_netlist(rng, rg.k)

    from ioplace.evaluator_gpu import GpuEvalContext
    ref_ctx = GpuEvalContext(nl, rg, device="cuda",
                              mst_chunk_budget=10**9, seg_chunk_budget=10**9,
                              edge_batch_size=10**9)
    ref = ref_ctx.evaluate(nl.node_x, nl.node_y)

    for mst_b, seg_b, edge_b in [(3, 3, 3), (7, 11, 5), (1, 4, 2)]:
        ctx = GpuEvalContext(nl, rg, device="cuda",
                              mst_chunk_budget=mst_b, seg_chunk_budget=seg_b,
                              edge_batch_size=edge_b)
        got = ctx.evaluate(nl.node_x, nl.node_y)
        _assert_batch_invariant_fields(ref, got)
        # also lock in against evaluator_ref, not just self-consistency
        cpu = evaluate(nl, nl.node_x, nl.node_y, rg)
        assert got.io_count == cpu.io_count
        assert got.ft_count == cpu.ft_count
        assert np.array_equal(got.per_net_crossings, cpu.per_net_crossings)
        assert np.array_equal(got.per_net_ft, cpu.per_net_ft)
        assert got.boundary_pair_demand == cpu.boundary_pair_demand


@pytest.mark.parametrize("k_shape", [
    (1, 1, 20),   # K=1: degenerate single-region grid
    (4, 2, 20),   # K=8
    (8, 4, 32),   # K=32
])
def test_gpu_evaluate_k_multichunk_empty_degree_buckets(k_shape):
    """T2 acceptance: 'K in {1,8,32} x multi-chunk x empty bucket x legacy
    field regression'. Builds a netlist with an explicit *gap* in the net
    degree distribution (only degree 2 and degree 40 nets -- degrees 3..39
    are all empty _buckets entries, exercising the `if len(net_ids)==0:
    continue` path robustly rather than incidentally), evaluates with tiny
    chunk budgets (forcing every bucket that *is* non-empty through multiple
    chunks), and checks bit-exact agreement with evaluator_ref."""
    from ioplace.netlist import Netlist
    from ioplace.evaluator_gpu import GpuEvalContext
    rows, cols, lattice = k_shape
    if rows == 1 and cols == 1:
        from ioplace.regions import RegionSet, RegionSpec
        rg = RegionGrid(RegionSet(die=DIE, lattice=lattice,
                                   regions=[RegionSpec("P0", np.array([[0., 0., 100., 100.]]))]))
    else:
        rg = RegionGrid(make_grid_regions(DIE, rows, cols, lattice=lattice))

    rng = np.random.default_rng(7)
    n_cells = 300
    node_x = rng.uniform(0.5, 99.5, n_cells)
    node_y = rng.uniform(0.5, 99.5, n_cells)
    degree_plan = [2] * 40 + [40] * 5  # degrees 3..39 are an empty gap in _buckets
    pins, p2n = [], []
    for net, d in enumerate(degree_plan):
        pins += list(rng.choice(n_cells, d, replace=False))
        p2n += [net] * d
    p2n = np.array(p2n, np.int32)
    pins = np.array(pins, np.int32)
    start = np.searchsorted(p2n, np.arange(len(degree_plan) + 1)).astype(np.int32)
    nl = Netlist(node_x=node_x, node_y=node_y, node_size_x=np.ones(n_cells),
                 node_size_y=np.ones(n_cells), num_movable=n_cells,
                 num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(len(pins)), pin_offset_y=np.zeros(len(pins)),
                 pin2node=pins, pin2net=p2n,
                 flat_net2pin=np.arange(len(pins), dtype=np.int32),
                 flat_net2pin_start=start, xl=0., yl=0., xh=100., yh=100.)

    cpu = evaluate(nl, nl.node_x, nl.node_y, rg)
    ctx = GpuEvalContext(nl, rg, device="cuda",
                          mst_chunk_budget=5, seg_chunk_budget=5, edge_batch_size=3)
    gpu = ctx.evaluate(nl.node_x, nl.node_y)

    assert gpu.io_count == cpu.io_count
    assert gpu.ft_count == cpu.ft_count
    assert np.array_equal(gpu.per_net_crossings, cpu.per_net_crossings)
    assert np.array_equal(gpu.per_net_ft, cpu.per_net_ft)
    assert gpu.tree_wl == pytest.approx(cpu.tree_wl, rel=1e-5)
    assert gpu.boundary_pair_demand == cpu.boundary_pair_demand
    assert gpu.hard_lambda_sum == cpu.hard_lambda_sum
    assert np.array_equal(gpu.per_net_lambda, cpu.per_net_lambda)
    assert np.array_equal(gpu.per_net_steiner, cpu.per_net_steiner)
    assert np.array_equal(gpu.per_net_home, cpu.per_net_home)
    assert gpu.io_rg == cpu.io_rg and gpu.ft_rg == cpu.ft_rg
    # structural self-check (spec §4.2: "MST edge total / per-net edge count
    # must be bit-exact" -- guards against a batch boundary silently
    # dropping/duplicating an edge): total MST edges is a pure function of
    # net degrees (sum(d-1) over the two-or-more-pin nets), independent of
    # chunk/batch size, and per_net_crossings' sum (io_count) already being
    # exact above is the direct evidence no edge was lost or double-counted.
    expected_edges = sum(d - 1 for d in degree_plan)
    edge_net_id, _, _ = ctx._batch_mst(
        *ctx._pin_positions(torch.as_tensor(nl.node_x, dtype=torch.float64, device=ctx.device),
                             torch.as_tensor(nl.node_y, dtype=torch.float64, device=ctx.device)))
    assert edge_net_id.numel() == expected_edges


@pytest.mark.slow
def test_gpu_evaluate_batch_invariance_bigblue4_k32_1e6_8e6_all():
    """T2's literal field-specific acceptance test (spec §4.2 T2 row): batch
    size in {1e6, 8e6, "all edges in one batch"} on a real, large-enough case
    that these settings actually produce a different number of MST-edge
    batches (bigblue4 K=32 has ~6.4M MST edges -- see the printed count) --
    unlike a small synthetic case, this exercises exactly the field-specific
    contract that matters at production scale: integer fields bit-exact,
    tree_wl/hpwl rel<=1e-12."""
    import os
    from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
    from ioplace.netlist import netlist_from_placedb
    from ioplace.evaluator_gpu import GpuEvalContext

    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    cfg = os.path.join(root, "install/test/ispd2005/bigblue4.json")
    params, placedb = _load_dreamplace(cfg)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, 32, "grid", 0))

    results = {}
    for tag, edge_batch in [("1e6", 1_000_000), ("8e6", 8_000_000), ("all", 10**9)]:
        ctx = GpuEvalContext(nl, rg, device="cuda", edge_batch_size=edge_batch)
        results[tag] = ctx.evaluate(nl.node_x, nl.node_y)

    print(f"\n[T2] bigblue4 K=32 batch invariance: MST edges under 1e6 batching "
          f"produced multiple batches (edge_batch_size=1e6 < total edges)")

    _assert_batch_invariant_fields(results["1e6"], results["8e6"])
    _assert_batch_invariant_fields(results["1e6"], results["all"])
    _assert_batch_invariant_fields(results["8e6"], results["all"])

    # cross-check against evaluator_ref too (existing rel<=1e-5 tree_wl contract)
    cpu = evaluate(nl, nl.node_x, nl.node_y, rg)
    for tag, res in results.items():
        assert res.io_count == cpu.io_count, tag
        assert res.ft_count == cpu.ft_count, tag
        assert np.array_equal(res.per_net_crossings, cpu.per_net_crossings), tag
        assert np.array_equal(res.per_net_ft, cpu.per_net_ft), tag
        assert res.tree_wl == pytest.approx(cpu.tree_wl, rel=1e-5), tag


# baselines measured on this host (L4, torch 2.8.0+cu128), max_memory_allocated,
# fresh subprocess per measurement (M2/M4 spec B1: peak stats pollute across
# arms in the same process, so cross-arm comparisons must be cross-process):
#   pre-M3-T1 (commit 794cd85, original (P,K) int64 one-hot):  8.354808330535889 GB
#   post-M3-T1 / pre-T2 HEAD (commit 0a2e32c..da818f4, paired
#     int8 accumulators but still a (P,K) int8 one-hot source):  6.244536399841309 GB
_BIGBLUE4_K32_PRE_M3T1_PEAK_GB = 8.354808330535889
_BIGBLUE4_K32_PRE_T2_HEAD_PEAK_GB = 6.244536399841309


@pytest.mark.slow
def test_gpu_evaluator_memory_bigblue4_k32_reduction():
    """T2 acceptance: bigblue4 K=32 peak memory reduction >=60%, reported
    against both baselines the M4 T2 instruction asked for (pre-M3-T1's
    original one-hot int64 implementation, and post-M3-T1/pre-T2 HEAD -- see
    the module-level constants above for provenance)."""
    import os
    from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
    from ioplace.netlist import netlist_from_placedb
    from ioplace.evaluator_gpu import GpuEvalContext

    root = os.environ.get("DREAMPLACE_ROOT", "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace")
    cfg = os.path.join(root, "install/test/ispd2005/bigblue4.json")
    params, placedb = _load_dreamplace(cfg)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, 32, "grid", 0))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ctx = GpuEvalContext(nl, rg, device="cuda")  # default edge_batch_size=1_000_000
    ctx.evaluate(nl.node_x, nl.node_y)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    reduction_vs_pre_m3t1 = 1.0 - peak_gb / _BIGBLUE4_K32_PRE_M3T1_PEAK_GB
    reduction_vs_pre_t2_head = 1.0 - peak_gb / _BIGBLUE4_K32_PRE_T2_HEAD_PEAK_GB
    print(f"\n[T2] bigblue4 K=32 peak_alloc={peak_gb:.4f}GB "
          f"reduction vs pre-M3-T1={reduction_vs_pre_m3t1:.1%} "
          f"reduction vs pre-T2 HEAD={reduction_vs_pre_t2_head:.1%}")

    assert reduction_vs_pre_m3t1 >= 0.60
    assert reduction_vs_pre_t2_head >= 0.60


@pytest.mark.slow
def test_gpu_evaluator_memory_mempool_group_k32_under_2gb():
    """T2 acceptance (M4-G1 gate): mempool_group (3.5M nets) K=32 real-run
    peak <=2GB. Skips (does not fail) if the ISPD2025 corpus/config isn't
    present on this host -- that corpus is produced by a parallel M4 T3
    task, not this one."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = os.path.join(repo, "benchmarks/ispd25/mempool_group.json")
    if not os.path.exists(cfg):
        pytest.skip(f"{cfg} not present (produced by the M4 T3 task)")

    from ioplace.dreamplace_env import setup_dreamplace
    setup_dreamplace()
    import Params, PlaceDB
    from ioplace.netlist import netlist_from_placedb
    from ioplace.drivers.run_placement import get_regions_for
    from ioplace.evaluator_gpu import GpuEvalContext

    params = Params.Params()
    params.load(cfg)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    placedb.initialize(params)
    nl = netlist_from_placedb(placedb)
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    rg = RegionGrid(get_regions_for(die, 32, "grid", 0))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ctx = GpuEvalContext(nl, rg, device="cuda")  # default edge_batch_size=1_000_000
    ctx.evaluate(nl.node_x, nl.node_y)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    print(f"\n[T2] mempool_group K=32 (n_nets={nl.num_nets}) peak_alloc={peak_gb:.4f}GB (gate: <=2GB)")
    assert peak_gb <= 2.0


def _sized_case(rng, n_cells=60, n_nets=40, max_d=10, size=6.0):
    """_random_case with cells big enough (6.0 on a 100-wide die cut at
    25/50/75) that a solid fraction of them straddles -- the default unit-size
    cells only straddle by accident."""
    nl = _random_case(rng, n_cells=n_cells, n_nets=n_nets, max_d=max_d)
    nl.node_size_x = np.full(n_cells, size)
    nl.node_size_y = np.full(n_cells, size)
    return nl


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gpu_straddle_matches_reference(seed):
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.straddle_cells > 0          # the case actually exercises the path
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_on_the_hand_built_corner_case():
    """The same layout tests/test_straddle.py pins by hand, so ref and GPU are
    both nailed to known numbers rather than only to each other."""
    from tests.test_straddle import _corner_case
    nl = _corner_case()
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.straddle_cells == 2 and ref.straddle_pin_split_nets == 1
    assert ref.per_net_pin_split.tolist() == [-1, 0, 1, -1]
    assert ref.straddle_area_fraction == pytest.approx(20.0 / 48.0)
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_with_fixed_nodes_and_pin_offsets():
    """num_movable < num_physical and non-zero pin offsets together: the
    terminal tail must stay out of both the geometry and the re-attribution."""
    rng = np.random.default_rng(7)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng, n_cells=50, n_nets=30)
    nl.num_movable = 35
    nl.num_terminals = 15
    nl.pin_offset_x = rng.uniform(0., 6., len(nl.pin2node))
    nl.pin_offset_y = rng.uniform(0., 6., len(nl.pin2node))
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.per_node_straddle[35:].sum() == 0
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_on_lattice_boundaries():
    """C1 again, for the corner lookups: the lattice-line coordinate
    xl + (ix0+1)*cell_w must be built from the 0-dim cell-size tensors, or the
    quadrant split lands one ULP off exactly at a region boundary."""
    from ioplace.netlist import Netlist
    DIE_ND = (0., 0., 10692., 10680.)
    rg = RegionGrid(make_grid_regions(DIE_ND, 4, 4, lattice=512))
    # cell_w = 10692/512 = 20.8828125; region columns break at lattice 128/256/384,
    # i.e. x = 2673.0 / 5346.0 / 8019.0. Each cell straddles one of them.
    node_x = np.array([2670.0, 5340.0, 8010.0, 1000.0])
    node_y = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=np.full(4, 20.0), node_size_y=np.full(4, 20.0),
                 num_movable=4, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(4), pin_offset_y=np.zeros(4),
                 pin2node=np.array([0, 1, 2, 3], np.int32),
                 pin2net=np.array([0, 0, 1, 1], np.int32),
                 flat_net2pin=np.arange(4, dtype=np.int32),
                 flat_net2pin_start=np.array([0, 2, 4], np.int32),
                 xl=0., yl=0., xh=10692., yh=10680.)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.straddle_cells == 3
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_on_edge_exact_boundaries():
    """Review fix round 1 (I1): the lattice-boundaries test above only makes
    cells *cross* a boundary; convention 1's closed-box `<=` (a cell whose
    right/top edge lands EXACTLY on a region line counts as straddling, but a
    cell whose left/bottom edge lands exactly on one does not) was previously
    pinned only in tests/test_straddle.py's numpy-only fixtures, never
    ref-vs-GPU. DIE=(0,0,100,100), lattice=10 -> cell_w=cell_h=10; a 2x2
    region grid puts the boundary at x=y=50 (same grid tests/test_straddle.py
    ::_grid builds)."""
    from ioplace.netlist import Netlist
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))
    node_x = np.array([46.0, 50.0, 46.0, 45.0])
    node_y = np.array([10.0, 10.0, 46.0, 45.0])
    node_size = np.array([4.0, 4.0, 4.0, 10.0])
    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=node_size, node_size_y=node_size,
                 num_movable=4, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(4), pin_offset_y=np.zeros(4),
                 pin2node=np.array([0, 1, 2, 3], np.int32),
                 pin2net=np.array([0, 0, 1, 1], np.int32),
                 flat_net2pin=np.arange(4, dtype=np.int32),
                 flat_net2pin_start=np.array([0, 2, 4], np.int32),
                 xl=0., yl=0., xh=100., yh=100.)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    # node 0: right edge exactly on x=50 -> straddles (closed box).
    # node 1: left edge exactly on x=50 -> does not spuriously straddle.
    # node 2: both the x=50 and y=50 lines exactly on its edges -> straddles.
    # node 3: 10x10 cell centred exactly on the boundary intersection (50,50).
    assert ref.per_node_straddle.tolist() == [1, 0, 1, 1]
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_on_a_nonzero_noninteger_die_origin():
    """Review fix round 1 (I1): every other parity test uses xl=yl=0, so the
    `self.xl + (ix0+1)*cell_w` cut-line term (and the plain `self.xl`/`self.yl`
    offsets `_to_idx` divides by) are never exercised against a non-zero,
    non-integer die origin -- real DEF diearea boxes routinely have one.
    DIE=(17.3,4.7,217.3,204.7), 3x3 regions, lattice=30 -> cell_w=cell_h=
    200/30 (not exactly representable in binary64), and the 3-way partition's
    boundary falls exactly at lattice index 10 (30/3), i.e. xl+10*cell_w /
    yl+10*cell_h -- so the boundary itself is also off the die origin and
    built from an inexact cell size, same as the four boundary-exact cases in
    test_gpu_straddle_matches_reference_on_edge_exact_boundaries above, just
    translated and rescaled onto this die."""
    from ioplace.netlist import Netlist
    DIE_ND = (17.3, 4.7, 217.3, 204.7)
    rg = RegionGrid(make_grid_regions(DIE_ND, 3, 3, lattice=30))
    xl, yl = DIE_ND[0], DIE_ND[1]
    cw, ch = rg.cell_w, rg.cell_h
    bx, by = xl + 10 * cw, yl + 10 * ch          # exact region-column/row line
    sx, sy = 0.4 * cw, 0.4 * ch                  # same 0.4-of-a-lattice-cell
                                                  # proportions as the integer
                                                  # boundary test above (4/10)
    safe_y = yl + 3 * ch                          # well inside the first row
    node_x = np.array([bx - sx, bx, bx - sx, bx - 0.5 * cw])
    node_y = np.array([safe_y, safe_y, by - sy, by - 0.5 * ch])
    node_size_x = np.array([sx, sx, sx, cw])
    node_size_y = np.array([sy, sy, sy, ch])
    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=node_size_x, node_size_y=node_size_y,
                 num_movable=4, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(4), pin_offset_y=np.zeros(4),
                 pin2node=np.array([0, 1, 2, 3], np.int32),
                 pin2net=np.array([0, 0, 1, 1], np.int32),
                 flat_net2pin=np.arange(4, dtype=np.int32),
                 flat_net2pin_start=np.array([0, 2, 4], np.int32),
                 xl=DIE_ND[0], yl=DIE_ND[1], xh=DIE_ND[2], yh=DIE_ND[3])
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.straddle_cells > 0     # the case actually exercises the path
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_bit_exact_across_batch_sizes_and_chunk_budgets():
    from ioplace.evaluator_gpu import GpuEvalContext
    rg = RegionGrid(make_grid_regions(DIE, 8, 4, lattice=32))
    rng = np.random.default_rng(123)
    nl = _lambda_ge4_netlist(rng, rg.k)
    nl.node_size_x = np.full(len(nl.node_x), 5.0)
    nl.node_size_y = np.full(len(nl.node_x), 5.0)
    ref_ctx = GpuEvalContext(nl, rg, device="cuda", mst_chunk_budget=10**9,
                             seg_chunk_budget=10**9, edge_batch_size=10**9)
    ref = ref_ctx.evaluate(nl.node_x, nl.node_y)
    assert ref.straddle_cells > 0
    cpu = evaluate(nl, nl.node_x, nl.node_y, rg)
    for mst_b, seg_b, edge_b in [(3, 3, 3), (7, 11, 5), (1, 4, 2)]:
        got = GpuEvalContext(nl, rg, device="cuda", mst_chunk_budget=mst_b,
                             seg_chunk_budget=seg_b,
                             edge_batch_size=edge_b).evaluate(nl.node_x, nl.node_y)
        _assert_batch_invariant_fields(ref, got)     # incl. the straddle block
        _assert_straddle_equal(cpu, got)             # and against evaluator_ref


@pytest.mark.parametrize("k_shape", [(1, 1, 20), (4, 2, 20), (8, 4, 32)])
def test_gpu_straddle_matches_reference_for_k1_k8_k32(k_shape):
    rows, cols, lattice = k_shape
    rng = np.random.default_rng(5)
    if rows == 1 and cols == 1:
        from ioplace.regions import RegionSet, RegionSpec
        rs = RegionSet(die=DIE, lattice=lattice,
                       regions=[RegionSpec("P0", np.array([[0., 0., 100., 100.]]))])
    else:
        rs = make_grid_regions(DIE, rows, cols, lattice=lattice)
    rg = RegionGrid(rs)
    nl = _sized_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    if rg.k == 1:
        assert ref.straddle_cells == 0     # one region: nothing can straddle
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_off_leaves_the_legacy_fields_bit_identical():
    from ioplace.evaluator_gpu import GpuEvalContext
    rng = np.random.default_rng(1)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng)
    on = GpuEvalContext(nl, rg, device="cuda").evaluate(nl.node_x, nl.node_y)
    off = GpuEvalContext(nl, rg, device="cuda",
                         straddle=False).evaluate(nl.node_x, nl.node_y)
    _assert_batch_invariant_fields(on, off, straddle=False)
    assert off.straddle_cells == 0 and off.per_node_straddle is None
    assert on.straddle_cells > 0


def test_gpu_per_call_straddle_switch_matches_the_context_default():
    from ioplace.evaluator_gpu import GpuEvalContext
    rng = np.random.default_rng(2)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng)
    ctx = GpuEvalContext(nl, rg, device="cuda")
    _assert_straddle_equal(ctx.evaluate(nl.node_x, nl.node_y),
                           ctx.evaluate(nl.node_x, nl.node_y, straddle=True))
    assert ctx.evaluate(nl.node_x, nl.node_y, straddle=False).per_node_straddle is None
    lean = GpuEvalContext(nl, rg, device="cuda", straddle=False)
    with pytest.raises(ValueError, match="straddle=False"):
        lean.evaluate(nl.node_x, nl.node_y, straddle=True)
