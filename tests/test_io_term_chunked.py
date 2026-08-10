import json, os
import numpy as np
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from ioplace.regions import make_grid_regions
from ioplace.ops.soft_assign import rect_table
from ioplace.ops.io_term import build_net_node_csr, IoTerm, IoTermRef
from tests.test_io_term import _nl, _pos, DIE

def _pair(nl, rs, K, chunk_budget, w_mode="unit", num_movable=None, n_filler=0):
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    kw = dict(csr=csr, rects=rects, rect2region=r2k, K=K,
              num_movable=nl.num_movable if num_movable is None else num_movable,
              num_physical=nl.num_physical, num_nodes=nl.num_physical + n_filler,
              device="cuda", w_mode=w_mode)
    return IoTerm(chunk_budget=chunk_budget, **kw), IoTermRef(**kw)

def _case(seed, n_nodes=60, n_nets=30):
    rng = np.random.default_rng(seed)
    xy = [(float(a), float(b)) for a, b in zip(rng.uniform(2, 98, n_nodes),
                                               rng.uniform(2, 98, n_nodes))]
    nets = [sorted(rng.choice(n_nodes, int(rng.integers(2, 8)), replace=False).tolist())
            for _ in range(n_nets)]
    return _nl(xy, nets)

@pytest.mark.parametrize("w_mode", ["unit", "inv_deg"])
@pytest.mark.parametrize("budget", [1, 60, 240, 10_000_000])
@pytest.mark.parametrize("seed", [0, 1])
def test_chunked_matches_reference_value_and_gradient(budget, seed, w_mode):
    """w_mode parametrized per Task 2b review I2: IoTerm.__init__ duplicates
    IoTermRef's inv_deg weight construction (1/(deg'-1)) with no equivalence
    coverage before this."""
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    nl = _case(seed)
    fast, ref = _pair(nl, rs, 16, budget, w_mode=w_mode)
    pf, pr = _pos(nl), _pos(nl)
    Lf = fast(pf, 11.0, 2.5, lambda_margin=0.5, margin_m=4.0, margin_tau=2.0)
    Lr = ref(pr, 11.0, 2.5, lambda_margin=0.5, margin_m=4.0, margin_tau=2.0)
    assert float(Lf) == pytest.approx(float(Lr), rel=1e-10)
    Lf.backward(); Lr.backward()
    assert torch.allclose(pf.grad, pr.grad, rtol=1e-9, atol=1e-12)

@pytest.mark.parametrize("k_chunk_target", [3, 5, 7])
@pytest.mark.parametrize("seed", [0, 1])
def test_chunked_matches_reference_nondivisor_chunk_sizes(k_chunk_target, seed):
    """Coordinator requirement (T2b addition 1): K=16 with chunk sizes that do
    NOT divide it evenly, so chunk_p_ell's k_lo != 0 path and a genuine
    trailing partial chunk are both exercised (no other call site does this).
    budget is computed from the *same* denom formula IoTerm.__init__ uses, so
    k_chunk lands exactly on the target instead of drifting with problem size.
    """
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    nl = _case(seed)
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    denom = max(nl.num_physical, len(csr.flat_net2node))
    budget = k_chunk_target * denom
    kw = dict(csr=csr, rects=rects, rect2region=r2k, K=16,
              num_movable=nl.num_movable, num_physical=nl.num_physical,
              num_nodes=nl.num_physical, device="cuda", w_mode="unit")
    fast = IoTerm(chunk_budget=budget, **kw)
    ref = IoTermRef(**kw)
    assert fast.k_chunk == k_chunk_target
    pf, pr = _pos(nl), _pos(nl)
    Lf = fast(pf, 11.0, 2.5, lambda_margin=0.5, margin_m=4.0, margin_tau=2.0)
    Lr = ref(pr, 11.0, 2.5, lambda_margin=0.5, margin_m=4.0, margin_tau=2.0)
    assert float(Lf) == pytest.approx(float(Lr), rel=1e-10)
    Lf.backward(); Lr.backward()
    assert torch.allclose(pf.grad, pr.grad, rtol=1e-9, atol=1e-12)

def test_chunk_budget_actually_bounds_allocation():
    """The contract: no (N,K) / (P,K) tensor is ever materialized."""
    rs = make_grid_regions(DIE, 8, 4, lattice=32)
    nl = _case(2, n_nodes=200, n_nets=120)
    fast, _ = _pair(nl, rs, 32, chunk_budget=400)
    pos = _pos(nl)
    fast(pos, 9.0, 1.0).backward()
    n_pins = int(fast.node_idx.numel())
    assert fast.last_peak_chunk_elems <= 400 * 2
    assert fast.last_peak_chunk_elems < nl.num_physical * 32
    assert fast.last_peak_chunk_elems < n_pins * 32

def test_chunked_diagnostics_match_reference():
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    nl = _case(3)
    fast, ref = _pair(nl, rs, 16, 120)
    pos = _pos(nl).detach()
    a, b = fast.diagnostics(pos, 8.0), ref.diagnostics(pos, 8.0)
    assert a["soft_lambda_sum"] == pytest.approx(b["soft_lambda_sum"], rel=1e-9)
    assert a["frac_soft"] == pytest.approx(b["frac_soft"], rel=1e-12)
    assert np.allclose(a["grad_share"], b["grad_share"], rtol=1e-6, atol=1e-9)
    assert fast.io_grad_l1(pos, 8.0) == pytest.approx(ref.io_grad_l1(pos, 8.0), rel=1e-9)

def test_chunked_zero_gradient_for_fixed_and_filler():
    """Task 2b review I3: IoTermRef zeroes fixed/filler gradient via an
    explicit detach-and-cat (_split_xy); IoTerm zeroes it by construction in
    _IoFn.backward's gx[num_movable:]=0. Provably the same result, but
    previously only IoTerm's own zeros were checked -- cross-check the two
    classes' *full* gradient vectors, not just the zero entries."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    nl = _nl([(25., 25.), (75., 25.), (25., 75.)], [[0, 1], [1, 2]])
    fast, ref = _pair(nl, rs, 4, 8, num_movable=2, n_filler=5)
    n_all = nl.num_physical + 5
    pos_f, pos_r = _pos(nl, n_filler=5), _pos(nl, n_filler=5)
    fast(pos_f, 12.0, 1.0).backward()
    ref(pos_r, 12.0, 1.0).backward()
    assert float(pos_f.grad[2].abs()) == 0.0 and float(pos_f.grad[n_all + 2].abs()) == 0.0
    assert float(pos_f.grad[nl.num_physical:n_all].abs().sum()) == 0.0
    assert torch.allclose(pos_f.grad, pos_r.grad, rtol=1e-9, atol=1e-12)

def test_chunked_no_nan_on_saturated_case():
    rs = make_grid_regions(DIE, 2, 1, lattice=10)
    nl = _nl([(5., 50.), (6., 50.), (7., 50.), (8., 50.)], [[0, 1, 2, 3]])
    fast, _ = _pair(nl, rs, 2, 2)
    pos = _pos(nl)
    fast(pos, 1e-3, 1.0).backward()
    assert torch.isfinite(pos.grad).all()

def test_float32_pos_is_supported_and_close_to_fp64():
    rs = make_grid_regions(DIE, 4, 4, lattice=20)
    nl = _case(4)
    fast, _ = _pair(nl, rs, 16, 200)
    p64 = _pos(nl)
    p32 = p64.detach().float().requires_grad_(True)
    fast(p64, 10.0, 1.0).backward()
    fast(p32, 10.0, 1.0).backward()
    assert torch.allclose(p32.grad.double(), p64.grad, rtol=1e-4, atol=1e-6)

@pytest.mark.slow
def test_chunk_budget_bounds_real_measured_memory():
    """Task 2b review I1: test_chunk_budget_actually_bounds_allocation only
    checks last_peak_chunk_elems against the closed-form it is itself
    computed from -- tautological, not a memory measurement. This uses
    torch.cuda.max_memory_allocated() on a ~200k-node synthetic (reusing
    spike_10m's own netlist synthesis, so the connectivity is production-
    shaped, not this file's tiny hand-built fixtures) at a scale where the
    (N,K)-vs-chunked gap actually shows up in real allocator behavior, for
    both the fwd+bwd hot path and diagnostics()."""
    from ioplace.diagnostics.spike_10m import _synthesize_netlist
    from ioplace.drivers.run_placement import get_regions_for
    die = (0.0, 0.0, 10_000.0, 10_000.0)
    nl, _ = _synthesize_netlist(200_000, 240_000, die)
    rs = get_regions_for(die, 32, "grid", 0)
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    kw = dict(csr=csr, rects=rects, rect2region=r2k, K=32,
              num_movable=nl.num_movable, num_physical=nl.num_physical,
              num_nodes=nl.num_physical, device="cuda")

    def _measure(chunk_budget, use_diagnostics):
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        term = IoTerm(chunk_budget=chunk_budget, **kw)
        pos = torch.cat([torch.as_tensor(nl.node_x), torch.as_tensor(nl.node_y)]).cuda()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        if use_diagnostics:
            term.diagnostics(pos, 10.0)
        else:
            pos = pos.requires_grad_(True)
            term(pos, 10.0, 1.0).backward()
        torch.cuda.synchronize()
        peak, k_chunk = torch.cuda.max_memory_allocated(), term.k_chunk
        del term, pos
        torch.cuda.empty_cache()
        return peak, k_chunk

    peak_fb_c1, k1 = _measure(1, use_diagnostics=False)
    peak_fb_cK, kK = _measure(10**12, use_diagnostics=False)
    assert k1 == 1 and kK == 32
    assert peak_fb_c1 < 0.25 * peak_fb_cK, (peak_fb_c1, peak_fb_cK)

    peak_diag_c1, _ = _measure(1, use_diagnostics=True)
    peak_diag_cK, _ = _measure(10**12, use_diagnostics=True)
    assert peak_diag_c1 < 0.25 * peak_diag_cK, (peak_diag_c1, peak_diag_cK)

@pytest.mark.slow
def test_spike_10m_report_exists_and_is_within_budget():
    p = "results/m2/spike/spike_10m.json"
    if not os.path.exists(p):
        pytest.skip("run ioplace/diagnostics/spike_10m.py first (Task 2b Step 5)")
    d = json.load(open(p))
    assert d["peak_gb"] <= 8.0
    assert d["ok"] is True
