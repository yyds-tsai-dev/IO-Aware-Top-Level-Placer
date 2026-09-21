"""GPU (torch) evaluator, semantically equivalent to ioplace.evaluator_ref.evaluate.

GpuEvalContext caches everything that only depends on the (static) netlist/region
topology -- the region grid, its Ph/Pv crossing-count prefix sums, and per-net-degree
pin index buckets -- so repeated `.evaluate(node_x, node_y)` calls (e.g. from the
Task 12 reweighting loop) only redo position-dependent work.

Numerically, all position/geometry arithmetic is done in float64 to match
evaluator_ref's numpy float64 path on the elementwise operations that decide
*integer* outcomes (grid indices, MST edge selection): pin position gathers, `_to_idx`,
and MST Manhattan distances are all simple elementwise ops, so float64 CPU (numpy) and
float64 GPU (torch/CUDA) give bit-identical results for them. Only the final tree_wl
reduction is sum-order-sensitive (GPU reduction order != Python loop accumulation
order), which is exactly why the equivalence test compares tree_wl with
pytest.approx(rel=1e-5) while everything else (crossings, ft, pair_demand -- all
derived from integer grid-index comparisons) must match exactly. tree_wl/hpwl carry a
*second*, independent tolerance contract vs. evaluator_gpu itself across different
construction parameters (edge_batch_size/mst_chunk_budget/seg_chunk_budget): rel<=1e-12,
not bit-exact -- float64 addition is not associative, so a different batch size can
land on a different (still correctly-rounded) reduction order/result; see
docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md §4.2 for the empirical
evidence this bound is based on. Every *integer* field (crossings/ft/lambda/pair_demand/
steiner/home, and MST/segment edge counts) is required to be bit-exact across those same
construction parameters -- integer accumulation is exact and order-independent by
construction here (plain addition / amax-OR / index_add_ of disjoint bits), so this is
not a looser empirical bound like tree_wl/hpwl's, it is a correctness invariant.

M4 T2 (structural streaming, see spec §4.2 items 3-7) eliminated two tensors that used
to scale with pin/edge count rather than net count: the (P,K) one-hot `pin_bits` used to
derive pin_bm (replaced by a `torch.unique` over a packed (net_id,region_id) composite
key + `index_add_` of single-bit values -- exact OR because the keys are already deduped
before being summed) and the dense (n_nets,K) `home_counts` used for the home-region
argmax (replaced by an injective count/region score encoding reduced with the same
`scatter_reduce_(amax)` pattern already used elsewhere in this file, off the same unique
pass). The per-edge geometry/crossing/FT/pair-demand pipeline in `evaluate()` is now
batched over `edge_batch_size` MST edges at a time (a construction parameter, independent
from `mst_chunk_budget`/`seg_chunk_budget`), bounding those tensors' peak size instead of
letting them scale with the full MST edge count. Every index tensor (pin2node/pin2net,
MST edge arrays, lattice coordinates) is int32, asserted against its index space's bound
in `__init__`; the one composite key in this file that packs two index spaces into one
integer (net_id*64+region_id, both here and in `_process_segments`' boundary-pair-demand
key) stays int64 (with its own headroom assert), since collapsing it to int32 would make
distinct (net,region) pairs collide.

Algorithm (see docs/superpowers/plans/2026-07-31-m0-m1-foundations-and-evaluator.md,
Task 11):
  - MST: nets are bucketed by *exact* pin-degree d (2 <= d <= max_degree); each bucket
    runs a batched dense Prim ((B,d,d) Manhattan distance tensor, d-1 argmin/relax
    steps, in-tree masked to +inf), chunked so no chunk's (B,d,d) tensor exceeds a
    fixed element budget. This is the same algorithm as
    evaluator_ref.net_mst_edges (start vertex = local index 0, first-occurrence
    argmin tie-break -- torch.argmin ties break to the first index too, verified
    empirically; real-valued random coordinates make exact ties measure-zero anyway).
  - Crossing counts: Ph[iy,ix] / Pv[iy,ix] are prefix sums of adjacent-cell
    region-id differences along each row / column of the region grid, precomputed
    once. Every MST edge decomposes into a horizontal segment (at the first
    endpoint's row) then a vertical segment (at the second endpoint's column),
    exactly like evaluator_ref.edge_regions_and_crossings; crossing count per segment
    is then an O(1) prefix-sum difference, computed for *all* edges in one batched
    gather (no per-edge Python loop).
  - FT / pair demand: each segment's lattice-step span is bucketed by
    next-power-of-two step count (1,2,4,...,up to the grid size) and gathered as a
    padded (M,L) window of region ids (out-of-range positions are clamped to the
    segment's own last valid cell, which safely repeats an already-counted id and
    never fabricates a spurious transition); each bucket's M segments are further
    chunked so no chunk's padded (numSelected,L) tensor exceeds a fixed element
    budget, the same chunk-budget pattern _batch_mst uses above -- this bounds peak
    memory regardless of how many edges land in one bucket. The visited-region set
    becomes a K-bit bitmask via an O(log L) pairwise-OR halving reduction along the
    row, then an OR-scatter (bit-planed + scatter_reduce(amax)) into a per-net
    "passed" bitmask -- chunk-order-independent since amax-OR is commutative and
    associative. ft = popcount(passed & ~pin_bm). Adjacent-cell-id transitions
    within the padded window give the boundary-pair-demand (a,b) pairs, packed as
    a*64+b and reduced with a single np.unique(..., return_counts=True) call after
    collecting every bucket/segment-type's/chunk's keys (one CPU sync at the very
    end, not per bucket or chunk).
"""
import numpy as np
import torch

from ioplace.evaluator_ref import EvalResult
from ioplace.region_graph import region_graph as build_region_graph, next_hop_table, steiner_tree_stats
from ioplace.region_segments import segment_utilisation
from ioplace.straddle import _REGION_STRIDE


def _pow2_bounds(n):
    """[1, 2, 4, ...] up to the first power of two >= n (n >= 1)."""
    bounds = [1]
    while bounds[-1] < n:
        bounds.append(bounds[-1] * 2)
    return bounds


class GpuEvalContext:
    def __init__(self, nl, rg, device="cuda", max_degree=256,
                 mst_chunk_budget=8_000_000, seg_chunk_budget=8_000_000,
                 edge_batch_size=1_000_000, straddle=True,
                 segments=None, segment_capacity=None):
        """
        mst_chunk_budget / seg_chunk_budget / edge_batch_size (M4 T2 §4.2 item 7:
        "batch size as a construction parameter"): all three bound the peak size
        of an intermediate tensor that would otherwise scale with the *number of
        edges* (MST edges ~= total pins) rather than a fixed budget --
        mst_chunk_budget caps the (b,d,d) Prim distance tensor per degree-bucket
        chunk (unchanged from M3), seg_chunk_budget caps the (numSelected,L)
        padded-gather tensor in _process_segments (unchanged from M3, and only
        binding when it's *smaller* than edge_batch_size -- each call already
        receives at most edge_batch_size segments), and edge_batch_size (new in
        T2) caps how many MST edges are pulled through the per-edge geometry/
        crossing/FT/pair-demand pipeline in evaluate() at once -- see the
        batching loop there.

        edge_batch_size's default is 1,000,000, *not* 8,000,000 like the other
        two -- matching mst_chunk_budget/seg_chunk_budget's pre-existing 8M
        value turned out to not actually bound anything for real multi-million-
        net cases (mempool_group's ~8.1M MST edges, bigblue4's ~6.4M): 8M is
        close enough to those edge counts that batching barely engages,
        measured peak_alloc 4.42GB/4.45GB respectively -- short of both this
        file's T2 acceptance targets (bigblue4 >=60% reduction, mempool_group
        <=2GB). 1,000,000 clears both out of the box (measured 1.26GB/1.83GB)
        without a caller having to know to override it. All three remain
        overridable constructor parameters for the T2 batch-invariance tests
        (edge_batch_size in {1_000_000, 8_000_000, "all edges in one batch"})
        and for callers with a different memory/kernel-launch-overhead tradeoff.

        straddle (v2 P-F, design sec 7): compute the straddle diagnostics.
        Costs two persistent (num_physical,) float64 size tensors and, per
        evaluate(), an O(N) geometry pass plus one extra torch.unique over the
        pin keys -- small next to the MST, but not free at 30M cells, so it is
        switchable both here (skip the allocation) and per call
        (evaluate(..., straddle=False), which is intended for the in-loop
        diagnostic callback Task 5/6 wires up in run_placement_io -- that call
        site does not exist yet).

        num_movable/num_physical and the two size tensors are snapshotted here
        at construction time, like every other cached tensor on this class
        (grid_t, pin2node_t, ...); a caller that mutates `nl` between
        construction and evaluate() gets a silent ref/GPU divergence, same as
        it would for any other cached field.
        """
        self.nl, self.rg = nl, rg
        self.device = torch.device(device)
        self.max_degree = max_degree
        self.k = rg.k
        # the bitmask vectorization below (pin_bm/passed_bm packed via _pack_bits
        # into a single int64, one bit per region id) assumes every region id fits
        # in the 64 bits of an int64 -- fail loudly here rather than silently
        # truncating/wrapping region ids >= 64 later inside _pack_bits/_bit_planes.
        #
        # M3 T1 (R2 remedy) tightens this from <=64 to <=32: _pow2_k below is a
        # *signed* int64 (`1 << 63 == INT64_MIN`), so K=64 was already out of
        # contract even before this change silently relied on bit-pattern
        # equality rather than well-defined arithmetic; K<=32 also matches the
        # M3 experiment matrix (K in {8,16,32}, spec §1) and is required for
        # region_graph()'s Steiner-tree tables. See
        # docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md §4.2 and
        # docs/reviews/2026-08-13-m4-draft-v1-adversarial-codex.md finding 9.
        assert self.k <= 32, f"GpuEvalContext bitmask vectorization requires rg.k <= 32, got rg.k={self.k}"
        # v2 P-F / M1: the straddle diagnostics' (net, region) packing in
        # _distinct_regions_per_net uses _REGION_STRIDE (imported from
        # ioplace.straddle) as its stride -- mirror straddle_diagnostics'
        # own guard (straddle.py:131-134) rather than relying only on the
        # k<=32 assert above, which is sufficient today only because the
        # stride happens to be 64.
        assert self.k <= int(_REGION_STRIDE), (
            "straddle diagnostics pack (net, region) into one int64 with "
            "stride %d; got k=%d" % (int(_REGION_STRIDE), self.k))

        # ---- static region-grid tensors: grid + Ph/Pv crossing prefix sums ----
        # M4 T2 item 5: grid_t holds raw region ids (0..K-1, K<=32) and Ph/Pv hold
        # prefix-sum *counts* bounded by max(nx,ny) -- both are plain small values
        # (never bit-shifted into a packed mask, unlike the "vals"/"seg_bm"
        # bitmask tensors in _process_segments below, which must stay int64 --
        # see the comment there), so int32 is safe and halves this tensor's
        # footprint. assert the lattice fits comfortably before downcasting.
        ny, nx = rg.grid.shape
        assert max(nx, ny) < 2**31 - 1, f"region grid {nx}x{ny} exceeds int32 index range"
        grid_np = rg.grid.astype(np.int32)
        self.grid_t = torch.from_numpy(grid_np).to(self.device)
        self.nx, self.ny = nx, ny
        xl, yl, xh, yh = rg.die
        self.xl, self.yl = float(xl), float(yl)
        self.cell_w, self.cell_h = float(rg.cell_w), float(rg.cell_h)
        # 0-dim tensors for _to_idx's division, NOT a stylistic substitute for the
        # plain python floats above (self.cell_w/cell_h are kept as-is; other code
        # may still want them). CUDA compiles `tensor / python_float` as a
        # reciprocal-multiply (x * (1/c)) rather than a true divide -- (1/c) is
        # itself rounded, so the product can land one ULP below an exact integer
        # boundary (e.g. 2673.0 / (10692/512) -> 127.99999999999999 instead of
        # 128.0), which `.to(torch.int64)` then truncates down to 127: an off-by-one
        # lattice-cell mis-assignment exactly at region boundaries. `tensor / tensor`,
        # where the 0-dim tensor lives on the same (CUDA) device, instead dispatches
        # to a correctly-rounded elementwise divide, matching numpy's (and hence
        # evaluator_ref's) result bit-for-bit. This is device-dependent: a 0-dim CPU
        # tensor divided into a CUDA tensor gets lifted back to a Scalar and
        # mis-rounds identically -- hoisting these tensors to CPU would silently
        # reintroduce the bug.
        # Verified empirically on this host (torch 2.8.0+cu128, L4/sm_89): dividing
        # by the python float mis-rounds 80/511 lattice-boundary indices for
        # cell_w=10692/512 (including the region-grid boundaries at index 128 and
        # 256 for a 4x4 partition), while dividing by this tensor matches numpy
        # exactly at every one of those points. See C1 in the whole-branch review
        # and tests/test_evaluator_gpu.py::test_gpu_matches_reference_on_lattice_boundaries.
        self._cell_w_t = torch.tensor(self.cell_w, dtype=torch.float64, device=self.device)
        self._cell_h_t = torch.tensor(self.cell_h, dtype=torch.float64, device=self.device)
        assert self._cell_w_t.device == self.grid_t.device and self._cell_h_t.device == self.grid_t.device

        hdiff = (self.grid_t[:, :-1] != self.grid_t[:, 1:]).to(torch.int32)
        Ph = torch.zeros((ny, nx), dtype=torch.int32, device=self.device)
        Ph[:, 1:] = torch.cumsum(hdiff, dim=1)
        self.Ph = Ph

        vdiff = (self.grid_t[:-1, :] != self.grid_t[1:, :]).to(torch.int32)
        Pv = torch.zeros((ny, nx), dtype=torch.int32, device=self.device)
        Pv[1:, :] = torch.cumsum(vdiff, dim=0)
        self.Pv = Pv

        self._seg_bounds = _pow2_bounds(max(nx, ny))
        self._seg_bounds_t = torch.tensor(self._seg_bounds, dtype=torch.int64, device=self.device)

        # ---- bit-plane helpers, precomputed once ----
        self._bit_range = torch.arange(self.k, device=self.device, dtype=torch.int64)
        self._pow2_k = torch.bitwise_left_shift(torch.ones(self.k, dtype=torch.int64, device=self.device),
                                                 self._bit_range)

        # ---- M3 T1: region-adjacency graph G_R (K<=32 constant tables) ----
        # Cheap (K<=32) to build on CPU/numpy once per context; the CPU copies
        # (self._rg_D / self._rg_next_hop) are reused by the rare Λ>=4 tier
        # below (delegated to region_graph.steiner_tree_stats, the exact same
        # routine evaluator_ref.py uses per net), while self.D_t is the device
        # copy the bulk Λ<=3 closed-form gather uses.
        rg_adj, rg_D, rg_ell = build_region_graph(rg)
        self._rg_D = rg_D.astype(np.int64)
        self._rg_next_hop = next_hop_table(rg_adj, rg_D)
        self.D_t = torch.from_numpy(self._rg_D).to(self.device)

        # ---- static pin/net topology tensors ----
        # M4 T2 item 5: pin2node/pin2net (and every derived MST-edge/index
        # tensor below) are pure *indices* into node/net/pin arrays, not
        # bit-shifted into a packed mask -- so int32 is safe once the index
        # space itself is asserted to fit. Composite *keys* that combine two
        # index spaces into one integer (e.g. net_id*64+region_id below) are
        # the opposite case and must stay int64 (spec §4.2/F4).
        n_physical = nl.num_physical
        n_pins_total = len(nl.pin2node)
        assert n_physical < 2**31 - 1, f"num_physical={n_physical} exceeds int32 index range"
        assert n_pins_total < 2**31 - 1, f"total pins={n_pins_total} exceeds int32 index range"
        self.pin2node_t = torch.from_numpy(nl.pin2node.astype(np.int32)).to(self.device)
        self.pin2net_t = torch.from_numpy(nl.pin2net.astype(np.int32)).to(self.device)
        self.pin_offset_x_t = torch.from_numpy(nl.pin_offset_x.astype(np.float64)).to(self.device)
        self.pin_offset_y_t = torch.from_numpy(nl.pin_offset_y.astype(np.float64)).to(self.device)

        # v2 P-F: movable/physical counts and the sizes the sec 7 diagnostics
        # need. float64 for the same reason every other geometry tensor here is
        # float64 -- these feed _to_idx, whose truncation decides integers.
        self.num_movable = int(nl.num_movable)
        self.num_physical = int(n_physical)
        self.straddle = bool(straddle)
        if self.straddle:
            self.node_size_x_t = torch.from_numpy(
                np.asarray(nl.node_size_x[:n_physical], dtype=np.float64)).to(self.device)
            self.node_size_y_t = torch.from_numpy(
                np.asarray(nl.node_size_y[:n_physical], dtype=np.float64)).to(self.device)
        else:
            self.node_size_x_t = self.node_size_y_t = None

        self.n_nets = nl.num_nets
        # M4 T2 (spec §4.2 F4 / §2.1): composite (net_id, region_id) keys used
        # below (evaluate()'s net_region_key) and elsewhere (pair-demand's
        # lo_ab*64+hi_ab) pack a "*64" index space alongside n_nets into a
        # single int64 -- assert the headroom once here rather than silently
        # risking overflow as n_nets grows with glue nets at 30M scale.
        assert self.n_nets * 64 < 2**63, f"n_nets={self.n_nets} leaves no int64 headroom for *64 composite keys"
        assert self.n_nets < 2**31 - 1, f"n_nets={self.n_nets} exceeds int32 index range"
        degrees = nl.net_degrees
        self.degrees_t = torch.from_numpy(degrees.astype(np.int64)).to(self.device)

        start = nl.flat_net2pin_start
        flat = nl.flat_net2pin
        self._buckets = []  # [(d, net_ids (B,) int32, pin_idx_mat (B,d) int32)] all on device
        for d in range(2, max_degree + 1):
            net_ids = np.nonzero(degrees == d)[0]
            if len(net_ids) == 0:
                continue
            s = start[net_ids].astype(np.int64)
            offsets = np.arange(d, dtype=np.int64)
            idx_mat = s[:, None] + offsets[None, :]
            pin_idx_mat = flat[idx_mat].astype(np.int32)
            self._buckets.append((
                d,
                torch.from_numpy(net_ids.astype(np.int32)).to(self.device),
                torch.from_numpy(pin_idx_mat).to(self.device),
            ))

        large_net_ids = np.nonzero(degrees > max_degree)[0]
        self.large_net_ids_t = torch.from_numpy(large_net_ids.astype(np.int32)).to(self.device)

        # memory budget: cap a chunk's (b,d,d) distance tensor at this many elements
        self._mst_chunk_budget = mst_chunk_budget

        # memory budget: cap a chunk's (numSelected, L) padded-gather tensor (used
        # by _process_segments) at this many elements. Per chunk, up to ~3 tensors
        # of this shape are concurrently alive during the bitmask-reduction
        # pass (cand/ids/vals), plus -- in the boundary-pair-demand pass that
        # follows, while cand/ids are still resident -- up to ~4 more tensors
        # sized by the (bool) diff mask's nonzero count (a/b/lo_ab/hi_ab), bounded by
        # the same numSelected*L in the pathological worst case where every adjacent
        # cell differs (real region grids -- contiguous rectangular partitions --
        # see far fewer boundary transitions per segment in practice, so this is a
        # deliberately conservative bound). Independently configurable from
        # mst_chunk_budget as of M4 T2 item 7 (both still default to the same
        # 8,000,000 value as before T2, so behaviour is unchanged from pre-T2
        # unless a caller overrides one; edge_batch_size below is the one that
        # changed default -- see its __init__ docstring entry).
        self._seg_chunk_budget = seg_chunk_budget

        # M4 T2 item 4/7: caps how many MST edges evaluate()'s per-edge
        # geometry/crossing/FT/pair-demand loop processes at once -- see the
        # batching loop there. A construction parameter (not derived from the
        # other two budgets) so it can be dialled independently for the T2
        # batch-invariance tests.
        self._edge_batch_size = edge_batch_size

        # ---- v2 P-D (design sec 5): the unit-edge -> segment-id raster ----
        # The GPU path reads the raster directly rather than the per-row CSR:
        # _process_segments already gathers every lattice cell along a leg for
        # the bitmask and pair-demand passes, so the segment id of a transition
        # is one more indexed read at coordinates it already holds. The CSR is
        # the reference's structure, where the cost model is the opposite.
        #
        # Task-7 brief drift: the brief places this block right after
        # `self._seg_bounds_t = ...` (pre-P-D __init__), but self.n_nets
        # (needed by the composite-key headroom assert below) is not assigned
        # until further up this same __init__ -- moved here, after every
        # field it reads is already set, rather than reproduced at a location
        # that would AttributeError.
        self.segments = segments
        self.segment_capacity = None
        self.edge_seg_v_t = None
        self.edge_seg_h_t = None
        if segments is not None:
            if (segments.k != rg.k or segments.grid_shape() != rg.grid.shape
                    or segments.die != tuple(float(v) for v in rg.die)):
                raise ValueError("segment table was built for a different region grid")
            self.num_segments = int(segments.num_segments)
            assert self.n_nets * self.k * self.k * max(self.num_segments, 1) < 2 ** 63, (
                "no int64 headroom for the (net,u,v,seg) composite candidate key")
            self.edge_seg_v_t = torch.from_numpy(
                np.ascontiguousarray(segments.edge_seg_v)).to(self.device)
            self.edge_seg_h_t = torch.from_numpy(
                np.ascontiguousarray(segments.edge_seg_h)).to(self.device)
            if segment_capacity is not None:
                self.segment_capacity = np.asarray(segment_capacity,
                                                   dtype=np.float64)
                if self.segment_capacity.shape != (self.num_segments,):
                    raise ValueError("segment_capacity must carry one value "
                                     "per segment")
        elif segment_capacity is not None:
            # Task-7 fix round 1: same message as evaluator_ref.evaluate's
            # equivalent guard (evaluator_ref.py), so a caller who forgets
            # segments= gets an identical error on both sides rather than an
            # error here and a silent no-op on the capacity_candidates path
            # (guarded separately in evaluate() below, since that flag is a
            # per-call argument here, not a constructor argument).
            raise ValueError("segment_capacity/capacity_candidates need segments=")

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _to_idx(self, x, y):
        # divide by the cached 0-dim tensors, not self.cell_w/cell_h (python floats)
        # -- see the comment where they're constructed in __init__ for why this
        # matters (C1: CUDA reciprocal-multiply off-by-one at lattice boundaries).
        # M4 T2 item 5: truncate to int32, not int64 -- lattice indices are
        # bounded by self.nx/self.ny (asserted < 2**31-1 in __init__) and this
        # is a plain grid-coordinate value, never bit-shifted into a packed
        # mask, so int32 is safe and halves every downstream (M,)-shaped index
        # tensor derived from pin/edge positions (ax/ay/bx/by, h_row/h_lo/h_hi,
        # v_col/v_lo/v_hi in evaluate()). The float64 division itself (the part
        # that must match numpy bit-for-bit) is unaffected -- only the final
        # truncation's *storage* width changes, not the truncated value.
        ix = ((x - self.xl) / self._cell_w_t).to(torch.int32).clamp_(0, self.nx - 1)
        iy = ((y - self.yl) / self._cell_h_t).to(torch.int32).clamp_(0, self.ny - 1)
        return ix, iy

    def _pin_positions(self, node_x, node_y):
        px = node_x[self.pin2node_t] + self.pin_offset_x_t
        py = node_y[self.pin2node_t] + self.pin_offset_y_t
        return px, py

    def _distinct_regions_per_net(self, pin_rid):
        """(n_nets,) int64 distinct-region count per net, 0 for degree<2 -- the
        torch mirror of straddle.distinct_regions_per_net.

        R-8: packs (net, region) using _REGION_STRIDE, the same named constant
        ioplace.straddle uses for the identical packing -- not the bare `64`
        evaluate()'s net_region_key/pin_bm packing above hardcodes for its own,
        unrelated (net, region) key (same numeric value today, but importing
        the name instead of re-hardcoding it means the two can never silently
        drift apart)."""
        stride = int(_REGION_STRIDE)
        key = torch.unique(self.pin2net_t.to(torch.int64) * stride
                           + pin_rid.to(torch.int64))
        counts = torch.bincount(key // stride, minlength=self.n_nets)
        return torch.where(self.degrees_t >= 2, counts, torch.zeros_like(counts))

    def _straddle_stats(self, node_x, node_y, pin_rid, per_net_lambda):
        """Torch mirror of ioplace.straddle.straddle_diagnostics. Every
        convention is documented there; this must not invent one.

        The lattice-line coordinate below divides/multiplies through the 0-dim
        self._cell_w_t / self._cell_h_t tensors, never the python floats
        self.cell_w / self.cell_h -- same C1 reciprocal-multiply hazard as
        _to_idx (see its comment); a python float here mis-rounds the quadrant
        split exactly at region boundaries.
        """
        dev = self.device
        m = self.num_movable
        x = node_x[:m]
        y = node_y[:m]
        w = self.node_size_x_t[:m]
        h = self.node_size_y_t[:m]
        xr = x + w
        yt = y + h

        def _rid(ax, ay):
            ix, iy = self._to_idx(ax, ay)
            return self.grid_t[iy, ix].to(torch.int64)

        r00 = _rid(x, y)
        r10 = _rid(xr, y)
        r01 = _rid(x, yt)
        r11 = _rid(xr, yt)
        owner = _rid(x + 0.5 * w, y + 0.5 * h)
        straddle = (r10 != r00) | (r01 != r00) | (r11 != r00)

        ix0, iy0 = self._to_idx(x, y)
        ix1, iy1 = self._to_idx(xr, yt)
        xm = torch.minimum(xr, self.xl + (ix0.to(torch.float64) + 1.0) * self._cell_w_t)
        ym = torch.minimum(yt, self.yl + (iy0.to(torch.float64) + 1.0) * self._cell_h_t)
        lw = (xm - x).clamp(min=0.0)
        rw = (xr - xm).clamp(min=0.0)
        bh = (ym - y).clamp(min=0.0)
        th = (yt - ym).clamp(min=0.0)
        out_area = (lw * bh * (r00 != owner).to(torch.float64)
                    + rw * bh * (r10 != owner).to(torch.float64)
                    + lw * th * (r01 != owner).to(torch.float64)
                    + rw * th * (r11 != owner).to(torch.float64))
        wide = ((ix1 - ix0) > 1) | ((iy1 - iy0) > 1)

        per_node = torch.zeros(self.num_physical, dtype=torch.uint8, device=dev)
        per_node[:m] = straddle.to(torch.uint8)
        owner_full = torch.zeros(self.num_physical, dtype=torch.int64, device=dev)
        owner_full[:m] = owner
        node_of_pin = self.pin2node_t.to(torch.int64)
        pin_rid_re = torch.where(per_node[node_of_pin].bool(),
                                 owner_full[node_of_pin], pin_rid.to(torch.int64))
        split = (per_net_lambda - self._distinct_regions_per_net(pin_rid_re))
        # v2 P-F fix round 1 (I2): follow this file's own explicit-del memory
        # discipline (see the "CPython function frames aren't block-scoped"
        # comment in evaluate()) -- these (M,)/(n_pins,) temporaries are done
        # contributing once split/out_area are in hand, and at 30M cells/~100M
        # pins leaving them referenced for the rest of this frame is several
        # GB of avoidable peak.
        del node_of_pin, pin_rid_re, owner_full

        total_area = float((w * h).sum().item())
        out_total = float(out_area.sum().item())
        del r00, r10, r01, r11, owner, xm, ym, lw, rw, bh, th, out_area
        return {
            "straddle_cells": int(straddle.sum().item()),
            "straddle_area_fraction": (out_total / total_area) if total_area > 0.0 else 0.0,
            "straddle_pin_split_nets": int((split > 0).sum().item()),
            "straddle_out_area": out_total,
            "straddle_movable_area": total_area,
            "straddle_wide_cells": int(wide.sum().item()),
            "per_node_straddle": per_node.cpu().numpy(),
            "per_net_pin_split": split.to(torch.int32).cpu().numpy(),
        }

    def _bit_planes(self, bm, dtype=None):
        """(...,) int64 *already-combined* bitmask -> (...,K) 0/1 bit planes.

        NOT for one-hot-encoding a single id -- decomposing a raw id value's own
        binary digits is a different (wrong) operation; use _one_hot_planes for that.

        `dtype` (default int64, unchanged behaviour for existing callers):
        pass torch.int8 to produce the output directly at int8 -- required by
        R2 (T1 §2.5 item 3): pin_bit_acc/passed_bit_acc go from (E,K) int64 to
        int8, and scatter_reduce_ requires self.dtype == src.dtype, so the
        *source* planes must be produced as int8 directly (the bitwise
        extraction itself still happens on `bm` at its own width -- only the
        final 0/1 result is narrowed, so this is not the "int64 then .to(int8)"
        pattern that would transiently double memory).
        """
        planes = torch.bitwise_and(torch.bitwise_right_shift(bm.unsqueeze(-1), self._bit_range), 1)
        return planes if dtype is None else planes.to(dtype)

    def _one_hot_planes(self, ids, dtype=None):
        """(...,) int64 id values in [0,K) -> (...,K) 0/1 one-hot planes (plane
        i is 1 iff ids==i). This is the "id -> single-bit-set bitmask" direction;
        _bit_planes is the reverse (combined bitmask -> planes) and is not
        interchangeable with this.

        `dtype` (default int64, unchanged behaviour for existing callers): see
        _bit_planes above -- the `==` comparison already produces a bool (1
        byte/element) tensor, so `.to(torch.int8)` is a cheap narrow, not a
        widen-then-narrow round trip.
        """
        return (ids.unsqueeze(-1) == self._bit_range).to(dtype or torch.int64)

    def _popcount_k(self, bm):
        """(...,) int64 packed bitmask -> (...,) int64 popcount.

        M4 T2 item 3: this used to be `self._bit_planes(bm).sum(dim=-1)` --
        but _bit_planes' `bitwise_and(bitwise_right_shift(...), 1)` promotes
        to int64 *before* any dtype narrowing (type promotion follows the
        widest input, and self._bit_range is int64), so passing
        dtype=torch.int8 to _bit_planes only narrows the *final* stored
        tensor, not the (...,K) int64 intermediate the bitwise ops themselves
        allocate at peak -- for pin_bm/ft_bm (called here on the full
        (n_nets,) array) that peak intermediate is exactly the (n_nets,K)
        int64 tensor T2 set out to eliminate, just relocated from
        `_one_hot_planes` into `_popcount_k`. This SWAR (SIMD-within-a-
        register) bit-counting recurrence instead computes the popcount with
        O(log K) elementwise (...,)-shaped int64 ops -- never a (...,K)
        tensor -- exploiting that every packed bitmask in this file (pin_bm/
        passed_bm/ft_bm) only ever has bits [0,K) set, K<=32 (asserted in
        __init__), well inside int64's 64-bit word; verified bit-exact
        against the old _bit_planes().sum(dim=-1) implementation, including
        at the K=32 bit-31 boundary, before landing this.
        """
        v = bm
        v = v - ((v >> 1) & 0x5555555555555555)
        v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
        v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0f
        v = v + (v >> 8)
        v = v + (v >> 16)
        v = v + (v >> 32)
        return v & 0x7f

    def _pack_bits(self, bit_planes):
        """(...,K) 0/1 int64 bit planes -> (...,) int64 bitmask."""
        return (bit_planes * self._pow2_k).sum(dim=-1)

    # ------------------------------------------------------------------
    # MST: batched dense Prim per exact-degree bucket
    # ------------------------------------------------------------------
    def _prim_batch(self, X, Y):
        """Manhattan-distance batched Prim, matching evaluator_ref.net_mst_edges:
        start vertex = local index 0, first-occurrence argmin tie-break.
        X, Y: (B,d) float64 -> a, b: (B,d-1) int64 local-index edge endpoints.
        """
        B, d = X.shape
        dev = X.device
        dist = (X.unsqueeze(2) - X.unsqueeze(1)).abs() + (Y.unsqueeze(2) - Y.unsqueeze(1)).abs()
        in_tree = torch.zeros((B, d), dtype=torch.bool, device=dev)
        in_tree[:, 0] = True
        best_cost = dist[:, 0, :].clone()
        best_from = torch.zeros((B, d), dtype=torch.int64, device=dev)
        a_edges = torch.zeros((B, d - 1), dtype=torch.int64, device=dev)
        b_edges = torch.zeros((B, d - 1), dtype=torch.int64, device=dev)
        inf = torch.finfo(torch.float64).max
        for t in range(d - 1):
            masked = torch.where(in_tree, inf, best_cost)
            j = torch.argmin(masked, dim=1)
            a_edges[:, t] = best_from.gather(1, j.unsqueeze(1)).squeeze(1)
            b_edges[:, t] = j
            in_tree.scatter_(1, j.unsqueeze(1), True)
            new_dist = dist.gather(1, j.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
            upd = new_dist < best_cost
            best_cost = torch.where(upd, new_dist, best_cost)
            best_from = torch.where(upd, j.unsqueeze(1).expand(B, d), best_from)
        return a_edges, b_edges

    def _batch_mst(self, px, py):
        """Returns (edge_net_id, edge_pin_a, edge_pin_b), all (M,) int32 -- pure
        indices into the net/pin arrays (bounded by n_nets / total pins, both
        asserted < 2**31-1 in __init__), never bit-shifted into a packed mask,
        so int32 halves this M-sized allocation vs. the pre-T2 int64 version
        (M4 T2 item 5). pa/pb inherit int32 from pin_idx_mat (the gather
        *source*); a_local/b_local (the gather *index*, from _prim_batch) stay
        int64 internally -- gather's index dtype is independent of its source's,
        and a_local/b_local are only ever (b,d)-shaped with d<=max_degree, not
        M-shaped, so there's no memory upside to touching them.
        """
        dev = self.device
        net_ids_list, pin_a_list, pin_b_list = [], [], []
        for d, net_ids, pin_idx_mat in self._buckets:
            B = net_ids.shape[0]
            chunk = max(1, self._mst_chunk_budget // (d * d))
            for start in range(0, B, chunk):
                end = min(B, start + chunk)
                sub_net_ids = net_ids[start:end]
                sub_pin_idx = pin_idx_mat[start:end]  # (b,d) pin-array indices
                a_local, b_local = self._prim_batch(px[sub_pin_idx], py[sub_pin_idx])
                pa = torch.gather(sub_pin_idx, 1, a_local)
                pb = torch.gather(sub_pin_idx, 1, b_local)
                net_rep = sub_net_ids.unsqueeze(1).expand(-1, d - 1)
                net_ids_list.append(net_rep.reshape(-1))
                pin_a_list.append(pa.reshape(-1))
                pin_b_list.append(pb.reshape(-1))
        if not net_ids_list:
            empty = torch.empty(0, dtype=torch.int32, device=dev)
            return empty, empty, empty
        return torch.cat(net_ids_list), torch.cat(pin_a_list), torch.cat(pin_b_list)

    # ------------------------------------------------------------------
    # FT (passed-region bitmask) + boundary pair demand, via bucketed padded gather
    # ------------------------------------------------------------------
    def _process_segments(self, fixed_idx, lo, hi, net_id, is_vert,
                          passed_bit_acc, pair_count_acc,
                          segment_demand_acc=None, edge_u=None, edge_v=None,
                          candidate_keys=None):
        dev = self.device
        cell_count = hi - lo + 1
        bucket_idx = torch.bucketize(cell_count, self._seg_bounds_t, right=False)
        for i, L in enumerate(self._seg_bounds):
            sel = bucket_idx == i
            f_all = fixed_idx[sel]
            B = f_all.numel()
            if B == 0:
                continue
            l0_all = lo[sel]
            h0_all = hi[sel]
            nid_all = net_id[sel]
            # M4 T2 item 5: int32 -- lo/hi are already int32 grid coordinates
            # (see _to_idx), so l0+offsets stays int32 (no auto-promotion to
            # int64) instead of the pre-T2 default int64 arange.
            offsets = torch.arange(L, device=dev, dtype=torch.int32)

            # chunk this bucket's B selected segments so no single (chunk, L)
            # padded-gather tensor exceeds _seg_chunk_budget elements -- same
            # chunk-budget pattern as _batch_mst above. Each chunk's contribution is
            # aggregated via in-place scatter_reduce(amax) into passed_bit_acc and by
            # a per-chunk torch.bincount accumulated into pair_count_acc (M3-scope
            # memory fix #2: this replaces appending each chunk's raw keys to a
            # growing Python list that used to be torch.cat(...).cpu().numpy()'d in
            # one shot at the very end -- that one final sync/concat held every
            # chunk's keys live on the GPU simultaneously; accumulating a fixed
            # 64*64-sized count tensor per chunk instead needs no such buildup and
            # only syncs once, on that small fixed-size tensor, in
            # _reduce_pair_demand). Both amax-OR and count-accumulation are
            # order-independent across chunks, so this is a pure memory-shape
            # change with no effect on the result.
            chunk = max(1, self._seg_chunk_budget // L)
            for start in range(0, B, chunk):
                end = min(B, start + chunk)
                f = f_all[start:end]
                l0 = l0_all[start:end]
                h0 = h0_all[start:end]
                nid = nid_all[start:end]
                cand = torch.minimum(l0.unsqueeze(1) + offsets.unsqueeze(0), h0.unsqueeze(1))
                if is_vert:
                    ids = self.grid_t[cand, f.unsqueeze(1).expand(-1, L)]
                else:
                    ids = self.grid_t[f.unsqueeze(1).expand(-1, L), cand]

                # visited-region bitmask per segment: O(log L) pairwise-OR halving reduce.
                # M4 T2 dtype hazard: `ids` (region ids, 0..K-1) is int32 (from
                # grid_t, item 5), but the *shifted* value 1<<ids is a packed
                # bitmask that can set bit 31 when K=32 -- in *signed* int32 that
                # is INT32_MIN, and a later arithmetic (sign-extending) right
                # shift on it would corrupt bit extraction in _bit_planes. The
                # base of the shift must stay int64 (63 usable positive bits);
                # only the shift-*amount* (ids) may be the narrower int32 --
                # verified empirically (torch 2.8.0+cu128/L4) that
                # bitwise_left_shift accepts a mismatched int64-base/int32-amount
                # pair and a bare torch.ones_like(ids) here would silently regress
                # to the unsafe int32 base.
                vals = torch.bitwise_left_shift(torch.ones_like(ids, dtype=torch.int64), ids)
                length = L
                while length > 1:
                    half = length // 2
                    vals = vals[:, :half] | vals[:, half:half * 2]
                    length = half
                seg_bm = vals[:, 0]
                # R2 remedy (T1 §2.5 item 3): source bit planes + accumulator
                # go int8 together (scatter_reduce_ requires self.dtype ==
                # src.dtype; 0/1 amax === bitwise OR, so the reduce semantics
                # are unchanged -- see docs/superpowers/specs/2026-08-13-m4-
                # scale-up-design-draft.md §4.2).
                seg_bits = self._bit_planes(seg_bm, dtype=torch.int8)
                nid_exp = nid.unsqueeze(1).expand(-1, self.k)
                passed_bit_acc.scatter_reduce_(0, nid_exp, seg_bits, reduce="amax", include_self=True)

                # boundary-pair demand: adjacent-cell id transitions within the (unpadded) span
                if L > 1:
                    diff = ids[:, 1:] != ids[:, :-1]
                    if diff.any():
                        a = ids[:, :-1][diff]
                        b = ids[:, 1:][diff]
                        lo_ab = torch.minimum(a, b)
                        hi_ab = torch.maximum(a, b)
                        keys = lo_ab * 64 + hi_ab
                        pair_count_acc += torch.bincount(keys, minlength=pair_count_acc.numel())

                        # v2 P-D: the same transitions, keyed by segment id.
                        # Within the unpadded span cand[:,t+1] == cand[:,t]+1,
                        # so the transition at position t is the unit edge whose
                        # boundary index is cand[:,t]. Padded positions repeat
                        # the last cell, so `diff` is False there -- the same
                        # property the pair-demand pass above relies on.
                        if segment_demand_acc is not None:
                            # Task-7 fix: `cand[:, :-1]` at a *padded* position
                            # (beyond the unpadded cell_count span) repeats the
                            # segment's last valid cell index, h0 -- which can
                            # be nx-1/ny-1, one past edge_seg_v_t/edge_seg_h_t's
                            # boundary-index range (nx-1/ny-1 wide). diff is
                            # False there (the whole point of the padding
                            # scheme), but advanced indexing evaluates the
                            # gather for *every* position before `[diff]`
                            # filters it, so an unclamped out-of-range boundary
                            # index CUDA-asserts even though its result is
                            # discarded. Clamping to each raster's own last
                            # valid boundary index is safe: a genuine transition
                            # (diff True) can only occur at an *internal*
                            # boundary, whose index is always < that bound.
                            base = cand[:, :-1]
                            fixed_exp = f.unsqueeze(1).expand(-1, L - 1)
                            if is_vert:
                                base_h = torch.clamp(base, max=self.edge_seg_h_t.shape[0] - 1)
                                seg_ids = self.edge_seg_h_t[base_h, fixed_exp][diff]
                            else:
                                base_v = torch.clamp(base, max=self.edge_seg_v_t.shape[1] - 1)
                                seg_ids = self.edge_seg_v_t[fixed_exp, base_v][diff]
                            seg_ids = seg_ids.to(torch.int64)
                            assert bool((seg_ids >= 0).all()), \
                                "a lattice transition mapped to no segment"
                            segment_demand_acc += torch.bincount(
                                seg_ids, minlength=segment_demand_acc.numel())
                            if candidate_keys is not None:
                                nid_exp = nid.unsqueeze(1).expand(-1, L - 1)[diff]
                                u_exp = edge_u[sel][start:end].unsqueeze(1).expand(-1, L - 1)[diff]
                                v_exp = edge_v[sel][start:end].unsqueeze(1).expand(-1, L - 1)[diff]
                                keep = u_exp != v_exp
                                candidate_keys["dropped"] += int((~keep).sum())
                                if bool(keep.any()):
                                    net_k = nid_exp[keep].to(torch.int64)
                                    lo_uv = torch.minimum(u_exp[keep], v_exp[keep]).to(torch.int64)
                                    hi_uv = torch.maximum(u_exp[keep], v_exp[keep]).to(torch.int64)
                                    key = (((net_k * self.k + lo_uv) * self.k + hi_uv)
                                           * self.num_segments + seg_ids[keep])
                                    uniq, counts = torch.unique(key, return_counts=True)
                                    candidate_keys["keys"].append(uniq)
                                    candidate_keys["counts"].append(counts)

    @staticmethod
    def _reduce_pair_demand(pair_count_acc):
        counts = pair_count_acc.cpu().numpy()
        nz = np.nonzero(counts)[0]
        return {(int(u) // 64, int(u) % 64): int(counts[u]) for u in nz}

    def _reduce_candidates(self, candidate_keys):
        """Collapse the per-batch unique (net,u,v,seg) keys into one ascending
        list. torch.unique sorts, and the reference builds the same composite
        key with np.unique -- which is what makes the two candidate lists
        bit-identical rather than merely equivalent as sets."""
        empty = np.zeros(0, dtype=np.int64)
        if not candidate_keys["keys"]:
            return dict(cand_net=empty, cand_u=empty, cand_v=empty,
                        cand_seg=empty, cand_count=empty,
                        cand_dropped=int(candidate_keys["dropped"]))
        keys = torch.cat(candidate_keys["keys"])
        counts = torch.cat(candidate_keys["counts"])
        uniq, inverse = torch.unique(keys, return_inverse=True)
        totals = torch.zeros(uniq.numel(), dtype=torch.int64, device=uniq.device)
        totals.index_add_(0, inverse, counts.to(torch.int64))
        uniq = uniq.cpu().numpy().astype(np.int64)
        size = self.num_segments
        seg = uniq % size
        rest = uniq // size
        v = rest % self.k
        rest = rest // self.k
        u = rest % self.k
        net = rest // self.k
        return dict(cand_net=net, cand_u=u, cand_v=v, cand_seg=seg,
                    cand_count=totals.cpu().numpy().astype(np.int64),
                    cand_dropped=int(candidate_keys["dropped"]))

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def evaluate(self, node_x, node_y, *, straddle=None, capacity_candidates=False):
        want_straddle = self.straddle if straddle is None else bool(straddle)
        if want_straddle and not self.straddle:
            raise ValueError("this GpuEvalContext was built with straddle=False, "
                             "so it holds no node-size tensors; rebuild it with "
                             "straddle=True to ask for the sec 7 diagnostics")
        # Task-7 fix round 1: evaluator_ref.evaluate rejects
        # capacity_candidates without segments= at call time (it takes both
        # in one call); here segments= is bound at construction and
        # capacity_candidates is a per-call argument, so the equivalent guard
        # has to live here rather than in __init__ (which already rejects
        # segment_capacity without segments= with the same message). Without
        # this, a caller who forgets segments= got a ValueError on the CPU
        # path and a silent cand_net=None no-op on the GPU path.
        if capacity_candidates and self.segments is None:
            raise ValueError("segment_capacity/capacity_candidates need segments=")
        dev = self.device
        node_x = torch.as_tensor(node_x, dtype=torch.float64, device=dev)
        node_y = torch.as_tensor(node_y, dtype=torch.float64, device=dev)

        px, py = self._pin_positions(node_x, node_y)
        pin_ix, pin_iy = self._to_idx(px, py)
        pin_rid = self.grid_t[pin_iy, pin_ix]

        n_nets = self.n_nets

        # ---- pin_bm (per-net OR of pin region bits) + home_e, M4 T2 item 3 ----
        # M3 T1 got pin_bm's *accumulator* down to int8 but still materialized
        # a (P,K) *source* tensor (`pin_bits`) to feed it -- 11.25GB->1.41GB
        # @12.71M nets/K=32, still the single largest tensor in this file (spec
        # §4.1/§4.2). T2 eliminates the (P,K) tensor entirely (and, as a bonus,
        # the dense (n_nets,K) `home_counts` this file used to build separately
        # for the home-region argmax) via the "unique(net*64+rid) then exact OR"
        # option spec §4.2 item 4 calls out: net_region_key packs (net_id,
        # region_id) into one int64 per pin (a *composite key*, so it stays
        # int64 per §4.2/F4 regardless of pin2net_t/pin_rid's own int32 dtype);
        # torch.unique dedupes it to one row per *distinct* (net,region) pair
        # actually touched -- U rows, U <= min(P, n_nets*K) and in practice far
        # below n_nets*K for real netlists -- without ever holding a P- or
        # (n_nets,K)-sized 2-D tensor.
        #
        # pin_bm: since the keys are already deduped, index_add_'ing 1<<region
        # (one term per unique (net,region) row) into a zeroed (n_nets,) int64
        # accumulator sums each net's *distinct* touched-region bits exactly
        # once -- for disjoint bit positions, sum == OR (integer addition is
        # exact and order-independent, so this is bit-exact by construction,
        # not just "very likely correct"). Verified against a dense-tensor
        # reference for both pin_bm and home below before landing this.
        #
        # home_e: the region holding the most pins, ties -> smallest region id
        # (torch.argmax breaks ties to the first/lowest index, matching
        # np.argmax -- the same convention the pre-T2 `home_counts.argmax(dim=1)`
        # relied on). `score = count*64 - region` is injective across every
        # (count, region) combination that can occur (region in [0,64), so
        # consecutive counts' score ranges [c*64-63, c*64] never overlap) --
        # scatter_reduce_(amax) finds each net's best score in one (n_nets,)
        # pass, and because the encoding is injective, exactly one unique row
        # per net-with->=1-touched-region achieves it, so the direct assignment
        # below (not a scatter) is race-free.
        net_region_key = self.pin2net_t.to(torch.int64) * 64 + pin_rid.to(torch.int64)
        del pin_ix, pin_iy
        # v2 P-D: the capacity candidate keys (net,u,v,seg) need each MST
        # edge endpoint's region id -- gathered from pin_rid in the edge-batch
        # loop far below (D-1/D-2: keyed on the MST-edge terminal pair). That
        # is the only other place in evaluate() pin_rid is read, so it must
        # survive at least that long whenever capacity_candidates is asked
        # for; task-7 brief drift: the brief's unconditional
        # `del pin_ix, pin_iy, pin_rid` predates P-F's want_straddle-gated
        # deletion here -- merged rather than replacing it.
        need_pin_rid_for_capacity = self.segments is not None and capacity_candidates
        if not want_straddle and not need_pin_rid_for_capacity:
            del pin_rid          # only net_region_key above needed it
        uniq_keys, uniq_counts = torch.unique(net_region_key, return_counts=True)
        del net_region_key
        uniq_net = uniq_keys // 64
        uniq_region = uniq_keys % 64

        pin_bm = torch.zeros(n_nets, dtype=torch.int64, device=dev)
        pin_bm.index_add_(0, uniq_net, torch.bitwise_left_shift(torch.ones_like(uniq_region), uniq_region))

        home_score = uniq_counts.to(torch.int64) * 64 - uniq_region
        best_home_score = torch.full((n_nets,), -1, dtype=torch.int64, device=dev)
        best_home_score.scatter_reduce_(0, uniq_net, home_score, reduce="amax", include_self=True)
        is_home_best = home_score == best_home_score[uniq_net]
        per_net_home = torch.zeros(n_nets, dtype=torch.int64, device=dev)
        per_net_home[uniq_net[is_home_best]] = uniq_region[is_home_best]
        per_net_home = torch.where(self.degrees_t >= 2, per_net_home,
                                    torch.zeros(n_nets, dtype=torch.int64, device=dev))
        # M4 T2: CPython function frames aren't block-scoped -- these locals
        # would otherwise stay referenced (and their GPU memory held) for the
        # rest of evaluate() even though nothing after this point uses them;
        # explicit del lets the caching allocator reclaim them before the
        # heavier phases below (per_net_lambda's popcount, the edge-batching
        # loop) run. Verified via a peak-memory diagnostic that this is not
        # cosmetic -- omitting it measurably raised mempool_group's peak.
        del uniq_keys, uniq_counts, uniq_net, uniq_region, home_score, best_home_score, is_home_best

        per_net_lambda = torch.where(self.degrees_t >= 2, self._popcount_k(pin_bm),
                                     torch.zeros_like(pin_bm))
        hard_lambda_sum = int((per_net_lambda - 1).clamp(min=0).sum().item())

        # v2 P-F (design sec 7): per_net_lambda is in hand and pin_rid is still
        # alive, which is the only point in evaluate() where both are true.
        # v2 P-F fix round 1 (I2): explicit del, not reassignment to None --
        # matching this file's own discipline (see the "CPython function
        # frames aren't block-scoped" comment above): pin_rid is a (n_pins,)
        # int64 tensor and nothing after this point needs it.
        if want_straddle:
            straddle_stats = self._straddle_stats(node_x, node_y, pin_rid,
                                                  per_net_lambda)
            if not need_pin_rid_for_capacity:
                del pin_rid
        else:
            straddle_stats = {}
            # pin_rid, if still alive here, is kept alive for the capacity
            # candidate-key gather in the edge-batch loop below and is
            # explicitly deleted there once it is no longer needed.

        # per_net_steiner (ST_e): Λ<=3 closed form (bulk, vectorized over only
        # the Λ==2 / Λ==3 subsets -- never a full (n_nets,K) bit-plane tensor);
        # Λ>=4 (F11: always expanded into an actual G_R subtree) delegated to
        # region_graph.steiner_tree_stats, exactly as evaluator_ref uses per
        # net, guaranteeing CPU/GPU agreement by construction. This tier is
        # rare in practice (<1% of active nets on the M3 evidence, §2.2) so a
        # Python-side loop over just those nets stays within the T1 runtime
        # budget; see the T1 report for the measured bigblue4 increment.
        per_net_steiner = torch.zeros(n_nets, dtype=torch.int32, device=dev)

        lam2 = (per_net_lambda == 2).nonzero(as_tuple=True)[0]
        if lam2.numel() > 0:
            bits2 = self._bit_planes(pin_bm[lam2])          # (B2,K) int64 0/1, exactly 2 ones/row
            _, cols2 = bits2.nonzero(as_tuple=True)
            cols2 = cols2.view(-1, 2)                        # ascending within each row (verified)
            per_net_steiner[lam2] = self.D_t[cols2[:, 0], cols2[:, 1]].to(torch.int32)

        lam3 = (per_net_lambda == 3).nonzero(as_tuple=True)[0]
        if lam3.numel() > 0:
            bits3 = self._bit_planes(pin_bm[lam3])
            _, cols3 = bits3.nonzero(as_tuple=True)
            cols3 = cols3.view(-1, 3)
            # exact for 3 terminals: min over every candidate branch point v of
            # D[v,t0]+D[v,t1]+D[v,t2] (see region_graph.py module docstring /
            # T1 report for the branch-disjointness proof at the optimal v).
            d3 = self.D_t[:, cols3[:, 0]] + self.D_t[:, cols3[:, 1]] + self.D_t[:, cols3[:, 2]]
            per_net_steiner[lam3] = d3.min(dim=0).values.to(torch.int32)

        lam_ge4 = (per_net_lambda >= 4).nonzero(as_tuple=True)[0]
        if lam_ge4.numel() > 0:
            ge4_bm = pin_bm[lam_ge4].cpu().numpy().tolist()
            st_ge4 = np.empty(len(ge4_bm), dtype=np.int64)
            for i, bm in enumerate(ge4_bm):
                terms = [b for b in range(self.k) if (bm >> b) & 1]
                st, _ft, _exact = steiner_tree_stats(self._rg_D, self._rg_next_hop, terms)
                st_ge4[i] = st
            per_net_steiner[lam_ge4] = torch.from_numpy(st_ge4.astype(np.int32)).to(dev)

        io_rg = int(per_net_steiner.sum().item())
        ft_rg = io_rg - hard_lambda_sum

        # hpwl: only degree>=2 nets contribute (matches evaluate_ref's `if d<=1: continue`)
        neg_inf = torch.finfo(torch.float64).min
        pos_inf = torch.finfo(torch.float64).max
        max_x = torch.full((n_nets,), neg_inf, dtype=torch.float64, device=dev)
        min_x = torch.full((n_nets,), pos_inf, dtype=torch.float64, device=dev)
        max_y = torch.full((n_nets,), neg_inf, dtype=torch.float64, device=dev)
        min_y = torch.full((n_nets,), pos_inf, dtype=torch.float64, device=dev)
        max_x.scatter_reduce_(0, self.pin2net_t, px, reduce="amax", include_self=True)
        min_x.scatter_reduce_(0, self.pin2net_t, px, reduce="amin", include_self=True)
        max_y.scatter_reduce_(0, self.pin2net_t, py, reduce="amax", include_self=True)
        min_y.scatter_reduce_(0, self.pin2net_t, py, reduce="amin", include_self=True)
        hpwl_term = (max_x - min_x) + (max_y - min_y)
        has2 = self.degrees_t >= 2
        hpwl_term = torch.where(has2, hpwl_term, torch.zeros_like(hpwl_term))
        hpwl = float(hpwl_term.sum().item())
        del max_x, min_x, max_y, min_y, hpwl_term, has2  # M4 T2: see the del comment above

        # large nets (degree > max_degree): presence lower bound, no MST/tree_wl/pairs
        per_net_crossings = torch.zeros(n_nets, dtype=torch.int64, device=dev)
        large_lb = 0
        if self.large_net_ids_t.numel() > 0:
            lb = self._popcount_k(pin_bm[self.large_net_ids_t]) - 1
            per_net_crossings[self.large_net_ids_t] = lb
            large_lb = int(lb.sum().item())

        edge_net_id, edge_pin_a, edge_pin_b = self._batch_mst(px, py)

        per_net_ft = torch.zeros(n_nets, dtype=torch.int64, device=dev)
        pair_demand = {}
        # tree_wl as a device-side 0-dim accumulator (M4 T2 item 4): each
        # batch's partial float64 sum is added in-place on the GPU, with a
        # single .item() sync at the very end (matching the "sync once, not
        # per chunk" pattern _reduce_pair_demand already uses below) --
        # cheaper than a per-batch host round trip and, per spec §4.2's
        # empirical evidence (heavy-tail float64 reduction, batch=1e6/3e6/8e6/
        # full all agreeing to within ~1e-16 relative), does not change this
        # field's accuracy contract (rel<=1e-12 across batch sizes, not
        # bit-exact -- float64 addition is not associative, so a different
        # batch size *can* land on a different but equally valid rounding).
        tree_wl_t = torch.zeros((), dtype=torch.float64, device=dev)

        # v2 P-D (design sec 5): capacity_fields collects everything that goes
        # into EvalResult only when segments= was supplied, so a caller that
        # never asks for capacity sees the same None-filled EvalResult it
        # always has (Task 6's "not measured" vs "measured as zero" contract).
        capacity_fields = {}

        M = edge_net_id.numel()
        if M > 0:
            # M4 T2 items 4/7: the MST edge arrays (edge_net_id/edge_pin_a/
            # edge_pin_b, M ~= total pins) are already streamed out of
            # _batch_mst at a bounded per-degree-bucket chunk size, but the
            # *downstream* per-edge pipeline below (geometry gather -> lattice
            # index -> Ph/Pv crossing lookup -> _process_segments) used to run
            # on the full (M,) arrays in one shot -- that's every "MST edge
            # array" / "edge endpoint coordinates" / "segment description" row
            # in spec §4.1's tensor budget at full M size simultaneously.
            # Batching over edge_batch_size here bounds those same rows to a
            # fixed size (a construction parameter, defaulting to 1,000,000 --
            # see the __init__ docstring for why not 8,000,000) regardless of
            # M, and per_net_crossings/passed_bit_acc/
            # pair_count_acc/tree_wl_t are all incrementally accumulated
            # in-place across batches -- order-independent for the integer
            # accumulators (exact addition/amax-OR) and within the ~1e-16
            # float64-reduction-order slack documented above for tree_wl.
            passed_bit_acc = torch.zeros((n_nets, self.k), dtype=torch.int8, device=dev)
            pair_count_acc = torch.zeros(64 * 64, dtype=torch.int64, device=dev)
            # v2 P-D: segment_demand_acc/candidate_keys mirror pair_count_acc's
            # incremental-accumulation pattern above -- only allocated when
            # segments= is in play, so a straddle-only or legacy-only caller
            # pays nothing extra.
            segment_demand_acc = None
            candidate_keys = None
            if self.segments is not None:
                segment_demand_acc = torch.zeros(self.num_segments,
                                                 dtype=torch.int64, device=dev)
                if capacity_candidates:
                    candidate_keys = {"keys": [], "counts": [], "dropped": 0}
            batch = self._edge_batch_size
            for start in range(0, M, batch):
                end = min(M, start + batch)
                b_net_id = edge_net_id[start:end]
                b_pin_a = edge_pin_a[start:end]
                b_pin_b = edge_pin_b[start:end]

                xa, ya = px[b_pin_a], py[b_pin_a]
                xb, yb = px[b_pin_b], py[b_pin_b]
                tree_wl_t += (torch.abs(xa - xb) + torch.abs(ya - yb)).sum()

                ax, ay = self._to_idx(xa, ya)
                bx, by = self._to_idx(xb, yb)

                # v2 P-D (D-1/D-2): the candidate key is grouped on the MST
                # edge's terminal-pin region ids, gathered from pin_rid here
                # (the same global pin indices _batch_mst drew b_pin_a/b_pin_b
                # from) -- see the pin_rid lifetime comments above.
                edge_u = edge_v = None
                if candidate_keys is not None:
                    edge_u = pin_rid[b_pin_a]
                    edge_v = pin_rid[b_pin_b]

                # horizontal segment (at ay, matching edge_regions_and_crossings' (x0,y0)->(x1,y0) leg)
                h_row = ay
                h_lo = torch.minimum(ax, bx)
                h_hi = torch.maximum(ax, bx)
                # vertical segment (at bx, matching the (x1,y0)->(x1,y1) leg)
                v_col = bx
                v_lo = torch.minimum(ay, by)
                v_hi = torch.maximum(ay, by)

                h_cross = self.Ph[h_row, h_hi] - self.Ph[h_row, h_lo]
                v_cross = self.Pv[v_hi, v_col] - self.Pv[v_lo, v_col]
                # per_net_crossings stays int64 (unlike the index tensors
                # above, this is an accumulated *count*, not a bounded index --
                # widen the int32 Ph/Pv difference before scatter_add_, which
                # requires self.dtype == src.dtype).
                edge_cross = (h_cross + v_cross).to(torch.int64)
                per_net_crossings.scatter_add_(0, b_net_id, edge_cross)

                # R2 remedy (unchanged from M3 T1): passed_bit_acc is the other
                # (E,K) int64 -> int8 accumulator (paired with
                # _process_segments' int8 seg_bits source).
                self._process_segments(h_row, h_lo, h_hi, b_net_id, False,
                                       passed_bit_acc, pair_count_acc,
                                       segment_demand_acc, edge_u, edge_v,
                                       candidate_keys)
                self._process_segments(v_col, v_lo, v_hi, b_net_id, True,
                                       passed_bit_acc, pair_count_acc,
                                       segment_demand_acc, edge_u, edge_v,
                                       candidate_keys)

            passed_bm = self._pack_bits(passed_bit_acc)
            ft_bm = passed_bm & (~pin_bm)
            per_net_ft = self._popcount_k(ft_bm)

            pair_demand = self._reduce_pair_demand(pair_count_acc)

            if segment_demand_acc is not None:
                capacity_fields["segment_demand"] = \
                    segment_demand_acc.cpu().numpy().astype(np.int64)
            if candidate_keys is not None:
                capacity_fields.update(self._reduce_candidates(candidate_keys))

        # v2 P-D: pin_rid's last possible use is the edge_u/edge_v gather
        # inside the loop above (only when need_pin_rid_for_capacity); explicit
        # del once past that point, matching this file's del discipline. Safe
        # even when M == 0 (loop body, and hence the gather, never ran) since
        # pin_rid was only left alive in the first place when
        # need_pin_rid_for_capacity was True.
        if need_pin_rid_for_capacity:
            del pin_rid

        if self.segments is not None:
            demand = capacity_fields.setdefault(
                "segment_demand", np.zeros(self.num_segments, dtype=np.int64))
            # sec 5: the Ph/Pv prefix-sum counts become a free assertion. One
            # device sync on two scalars, not per edge.
            segment_demand_total = int(demand.sum())
            assert segment_demand_total == int(per_net_crossings.sum().item()) - large_lb, \
                "per-segment demand does not reconcile with the Ph/Pv crossing count"
            # Task-7 fix round 1: the same sum the assert above already
            # computes, not a second independent computation -- unconditional
            # (does not need segment_capacity), mirroring evaluator_ref.
            capacity_fields["segment_demand_total"] = segment_demand_total
            if self.segment_capacity is not None:
                util, scalars = segment_utilisation(demand, self.segment_capacity)
                capacity_fields["segment_capacity"] = self.segment_capacity
                capacity_fields["segment_util"] = util
                capacity_fields.update(
                    (name, scalars[name]) for name in
                    ("num_over_capacity", "num_zero_capacity_segments",
                     "zero_capacity_demand", "max_util", "p99_util"))
            if capacity_candidates and "cand_net" not in capacity_fields:
                empty = np.zeros(0, dtype=np.int64)
                capacity_fields.update(cand_net=empty, cand_u=empty, cand_v=empty,
                                       cand_seg=empty, cand_count=empty,
                                       cand_dropped=0)

        tree_wl = float(tree_wl_t.item())

        return EvalResult(
            io_count=int(per_net_crossings.sum().item()),
            ft_count=int(per_net_ft.sum().item()),
            tree_wl=tree_wl,
            hpwl=hpwl,
            per_net_crossings=per_net_crossings.to(torch.int32).cpu().numpy(),
            per_net_ft=per_net_ft.to(torch.int32).cpu().numpy(),
            boundary_pair_demand=pair_demand,
            large_net_lb=large_lb,
            hard_lambda_sum=hard_lambda_sum,
            per_net_lambda=per_net_lambda.to(torch.int32).cpu().numpy(),
            io_rg=io_rg,
            ft_rg=ft_rg,
            per_net_steiner=per_net_steiner.cpu().numpy(),
            per_net_home=per_net_home.to(torch.uint8).cpu().numpy(),
            **straddle_stats,
            **capacity_fields,
        )


def evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda", straddle=True,
                 segments=None, segment_capacity=None, capacity_candidates=False):
    ctx = GpuEvalContext(nl, rg, device=device, max_degree=max_degree,
                         straddle=straddle, segments=segments,
                         segment_capacity=segment_capacity)
    return ctx.evaluate(node_x, node_y, capacity_candidates=capacity_candidates)
