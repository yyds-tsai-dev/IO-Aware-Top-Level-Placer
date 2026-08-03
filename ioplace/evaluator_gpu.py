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
derived from integer grid-index comparisons) must match exactly.

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


def _pow2_bounds(n):
    """[1, 2, 4, ...] up to the first power of two >= n (n >= 1)."""
    bounds = [1]
    while bounds[-1] < n:
        bounds.append(bounds[-1] * 2)
    return bounds


class GpuEvalContext:
    def __init__(self, nl, rg, device="cuda", max_degree=256):
        self.nl, self.rg = nl, rg
        self.device = torch.device(device)
        self.max_degree = max_degree
        self.k = rg.k
        # the bitmask vectorization below (pin_bm/passed_bm packed via _pack_bits
        # into a single int64, one bit per region id) assumes every region id fits
        # in the 64 bits of an int64 -- fail loudly here rather than silently
        # truncating/wrapping region ids >= 64 later inside _pack_bits/_bit_planes.
        assert self.k <= 64, f"GpuEvalContext bitmask vectorization requires rg.k <= 64, got rg.k={self.k}"

        # ---- static region-grid tensors: grid + Ph/Pv crossing prefix sums ----
        grid_np = rg.grid.astype(np.int64)
        self.grid_t = torch.from_numpy(grid_np).to(self.device)
        ny, nx = grid_np.shape
        self.nx, self.ny = nx, ny
        xl, yl, xh, yh = rg.die
        self.xl, self.yl = float(xl), float(yl)
        self.cell_w, self.cell_h = float(rg.cell_w), float(rg.cell_h)

        hdiff = (self.grid_t[:, :-1] != self.grid_t[:, 1:]).to(torch.int64)
        Ph = torch.zeros((ny, nx), dtype=torch.int64, device=self.device)
        Ph[:, 1:] = torch.cumsum(hdiff, dim=1)
        self.Ph = Ph

        vdiff = (self.grid_t[:-1, :] != self.grid_t[1:, :]).to(torch.int64)
        Pv = torch.zeros((ny, nx), dtype=torch.int64, device=self.device)
        Pv[1:, :] = torch.cumsum(vdiff, dim=0)
        self.Pv = Pv

        self._seg_bounds = _pow2_bounds(max(nx, ny))
        self._seg_bounds_t = torch.tensor(self._seg_bounds, dtype=torch.int64, device=self.device)

        # ---- bit-plane helpers, precomputed once ----
        self._bit_range = torch.arange(self.k, device=self.device, dtype=torch.int64)
        self._pow2_k = torch.bitwise_left_shift(torch.ones(self.k, dtype=torch.int64, device=self.device),
                                                 self._bit_range)

        # ---- static pin/net topology tensors ----
        self.pin2node_t = torch.from_numpy(nl.pin2node.astype(np.int64)).to(self.device)
        self.pin2net_t = torch.from_numpy(nl.pin2net.astype(np.int64)).to(self.device)
        self.pin_offset_x_t = torch.from_numpy(nl.pin_offset_x.astype(np.float64)).to(self.device)
        self.pin_offset_y_t = torch.from_numpy(nl.pin_offset_y.astype(np.float64)).to(self.device)

        self.n_nets = nl.num_nets
        degrees = nl.net_degrees
        self.degrees_t = torch.from_numpy(degrees.astype(np.int64)).to(self.device)

        start = nl.flat_net2pin_start
        flat = nl.flat_net2pin
        self._buckets = []  # [(d, net_ids (B,) long, pin_idx_mat (B,d) long)] all on device
        for d in range(2, max_degree + 1):
            net_ids = np.nonzero(degrees == d)[0]
            if len(net_ids) == 0:
                continue
            s = start[net_ids].astype(np.int64)
            offsets = np.arange(d, dtype=np.int64)
            idx_mat = s[:, None] + offsets[None, :]
            pin_idx_mat = flat[idx_mat].astype(np.int64)
            self._buckets.append((
                d,
                torch.from_numpy(net_ids.astype(np.int64)).to(self.device),
                torch.from_numpy(pin_idx_mat).to(self.device),
            ))

        large_net_ids = np.nonzero(degrees > max_degree)[0]
        self.large_net_ids_t = torch.from_numpy(large_net_ids.astype(np.int64)).to(self.device)

        # memory budget: cap a chunk's (b,d,d) distance tensor at this many elements
        self._mst_chunk_budget = 8_000_000

        # memory budget: cap a chunk's (numSelected, L) padded-gather tensor (used
        # by _process_segments) at this many elements. Per chunk, up to ~3 int64
        # tensors of this shape are concurrently alive during the bitmask-reduction
        # pass (cand/ids/vals), plus -- in the boundary-pair-demand pass that
        # follows, while cand/ids are still resident -- up to ~4 more int64 tensors
        # sized by the (bool) diff mask's nonzero count (a/b/lo_ab/hi_ab), bounded by
        # the same numSelected*L in the pathological worst case where every adjacent
        # cell differs (real region grids -- contiguous rectangular partitions --
        # see far fewer boundary transitions per segment in practice, so this is a
        # deliberately conservative bound). Reusing _mst_chunk_budget's value here
        # (same 8-byte dtype, same "cap the dominant tensor's element count" idea)
        # keeps a single mental model for both chunk budgets in this file, and
        # empirically (2M cells/nets, K=16, lattice=512 synthetic case) this value
        # keeps this function's own contribution to peak GPU memory to ~1.8GB on top
        # of the ~2.9GB pre-existing baseline (pin one-hot bitmask + MST, unrelated
        # to this fix) that evaluate() already uses before _process_segments ever
        # runs -- total measured peak 4.62GB, vs. 15.00GB unchunked before this fix.
        self._seg_chunk_budget = self._mst_chunk_budget

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _to_idx(self, x, y):
        ix = ((x - self.xl) / self.cell_w).to(torch.int64).clamp_(0, self.nx - 1)
        iy = ((y - self.yl) / self.cell_h).to(torch.int64).clamp_(0, self.ny - 1)
        return ix, iy

    def _pin_positions(self, node_x, node_y):
        px = node_x[self.pin2node_t] + self.pin_offset_x_t
        py = node_y[self.pin2node_t] + self.pin_offset_y_t
        return px, py

    def _bit_planes(self, bm):
        """(...,) int64 *already-combined* bitmask -> (...,K) 0/1 bit planes.

        NOT for one-hot-encoding a single id -- decomposing a raw id value's own
        binary digits is a different (wrong) operation; use _one_hot_planes for that.
        """
        return torch.bitwise_and(torch.bitwise_right_shift(bm.unsqueeze(-1), self._bit_range), 1)

    def _one_hot_planes(self, ids):
        """(...,) int64 id values in [0,K) -> (...,K) 0/1 one-hot planes (plane
        i is 1 iff ids==i). This is the "id -> single-bit-set bitmask" direction;
        _bit_planes is the reverse (combined bitmask -> planes) and is not
        interchangeable with this."""
        return (ids.unsqueeze(-1) == self._bit_range).to(torch.int64)

    def _popcount_k(self, bm):
        return self._bit_planes(bm).sum(dim=-1)

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
            empty = torch.empty(0, dtype=torch.int64, device=dev)
            return empty, empty, empty
        return torch.cat(net_ids_list), torch.cat(pin_a_list), torch.cat(pin_b_list)

    # ------------------------------------------------------------------
    # FT (passed-region bitmask) + boundary pair demand, via bucketed padded gather
    # ------------------------------------------------------------------
    def _process_segments(self, fixed_idx, lo, hi, net_id, is_vert, passed_bit_acc, pair_key_chunks):
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
            offsets = torch.arange(L, device=dev)

            # chunk this bucket's B selected segments so no single (chunk, L)
            # padded-gather tensor exceeds _seg_chunk_budget elements -- same
            # chunk-budget pattern as _batch_mst above. Each chunk's contribution is
            # aggregated via in-place scatter_reduce(amax) into passed_bit_acc and by
            # appending to pair_key_chunks, both order-independent across chunks, so
            # this is a pure memory-shape change with no effect on the result.
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

                # visited-region bitmask per segment: O(log L) pairwise-OR halving reduce
                vals = torch.bitwise_left_shift(torch.ones_like(ids), ids)
                length = L
                while length > 1:
                    half = length // 2
                    vals = vals[:, :half] | vals[:, half:half * 2]
                    length = half
                seg_bm = vals[:, 0]
                seg_bits = self._bit_planes(seg_bm)
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
                        pair_key_chunks.append(lo_ab * 64 + hi_ab)

    @staticmethod
    def _reduce_pair_demand(pair_key_chunks):
        if not pair_key_chunks:
            return {}
        keys = torch.cat(pair_key_chunks).cpu().numpy()
        uniq, counts = np.unique(keys, return_counts=True)
        return {(int(u) // 64, int(u) % 64): int(c) for u, c in zip(uniq, counts)}

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def evaluate(self, node_x, node_y):
        dev = self.device
        node_x = torch.as_tensor(node_x, dtype=torch.float64, device=dev)
        node_y = torch.as_tensor(node_y, dtype=torch.float64, device=dev)

        px, py = self._pin_positions(node_x, node_y)
        pin_ix, pin_iy = self._to_idx(px, py)
        pin_rid = self.grid_t[pin_iy, pin_ix]

        n_nets = self.n_nets

        # pin_bm: per-net OR of pin region bits (same semantics as RegionGrid.pin_region_bitmask)
        pin_bits = self._one_hot_planes(pin_rid)
        pin_bit_acc = torch.zeros((n_nets, self.k), dtype=torch.int64, device=dev)
        idx_exp = self.pin2net_t.unsqueeze(1).expand(-1, self.k)
        pin_bit_acc.scatter_reduce_(0, idx_exp, pin_bits, reduce="amax", include_self=True)
        pin_bm = self._pack_bits(pin_bit_acc)

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
        tree_wl = 0.0

        if edge_net_id.numel() > 0:
            xa, ya = px[edge_pin_a], py[edge_pin_a]
            xb, yb = px[edge_pin_b], py[edge_pin_b]
            tree_wl = float((torch.abs(xa - xb) + torch.abs(ya - yb)).sum().item())

            ax, ay = self._to_idx(xa, ya)
            bx, by = self._to_idx(xb, yb)

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
            edge_cross = h_cross + v_cross
            per_net_crossings.scatter_add_(0, edge_net_id, edge_cross)

            passed_bit_acc = torch.zeros((n_nets, self.k), dtype=torch.int64, device=dev)
            pair_key_chunks = []
            self._process_segments(h_row, h_lo, h_hi, edge_net_id, False, passed_bit_acc, pair_key_chunks)
            self._process_segments(v_col, v_lo, v_hi, edge_net_id, True, passed_bit_acc, pair_key_chunks)

            passed_bm = self._pack_bits(passed_bit_acc)
            ft_bm = passed_bm & (~pin_bm)
            per_net_ft = self._popcount_k(ft_bm)

            pair_demand = self._reduce_pair_demand(pair_key_chunks)

        return EvalResult(
            io_count=int(per_net_crossings.sum().item()),
            ft_count=int(per_net_ft.sum().item()),
            tree_wl=tree_wl,
            hpwl=hpwl,
            per_net_crossings=per_net_crossings.to(torch.int32).cpu().numpy(),
            per_net_ft=per_net_ft.to(torch.int32).cpu().numpy(),
            boundary_pair_demand=pair_demand,
            large_net_lb=large_lb,
        )


def evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda"):
    return GpuEvalContext(nl, rg, device=device, max_degree=max_degree).evaluate(node_x, node_y)
