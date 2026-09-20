"""GrandPlan Eq.1/Eq.2 grouping loss (digest section 4.1), evaluated through the
rasterised anchor tables of producer/hull.py.

Attached to DREAMPlace with dp_hook.attach_terms -- no new patch. The tables are
rebuilt every T_hull iterations by run_region_producer's iteration callback and
are FROZEN in between, which is the paper's own semantics (digest section 2.1:
the hull is "recomputed from the current cells each time and then held constant
inside the gradient"). With a frozen anchor Eq.2 is the exact gradient, so the
implementation is literally a quadratic spring and autograd gets it right.

Eq.3's coefficient lambda is NOT computed here (ruling D5 / spec section 0):
norm.TermNormalizer is the single owner of every extra term's coefficient, and
run_region_producer registers this term with it. This module is a pure energy.
"""
import numpy as np
import torch


class GroupingTerm(torch.nn.Module):
    """Callable as `term(pos, lam)`; the driver wraps it in a one-argument
    closure for dp_hook.attach_terms.

    Positions are the cell CENTRE (spec section 7's --node-anchor center):
    x + 0.5*node_size_x, y + 0.5*node_size_y. Only the movable prefix is read,
    so fixed nodes and fillers structurally receive no gradient.
    """

    def __init__(self, part, node_size_x, node_size_y, num_movable,
                 num_nodes, alpha_pull=1.0, alpha_push=1.0, device="cuda"):
        super().__init__()
        m = int(num_movable)
        assert len(part) == m, f"part has {len(part)} entries, expected {m}"
        self.num_movable = m
        # No num_physical (ruling D9): only the movable prefix and the total
        # node count index into `pos`, so storing it was dead state.
        self.num_nodes = int(num_nodes)
        self.alpha_pull = float(alpha_pull)
        self.alpha_push = float(alpha_push)
        self.n_rebuilds = 0
        self.tables = None
        self.register_buffer("part", torch.as_tensor(
            np.asarray(part, dtype=np.int64), dtype=torch.int64, device=device))
        self.register_buffer("half_w", torch.as_tensor(
            0.5 * np.asarray(node_size_x, dtype=np.float64)[:m],
            dtype=torch.float64, device=device))
        self.register_buffer("half_h", torch.as_tensor(
            0.5 * np.asarray(node_size_y, dtype=np.float64)[:m],
            dtype=torch.float64, device=device))

    def set_tables(self, tables):
        """Install a freshly rasterised AnchorTables. Discretely changes the
        objective: the caller MUST bump its obj_version and call
        dp_hook.refresh_nesterov_secant before the next optimizer step."""
        assert tables.k > int(self.part.max()), \
            "anchor tables must cover every region id in `part`"
        self.tables = tables
        self.n_rebuilds += 1

    def _centres(self, pos):
        m = self.num_movable
        x = pos[:m].double() + self.half_w
        y = pos[self.num_nodes:self.num_nodes + m].double() + self.half_h
        return x, y

    def _lookup(self, x, y):
        """-> (bin_centre_x, bin_centre_y, pull_off, pull_on, push_off, push_cnt),
        all detached: the anchor is frozen inside the gradient.

        The returned offsets are in the die's own units. `AnchorTables` stores
        them in BIN WIDTHS -- fp16's range then depends on the lattice rather
        than on the die size, which is what keeps a 30M-cell die from tripping
        the overflow guard (P-C Task 10 fix round 1, finding I3) -- so this is
        the one place that multiplies them back by the bin size. One extra
        (M, 2) multiply per lookup; the fp16 quantisation is unchanged, still
        bounded by 1/4 bin.
        """
        t = self.tables
        xl, yl, xh, yh = t.die
        L = t.lattice
        cw, ch = (xh - xl) / L, (yh - yl) / L
        with torch.no_grad():
            ix = ((x - xl) / cw).floor().long().clamp_(0, L - 1)
            iy = ((y - yl) / ch).floor().long().clamp_(0, L - 1)
            b = iy * L + ix
            k = self.part
            cx = xl + (ix.double() + 0.5) * cw
            cy = yl + (iy.double() + 0.5) * ch
            bin_size = torch.tensor([cw, ch], dtype=torch.float64,
                                    device=x.device)
            return (cx, cy, t.pull_off[k, b].double() * bin_size,
                    t.pull_on[k, b],
                    t.push_off[k, b].double() * bin_size,
                    t.push_cnt[k, b].double())

    def forward(self, pos, lam):
        if self.tables is None or lam == 0.0:
            return pos.new_zeros(())
        x, y = self._centres(pos)
        cx, cy, pull_off, pull_on, push_off, push_n = self._lookup(x, y)
        dxp = x - (cx + pull_off[:, 0])
        dyp = y - (cy + pull_off[:, 1])
        pull = (0.5 * self.alpha_pull) * (
            pull_on.double() * (dxp * dxp + dyp * dyp)).sum()
        dxs = x - (cx + push_off[:, 0])
        dys = y - (cy + push_off[:, 1])
        push = (0.5 * self.alpha_push) * (
            push_n * (dxs * dxs + dys * dys)).sum()
        return (float(lam) * (pull + push)).to(pos.dtype)

    def grad_l1(self, pos):
        """||grad Group||_1 at lam=1 on a detached clone. Standalone diagnostic
        only: the production probe is norm.TermNormalizer.probe, which does the
        same isolated fwd+bwd and additionally zeroes the fixed and filler
        entries before taking the norm (ruling D5)."""
        if self.tables is None:
            return 0.0
        p = pos.detach().clone().requires_grad_(True)
        self.forward(p, lam=1.0).backward()
        return float(p.grad.abs().sum())
