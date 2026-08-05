import warnings
import numpy as np
from ioplace.netlist import pin_positions

class RegionGrid:
    def __init__(self, rs):
        rs.validate()
        self.die = rs.die
        self.k = rs.k
        n = rs.lattice
        self.nx = self.ny = n
        xl, yl, xh, yh = rs.die
        self.cell_w = (xh - xl) / n
        self.cell_h = (yh - yl) / n
        self.grid = np.full((n, n), -1, dtype=np.int16)
        for rid, r in enumerate(rs.regions):
            for (rxl, ryl, rxh, ryh) in r.rects:
                ix0 = int(round((rxl - xl) / self.cell_w))
                ix1 = int(round((rxh - xl) / self.cell_w))
                iy0 = int(round((ryl - yl) / self.cell_h))
                iy1 = int(round((ryh - yl) / self.cell_h))
                self.grid[iy0:iy1, ix0:ix1] = rid
        assert (self.grid >= 0).all()

    def to_idx(self, x, y):
        xl, yl, xh, yh = self.die
        ix = np.clip(((x - xl) / self.cell_w).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((y - yl) / self.cell_h).astype(np.int64), 0, self.ny - 1)
        return ix, iy

    def _to_idx(self, x, y):
        """Deprecated alias for to_idx -- kept so any pre-existing external caller
        of the old private name keeps working. New code should call to_idx()."""
        warnings.warn("RegionGrid._to_idx is deprecated; use to_idx()", DeprecationWarning, stacklevel=2)
        return self.to_idx(x, y)

    def region_of_points(self, x, y):
        ix, iy = self.to_idx(np.asarray(x, dtype=np.float64),
                              np.asarray(y, dtype=np.float64))
        return self.grid[iy, ix]

    def pin_region_bitmask(self, nl, node_x, node_y):
        assert self.k <= 64
        px, py = pin_positions(nl, node_x, node_y)
        pin_rid = self.region_of_points(px, py).astype(np.uint64)
        bm = np.zeros(nl.num_nets, dtype=np.uint64)
        np.bitwise_or.at(bm, nl.pin2net, np.left_shift(np.uint64(1), pin_rid))
        return bm
