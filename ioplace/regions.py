from dataclasses import dataclass, field
import json
import numpy as np

@dataclass
class RegionSpec:
    name: str
    rects: np.ndarray  # (R,4) [xl,yl,xh,yh]

@dataclass
class RegionSet:
    die: tuple
    lattice: int
    regions: list

    @property
    def k(self):
        return len(self.regions)

    def _cell_wh(self):
        xl, yl, xh, yh = self.die
        return (xh - xl) / self.lattice, (yh - yl) / self.lattice

    def validate(self):
        xl, yl, xh, yh = self.die
        cw, ch = self._cell_wh()
        cover = np.zeros((self.lattice, self.lattice), dtype=np.int32)
        for r in self.regions:
            rects = np.asarray(r.rects, dtype=np.float64).reshape(-1, 4)
            area = float(((rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])).sum())
            if area == 0:
                raise ValueError(f"region {r.name} has zero area")
            for (rxl, ryl, rxh, ryh) in r.rects:
                for v, lo, step in ((rxl, xl, cw), (rxh, xl, cw), (ryl, yl, ch), (ryh, yl, ch)):
                    idx = (v - lo) / step
                    if abs(idx - round(idx)) > 1e-6:
                        raise ValueError(f"region {r.name}: coord {v} not on lattice")
                ix0 = int(round((rxl - xl) / cw)); ix1 = int(round((rxh - xl) / cw))
                iy0 = int(round((ryl - yl) / ch)); iy1 = int(round((ryh - yl) / ch))
                cover[iy0:iy1, ix0:ix1] += 1
        if (cover > 1).any():
            raise ValueError("regions overlap")
        if (cover == 0).any():
            raise ValueError("regions do not cover die")

    def to_json(self, path):
        obj = {"die": list(self.die), "lattice": self.lattice,
               "regions": [{"name": r.name, "rects": np.asarray(r.rects).tolist()}
                           for r in self.regions]}
        with open(path, "w") as f:
            json.dump(obj, f, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            obj = json.load(f)
        regs = [RegionSpec(d["name"], np.array(d["rects"], dtype=np.float64))
                for d in obj["regions"]]
        return cls(die=tuple(obj["die"]), lattice=int(obj["lattice"]), regions=regs)

def _snap(v, lo, step):
    return lo + round((v - lo) / step) * step

def make_grid_regions(die, nx, ny, lattice=512):
    xl, yl, xh, yh = die
    cw, ch = (xh - xl) / lattice, (yh - yl) / lattice
    xs = [_snap(xl + (xh - xl) * i / nx, xl, cw) for i in range(nx + 1)]
    ys = [_snap(yl + (yh - yl) * j / ny, yl, ch) for j in range(ny + 1)]
    regs = []
    for j in range(ny):
        for i in range(nx):
            regs.append(RegionSpec(f"P{j*nx+i}",
                np.array([[xs[i], ys[j], xs[i+1], ys[j+1]]], dtype=np.float64)))
    return RegionSet(die=die, lattice=lattice, regions=regs)

def make_slicing_regions(die, k, seed=0, lattice=512, min_frac=0.1):
    xl, yl, xh, yh = die
    cw, ch = (xh - xl) / lattice, (yh - yl) / lattice
    rng = np.random.default_rng(seed)
    boxes = [np.array([xl, yl, xh, yh], dtype=np.float64)]
    while len(boxes) < k:
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        b = boxes.pop(int(np.argmax(areas)))
        w, h = b[2]-b[0], b[3]-b[1]
        frac = rng.uniform(min_frac, 1.0 - min_frac)
        if w >= h:
            cut = _snap(b[0] + w * frac, xl, cw)
            if cut <= b[0] or cut >= b[2]:
                cut = _snap(b[0] + w * 0.5, xl, cw)
            boxes += [np.array([b[0], b[1], cut, b[3]]), np.array([cut, b[1], b[2], b[3]])]
        else:
            cut = _snap(b[1] + h * frac, yl, ch)
            if cut <= b[1] or cut >= b[3]:
                cut = _snap(b[1] + h * 0.5, yl, ch)
            boxes += [np.array([b[0], b[1], b[2], cut]), np.array([b[0], cut, b[2], b[3]])]
    regs = [RegionSpec(f"P{i}", b.reshape(1, 4)) for i, b in enumerate(boxes)]
    return RegionSet(die=die, lattice=lattice, regions=regs)
