"""Simulated-annealing refinement of the extracted bin map.

Energies are digest section 4.2's Eq.4-8 verbatim; the schedule and move set are
digest section 5 with spec section 2's defaults for every value the paper leaves
unspecified. Pure numpy/scipy, deterministic given `SaConfig.seed`.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_CC = ndimage.generate_binary_structure(2, 1)      # 4-connected, as region_graph
# Every axis-aligned window shape with 2..9 bins (digest section 5).
_WINDOWS = tuple((w, h) for w in range(1, 10) for h in range(1, 10)
                 if 2 <= w * h <= 9)


@dataclass
class SaConfig:
    beta: tuple = (1.0, 0.3, 0.5, 0.2)
    theta: float = 0.05
    c_max: int = 2
    rho_target: float = 0.8
    probe_moves: int = 200
    cooling: float = 0.92
    moves_per_level: int = 50
    levels: int = 150
    idle_levels_stop: int = 3
    corner_tries: int = 20
    seed: int = 0


# ------------------------------------------------------------------ energies

def e_area(labels, k, ea, bin_area, theta=0.05):
    """Eq.5: d_i = (A_min,i - A_c,i)/EA_i with A_min,i = (1-theta)*EA_i;
    E_area,i = 50[d_i]_+^3 + 25[d_i]_+^2. Only shortfall is penalised."""
    cnt = np.bincount(np.asarray(labels).ravel(), minlength=k)[:k].astype(np.float64)
    ea = np.asarray(ea, dtype=np.float64)
    d = ((1.0 - theta) * ea - cnt * float(bin_area)) / np.where(ea > 0.0, ea, 1.0)
    dp = np.maximum(d, 0.0)
    return float((50.0 * dp ** 3 + 25.0 * dp ** 2).sum())


def shared_edges(labels, k):
    """(k,k) count of unit lattice edges separating each pair -- the same
    quantity region_graph.region_graph calls `ell` (region_graph.py:62-71),
    computed here on a bin map instead of a RegionGrid."""
    a = np.asarray(labels, dtype=np.int64)
    ell = np.zeros((k, k), dtype=np.int64)
    for p, q in ((a[:, :-1], a[:, 1:]), (a[:-1, :], a[1:, :])):
        m = p != q
        np.add.at(ell, (p[m], q[m]), 1)
        np.add.at(ell, (q[m], p[m]), 1)
    return ell


def corner_counts(labels, k):
    """C_ij for Eq.6. At each interior lattice vertex the four surrounding bins
    define four incident unit edges (S, N, W, E). Pair (i,j) has a corner there
    iff exactly two of those edges separate i from j and they are perpendicular
    -- one of {S,N} and one of {W,E}. Straight boundary -> 0, L-turn -> 1,
    one-bin notch -> 4."""
    a = np.asarray(labels, dtype=np.int64)
    A, Bb, C, D = a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]

    def key(p, q):
        return np.where(p == q, -1, np.minimum(p, q) * k + np.maximum(p, q))

    kS, kN, kW, kE = key(A, Bb), key(C, D), key(A, C), key(Bb, D)
    flat = np.zeros(k * k, dtype=np.int64)
    for p, q, o1, o2 in ((kS, kW, kN, kE), (kS, kE, kN, kW),
                         (kN, kW, kS, kE), (kN, kE, kS, kW)):
        m = (p >= 0) & (p == q) & (o1 != p) & (o2 != p)
        if m.any():
            np.add.at(flat, p[m], 1)
    upper = flat.reshape(k, k)
    return upper + upper.T


def e_boundary(labels, k, c_max=2):
    """Eq.6 over adjacent pairs only: sum 0.1*(C_ij - C_max)^2. Not one-sided
    (digest section 4.2), so a perfectly straight shared boundary still costs
    0.1*C_max^2."""
    ell = shared_edges(labels, k)
    cc = corner_counts(labels, k)
    iu = np.triu_indices(k, 1)
    adj = ell[iu] > 0
    return float((0.1 * (cc[iu][adj].astype(np.float64) - c_max) ** 2).sum())


def e_compact(labels, k, rho_target=0.8):
    """Eq.7: 5.0 * ((rho_target - rho_fill,i)/rho_target)_+^2 with
    rho_fill,i = A_c,i / A_bb,i on the bin grid."""
    lab = np.asarray(labels)
    total = 0.0
    for kk in range(k):
        m = lab == kk
        n = int(m.sum())
        if n == 0:
            continue
        ys, xs = np.nonzero(m)
        abb = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        d = max((rho_target - n / abb) / rho_target, 0.0)
        total += 5.0 * d * d
    return float(total)


def e_diff(labels, labels0):
    """Eq.8: D_diff / N_bins."""
    lab = np.asarray(labels)
    return float(np.count_nonzero(lab != np.asarray(labels0))) / float(lab.size)


# ------------------------------------------------------------------- SA state

def _halo(mask):
    """4-neighbourhood of a boolean mask (the mask itself excluded by caller)."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


class SaState:
    """Holds the map, the frozen min-max normalisation and one pending move.

    Move protocol: `mv = propose(rng)`; `d = try_move(mv)` applies it, rejects
    (returns None, already rolled back) if it fragments or empties a region,
    otherwise returns the normalised energy delta; then exactly one of
    `commit()` / `rollback()`.
    """

    def __init__(self, labels0, k, ea, bin_area, cfg):
        self.cfg = cfg
        self.k = int(k)
        self.ea = np.asarray(ea, dtype=np.float64)
        self.bin_area = float(bin_area)
        self.labels0 = np.asarray(labels0, dtype=np.int16).copy()
        self.labels = self.labels0.copy()
        self._flat = self.labels.reshape(-1)
        self.lo = np.zeros(4)
        self.hi = np.ones(4)
        self._pending = None
        self._raw = self.raw()

    def raw(self):
        c = self.cfg
        return np.array([
            e_area(self.labels, self.k, self.ea, self.bin_area, c.theta),
            e_boundary(self.labels, self.k, c.c_max),
            e_compact(self.labels, self.k, c.rho_target),
            e_diff(self.labels, self.labels0)], dtype=np.float64)

    def total(self, raw):
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        norm = (np.asarray(raw, dtype=np.float64) - self.lo) / span
        return float(np.dot(np.asarray(self.cfg.beta, dtype=np.float64), norm)), norm

    # -- move application ---------------------------------------------------
    def _apply(self, mv):
        flat, new = mv
        old = self._flat[flat].copy()
        self._flat[flat] = np.int16(new)
        self._pending = (flat, old, int(new))

    def rollback(self):
        flat, old, _ = self._pending
        self._flat[flat] = old
        self._pending = None

    def commit(self):
        self._raw = self.raw()
        self._pending = None

    def _breaks_a_region(self):
        flat, old, new = self._pending
        for kk in set(int(v) for v in np.unique(old)) | {new}:
            m = self.labels == kk
            if not m.any():
                return True
            if ndimage.label(m, structure=_CC)[1] != 1:
                return True
        return False

    def try_move(self, mv):
        if mv is None:
            return None
        self._apply(mv)
        if self._breaks_a_region():
            self.rollback()
            return None
        return self.total(self.raw())[0] - self.total(self._raw)[0]

    # -- move generation ----------------------------------------------------
    def _deficits(self):
        cnt = np.bincount(self._flat, minlength=self.k)[:self.k].astype(np.float64)
        a_min = (1.0 - self.cfg.theta) * self.ea
        need = np.maximum(np.ceil((a_min - cnt * self.bin_area) / self.bin_area), 0)
        excess = np.maximum(np.floor((cnt * self.bin_area - a_min) / self.bin_area), 0)
        return need.astype(np.int64), excess.astype(np.int64)

    def has_area_violation(self):
        return bool((self._deficits()[0] > 0).any())

    def _area_balancing(self, rng):
        """digest section 5: pick a deficit region, score every contiguous run
        of a surplus neighbour's boundary bins by
        min(#excess_bins, #bins_needed, run length), and take the best run's
        bins in raster order."""
        need, excess = self._deficits()
        deficit = np.nonzero(need > 0)[0]
        if not len(deficit):
            return None
        t = int(deficit[int(rng.integers(len(deficit)))])
        tmask = self.labels == t
        halo = _halo(tmask) & ~tmask
        if not halo.any():
            return None
        best = None
        for d in np.unique(self.labels[halo]):
            d = int(d)
            if excess[d] <= 0:
                continue
            cc, n = ndimage.label(halo & (self.labels == d), structure=_CC)
            ccf = cc.reshape(-1)
            for c in range(1, n + 1):
                run = np.flatnonzero(ccf == c)
                score = int(min(excess[d], need[t], len(run)))
                if score <= 0:
                    continue
                key = (-score, d, int(run[0]))
                if best is None or key < best[0]:
                    best = (key, run[:score])
        if best is None:
            return None
        return best[1], t

    def _corner_filling(self, rng):
        """digest section 5: a random 2..9-bin window on a boundary; every bin
        in it goes to the window's majority label. "On a boundary" is the
        operational corner test -- the only one defined for a 1x2 window; Eq.6
        does the actual corner accounting."""
        b = self.labels.shape[0]
        for _ in range(self.cfg.corner_tries):
            w, h = _WINDOWS[int(rng.integers(len(_WINDOWS)))]
            if w > b or h > b:
                continue
            x0 = int(rng.integers(0, b - w + 1))
            y0 = int(rng.integers(0, b - h + 1))
            win = self.labels[y0:y0 + h, x0:x0 + w]
            vals, cnts = np.unique(win, return_counts=True)
            if len(vals) < 2:
                continue
            maj = int(vals[int(np.argmax(cnts))])       # np.unique sorts -> low id wins
            idx = np.nonzero(win.reshape(-1) != maj)[0]
            ys, xs = np.divmod(idx, w)
            return ((y0 + ys) * b + (x0 + xs)).astype(np.int64), maj
        return None

    def propose(self, rng, force_both=False):
        """Equal probability between the two move types while any area
        violation exists; corner-filling only afterwards (digest section 5)."""
        if (force_both or self.has_area_violation()) and rng.random() < 0.5:
            mv = self._area_balancing(rng)
            if mv is not None:
                return mv
        return self._corner_filling(rng)

    # -- calibration --------------------------------------------------------
    def calibrate(self, rng):
        """Min-max normalise each term over the first `probe_moves` samples
        (digest section 4.2: "normalized into similar scale before we use the
        weights beta_j"), then T_0 = mean |Delta E| over the same probes (spec
        section 2). Leaves the map exactly as it found it."""
        base = self._raw.copy()
        samples = [base]
        probes = []
        for _ in range(self.cfg.probe_moves):
            mv = self.propose(rng, force_both=True)
            if mv is None:
                continue
            self._apply(mv)
            if self._breaks_a_region():
                self.rollback()
                continue
            r = self.raw()
            samples.append(r)
            probes.append(r)
            self.rollback()
        arr = np.stack(samples)
        self.lo, self.hi = arr.min(axis=0), arr.max(axis=0)
        base_tot = self.total(base)[0]
        deltas = [abs(self.total(r)[0] - base_tot) for r in probes]
        t0 = float(np.mean(deltas)) if deltas else 1.0
        return self.lo, self.hi, max(t0, 1e-9)


def anneal(labels0, k, ea, bin_area, cfg=None):
    """Run the full schedule. Returns (labels int16, report dict)."""
    cfg = cfg or SaConfig()
    st = SaState(labels0, k, ea, bin_area, cfg)
    rng = np.random.default_rng(cfg.seed)
    _, _, t0 = st.calibrate(rng)
    raw_initial = st.raw()
    st._raw = raw_initial.copy()
    temp = t0
    idle = 0
    accepted = proposed = 0
    level = 0
    for level in range(1, cfg.levels + 1):
        n_acc = 0
        for _ in range(cfg.moves_per_level):
            mv = st.propose(rng)
            if mv is None:
                continue
            proposed += 1
            d = st.try_move(mv)
            if d is None:
                continue
            if d <= 0.0 or rng.random() < np.exp(-d / max(temp, 1e-12)):
                st.commit()
                n_acc += 1
            else:
                st.rollback()
        accepted += n_acc
        idle = idle + 1 if n_acc == 0 else 0
        if idle >= cfg.idle_levels_stop:
            break
        temp *= cfg.cooling
    raw_final = st.raw()
    report = {
        "t0": float(t0), "levels_run": int(level),
        "moves_proposed": int(proposed), "moves_accepted": int(accepted),
        "beta": list(cfg.beta),
        "norm_lo": st.lo.tolist(), "norm_hi": st.hi.tolist(),
        "e_raw_initial": raw_initial.tolist(), "e_raw_final": raw_final.tolist(),
        "e_norm_initial": st.total(raw_initial)[1].tolist(),
        "e_norm_final": st.total(raw_final)[1].tolist(),
        "e_total_initial": st.total(raw_initial)[0],
        "e_total_final": st.total(raw_final)[0]}
    return st.labels, report
