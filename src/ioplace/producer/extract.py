"""Boundary extraction: legalized flat placement -> rectilinear bin map.

digest section 2.2 / spec section 2: per-partition density maps at 2048^2, per
bin argmax, majority vote down to 64^2 (32^2 for arm (e)), morphological opening
then closing, largest connected component per partition, whitespace bins to the
nearest partition. Pure numpy/scipy; the caller supplies coordinates in whatever
system it wants the output interpreted in.
"""
import numpy as np
from scipy import ndimage

FINE_BINS = 2048

# Full 3x3 square. The 4-connected cross erodes every rectangle's corners during
# opening (dilation cannot restore them), manufacturing exactly the one-bin
# notches E_boundary and the rect_max budget exist to remove.
_SE = ndimage.generate_binary_structure(2, 2)
# Connected components and adjacency stay 4-connected, matching
# region_graph.region_graph's unit-edge adjacency (region_graph.py:62-71).
_CC = ndimage.generate_binary_structure(2, 1)


def density_maps(node_x, node_y, node_w, node_h, part, k, die, bins=FINE_BINS):
    """(k, bins, bins) cell-area density, accumulated at the cell CENTRE (spec
    section 7's anchor decision). node_x/node_y are lower-left corners.

    Host memory (ruling D11): this allocates k * bins^2 float64 (537 MiB at
    K=16 / 2048^2, 1.07 GiB at K=32), plus an equal-sized np.bincount output --
    the dominant host-RAM cost of extraction, once, after the GP.
    """
    xl, yl, xh, yh = (float(v) for v in die)
    cw, ch = (xh - xl) / bins, (yh - yl) / bins
    cx = np.asarray(node_x, dtype=np.float64) + 0.5 * np.asarray(node_w, dtype=np.float64)
    cy = np.asarray(node_y, dtype=np.float64) + 0.5 * np.asarray(node_h, dtype=np.float64)
    ix = np.clip(((cx - xl) / cw).astype(np.int64), 0, bins - 1)
    iy = np.clip(((cy - yl) / ch).astype(np.int64), 0, bins - 1)
    p = np.asarray(part, dtype=np.int64)
    area = np.asarray(node_w, dtype=np.float64) * np.asarray(node_h, dtype=np.float64)
    flat = (p * bins + iy) * bins + ix
    return np.bincount(flat, weights=area,
                       minlength=k * bins * bins).reshape(k, bins, bins)


def block_sum(dens, out_bins):
    """Aggregate a (k, F, F) map into (k, out_bins, out_bins)."""
    k, f, _ = dens.shape
    assert f % out_bins == 0, f"{f} fine bins do not divide into {out_bins}"
    s = f // out_bins
    return dens.reshape(k, out_bins, s, out_bins, s).sum(axis=(2, 4))


def argmax_labels(dens):
    """Per-bin argmax partition; bins with no density become -1. Ties break on
    the lowest partition id (np.argmax's own rule)."""
    lab = dens.argmax(axis=0).astype(np.int16)
    lab[dens.sum(axis=0) <= 0.0] = -1
    return lab


def majority_downsample(fine, out_bins, k):
    """Majority vote of `fine` (F x F, -1 = empty) into out_bins x out_bins.
    Empty fine bins do not vote; a block with no votes becomes -1. Ties break on
    the lowest partition id."""
    f = fine.shape[0]
    assert f % out_bins == 0, f"{f} fine bins do not divide into {out_bins}"
    s = f // out_bins
    blocks = fine.reshape(out_bins, s, out_bins, s).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(out_bins * out_bins, s * s)
    counts = np.stack([(blocks == kk).sum(axis=1) for kk in range(k)], axis=1)
    out = np.full(out_bins * out_bins, -1, dtype=np.int16)
    has = counts.sum(axis=1) > 0
    out[has] = counts[has].argmax(axis=1).astype(np.int16)
    return out.reshape(out_bins, out_bins)


def _halo(mask):
    """4-neighbourhood of a boolean mask (the mask itself is not excluded)."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _open_close_one(m):
    """Opening then closing, with the die border treated as SET during erosion.

    Ruling D2: ndimage.binary_erosion defaults to border_value=0, i.e. it
    treats everything outside the array as background and erodes the whole
    outer ring of the die; the dilations cannot put it back, and border_value=1
    on binary_opening does not help (scipy applies that to the OUTPUT border,
    not to the erosion). At 64^2 the default discards 252 of 4096 bins on every
    call and hands them to fill_whitespace. Spelling the four steps out is the
    only form that preserves the border.
    """
    e = ndimage.binary_erosion(m, _SE, border_value=1)
    d = ndimage.binary_dilation(e, _SE, border_value=0)
    d = ndimage.binary_dilation(d, _SE, border_value=0)
    return ndimage.binary_erosion(d, _SE, border_value=1)


def _restore_vanished(labels, out, k):
    """Give back a bin to any partition with no bins in `out` (mutated and
    returned), so opening/closing never deletes a partition outright (spec
    section 10 risk 6). Pure per-bin logic, factored out of morph_open_close
    so it can be driven from a hand-built `out` directly, without needing to
    reverse-engineer an input through the structuring element to reach a
    particular case.

    This is a deliberately SIMPLER rescue than ensure_nonempty, and always
    will be: it has no coarse density, and round-2 review accepted that it
    should not gain one. In priority order:
      (a) a bin that was originally this partition's own (`had`) AND is
          still genuinely unclaimed (out < 0) after morphology;
      (b) failing that, the raster-first `had` bin whose current owner would
          still keep >= 2 bins afterwards -- only ensure_nonempty's donor-
          count THRESHOLD is mirrored here, not its density-informed CHOICE
          of which bin to take. ensure_nonempty runs immediately after
          morph_open_close/largest_component in extract() and does the
          density-informed rescue properly; threading coarse_dens through
          this otherwise-pure per-bin function would duplicate that for no
          benefit. Cost of the simplification: in this rare case the
          restored bin is raster-first rather than densest, and the region
          shape is marginally worse before SA refines it anyway.
      (c) failing even that, leave the partition vanished. This is a
          DEFENSIVE INVARIANT WITH NO KNOWN ORGANIC INPUT: reaching it needs
          a `had` set whose every bin's current owner already has exactly 1
          bin, and neither a directed construction (_open_close_one's
          smallest nonzero survivor is 4 bins, confirmed by brute-force over
          every 4x4 mask) nor a 200000-trial randomised fuzz has ever
          produced one from real morphology (see the task-5 fix-round-2
          report). Its only coverage is
          test_restore_vanished_leaves_a_partition_vanished_when_every_donor_has_only_one_bin,
          a hand-built `out` that bypasses morphology to exercise it
          directly. ensure_nonempty, called downstream with coarse density,
          is the correct place to find such a partition a bin for real.
    """
    for kk in range(k):
        had = labels == kk
        if not had.any() or (out == kk).any():
            continue
        unclaimed = had & (out < 0)
        if unclaimed.any():
            out[unclaimed] = np.int16(kk)
            continue
        counts = np.bincount(out[out >= 0].ravel(), minlength=k)
        owned = had & (out >= 0)
        donor_ok = np.zeros(out.shape, dtype=bool)
        donor_ok[owned] = counts[out[owned]] >= 2
        if donor_ok.any():
            out[np.unravel_index(int(np.argmax(donor_ok)), out.shape)] = np.int16(kk)
    return out


def morph_open_close(labels, k):
    """Per-partition binary opening then closing (digest section 2.2). Bins
    claimed by more than one partition afterwards go to the partition with the
    most 4-neighbours in the pre-morphology map (tie -> lowest id); bins claimed
    by none become -1. Opening is allowed to shave a partition but never to
    delete it outright -- see _restore_vanished for how a vanished partition
    gets a bin back."""
    masks = np.stack([_open_close_one(labels == kk) for kk in range(k)])
    n_claim = masks.sum(axis=0)
    out = np.full(labels.shape, -1, dtype=np.int16)
    single = n_claim == 1
    if single.any():
        out[single] = masks[:, single].argmax(axis=0).astype(np.int16)
    multi = np.nonzero(n_claim > 1)
    if len(multi[0]):
        cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
        nb = np.stack([ndimage.convolve((labels == kk).astype(np.int32), cross,
                                        mode="constant", cval=0)
                       for kk in range(k)])
        cand = np.where(masks[:, multi[0], multi[1]],
                        nb[:, multi[0], multi[1]], -1)
        out[multi] = cand.argmax(axis=0).astype(np.int16)
    return _restore_vanished(labels, out, k)


def largest_component(labels, k):
    """Keep only each partition's largest 4-connected component; the rest
    becomes -1. ndimage.label numbers components in raster order and np.argmax
    returns the lowest index on ties, so equal-sized components break on the one
    whose first bin comes first in raster order."""
    out = np.full(labels.shape, -1, dtype=np.int16)
    for kk in range(k):
        m = labels == kk
        if not m.any():
            continue
        cc, n = ndimage.label(m, structure=_CC)
        sizes = np.bincount(cc.ravel(), minlength=n + 1)
        sizes[0] = 0
        out[cc == int(np.argmax(sizes))] = kk
    return out


def ensure_nonempty(labels, coarse_dens, k):
    """spec section 10 risk 6: at K=16 on 64^2 bins the argmax + largest-CC
    pipeline can leave a partition with no bins at all, which would build an
    empty RegionSpec and fail RegionSet.validate(). Give every such partition
    the single bin where its own coarse density is highest among bins whose
    current owner still has at least 2 bins. Deterministic; moves at most one
    bin per empty partition."""
    out = labels.copy()
    for kk in range(k):
        if (out == kk).any():
            continue
        counts = np.bincount(out[out >= 0].ravel(), minlength=k)
        donor_ok = np.zeros(out.shape, dtype=bool)
        owned = out >= 0
        donor_ok[owned] = counts[out[owned]] >= 2
        if not donor_ok.any():
            raise ValueError(
                f"partition {kk} vanished and no donor bin is available; "
                "reduce --extract-bins or K")
        score = np.where(donor_ok, coarse_dens[kk], -np.inf)
        out[np.unravel_index(int(np.argmax(score)), out.shape)] = kk
    return out


def fill_whitespace(labels):
    """Assign every -1 bin to the nearest labelled bin (Euclidean distance
    transform; scipy's return_indices tie-break is deterministic)."""
    unl = labels < 0
    if not unl.any():
        return labels
    if unl.all():
        raise ValueError("every bin is unlabelled; extraction produced no regions")
    _, idx = ndimage.distance_transform_edt(unl, return_indices=True)
    return labels[idx[0], idx[1]].astype(np.int16)


def canonicalise_connectivity(labels, k):
    """Make every region 4-connected (ruling D7).

    While any region has more than one component, reassign every component but
    the largest to the majority label among that component's own 4-neighbours
    (ties -> lowest label id). Neither ensure_nonempty (which takes a bin off a
    donor with only a counts >= 2 guard) nor fill_whitespace (a nearest-
    labelled-bin EDT, not a connected dilation) preserves connectivity, yet
    sa.SaState._breaks_a_region and rectify._feasible both require it: a
    violation turns SA into a silent no-op and rectification into a
    RuntimeError. Each pass strictly reduces the total component count, so the
    loop is bounded; the outer range is a belt-and-braces guard.
    """
    out = np.asarray(labels, dtype=np.int16).copy()
    for _ in range(out.size):
        changed = False
        for kk in range(k):
            m = out == kk
            if not m.any():
                continue
            cc, n = ndimage.label(m, structure=_CC)
            if n <= 1:
                continue
            sizes = np.bincount(cc.ravel(), minlength=n + 1)
            sizes[0] = 0
            keep = int(np.argmax(sizes))
            for c in range(1, n + 1):
                if c == keep:
                    continue
                sel = cc == c
                # A halo pixel here can never itself carry label kk: any
                # pixel 4-adjacent to `sel` that also equals kk would, by
                # ndimage.label's own definition, belong to the same
                # component as sel (contradicting `& ~sel`), and this loop
                # only ever moves kk-pixels to some OTHER label, never the
                # reverse -- so no `nb == kk` filter is needed here.
                nb = out[_halo(sel) & ~sel]
                if not nb.size:
                    continue
                out[sel] = np.int16(np.bincount(nb, minlength=k).argmax())
                changed = True
        if not changed:
            break
    return out


def extract(node_x, node_y, node_w, node_h, part, k, die, out_bins,
            fine_bins=FINE_BINS):
    """The whole digest section 2.2 pipeline. Returns an (out_bins, out_bins)
    int16 label grid with every bin owned by exactly one partition, so the
    result tiles the die with zero whitespace by construction, and every region
    is non-empty and 4-connected (ruling D7 -- asserted below).

    Host memory: density_maps holds k * fine_bins^2 float64 (537 MiB at K=16 /
    2048^2, 1.07 GiB at K=32) plus an equal-sized bincount output.
    """
    dens = density_maps(node_x, node_y, node_w, node_h, part, k, die,
                        bins=fine_bins)
    coarse_dens = block_sum(dens, out_bins)
    lab = majority_downsample(argmax_labels(dens), out_bins, k)
    lab = morph_open_close(lab, k)
    lab = largest_component(lab, k)
    lab = ensure_nonempty(lab, coarse_dens, k)
    lab = fill_whitespace(lab)
    lab = canonicalise_connectivity(lab, k)
    assert (lab >= 0).all()
    # Ruling D7: this is the precondition Tasks 6 and 7 rely on. Assert it here
    # rather than discovering it as a silent SA no-op or a rectify RuntimeError.
    for kk in range(k):
        m = lab == kk
        assert m.any(), "region %d is empty after extraction" % kk
        assert ndimage.label(m, structure=_CC)[1] == 1, \
            "region %d is not 4-connected after extraction" % kk
    return lab
