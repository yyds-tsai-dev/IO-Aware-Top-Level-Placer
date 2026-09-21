"""Per-segment boundary capacity, extracted once before global placement.

Two sources (v2 design sec 5):

  openroad   one `global_route` on the *input* DEF with minimal congestion
             iterations and -allow_congestion, reusing
             `route_eval/or_scripts/dump_online_route.py:90-114` and
             `route_eval/online_openroad.py:44-159` verbatim. Only
             horizontal_capacity / vertical_capacity and the GCell edges are
             kept; `usage` is discarded, because it reflects the input
             placement and this run exists only to read resources.
  lef_pitch  the fallback C_seg = rho*l, rho = sum over the layers
             perpendicular to the segment of 1/pitch, from the tech LEF.

The unit rule (round-feedback spec:169-196) lives here: this module is the
only place in P-D that touches the GCell grid, it runs once, offline, and its
whole output is one scalar per segment id. Zero-capacity segments are written
out as zero and stay blocked downstream; no epsilon is substituted.
"""
import argparse
import json
import os
import re
import tempfile

import numpy as np

from ioplace.artifacts import _restore_umask_permissions
from ioplace.region_segments import ORIENT_V, SEGMENT_SCHEMA_VERSION, segments_digest

CAPACITY_SCHEMA_VERSION = 1

# Spec sec 5, verbatim. Do not reword: downstream readers compare the string.
CAPACITY_SEMANTICS = ("usable tracks crossing the segment; "
                      "one net crossing consumes one track")

_SEGMENT_ARRAYS = ("orient", "line", "lo", "hi", "pair_a", "pair_b",
                   "length_units", "length")


# ---------------------------------------------------------------- tech LEF --
def parse_tech_lef_layers(path):
    """Routing layers of a tech LEF, in file order: name, PITCH (microns),
    DIRECTION. Only `TYPE ROUTING` layers are returned; MASTERSLICE, CUT and
    OVERLAP layers are skipped."""
    layers = []
    current = None
    with open(path) as stream:
        for raw in stream:
            line = raw.strip()
            match = re.match(r"^LAYER\s+(\S+)", line)
            if match is not None:
                current = {"name": match.group(1)}
                continue
            if current is None:
                continue
            if line.startswith("TYPE"):
                current["type"] = line.split()[1].rstrip(";").strip()
            elif line.startswith("PITCH"):
                current["pitch"] = float(line.split()[1])
            elif line.startswith("DIRECTION"):
                current["direction"] = line.split()[1].rstrip(";").strip()
            elif line.startswith("END "):
                if current.get("type") == "ROUTING":
                    if "pitch" not in current or "direction" not in current:
                        raise ValueError(
                            "routing layer %r has no PITCH/DIRECTION" % (current["name"],))
                    current["index"] = len(layers)
                    layers.append(current)
                current = None
    if not layers:
        raise ValueError("no TYPE ROUTING layers in %s" % (path,))
    return layers


def track_density(layers, direction, layer_range=None):
    """Tracks per micron available in `direction`, summed over the routing
    layers whose preferred DIRECTION matches. `layer_range` is an inclusive
    (first, last) layer-name pair -- pass ("metal2", "metal10") to match the
    sec 8 GRT protocol's `set_routing_layers metal2-metal10`."""
    selected = layers
    if layer_range is not None:
        names = [layer["name"] for layer in layers]
        try:
            lo, hi = names.index(layer_range[0]), names.index(layer_range[1])
        except ValueError:
            raise ValueError("layer range %r not in this LEF (%r)"
                             % (layer_range, names))
        if lo > hi:
            raise ValueError("layer range %r is inverted" % (layer_range,))
        selected = layers[lo:hi + 1]
    return float(sum(1.0 / layer["pitch"] for layer in selected
                     if layer["direction"] == direction))


def lef_capacity(table, layers, *, scale_factor, dbu_per_micron,
                 layer_range=("metal2", "metal10")):
    """Fallback C_seg = rho * l. A vertical segment is crossed by wires running
    horizontally, so its rho is the HORIZONTAL track density, and vice versa
    (sec 5: "rho = sum over layers perpendicular to seg of 1/pitch").

    `table.length` is in the RegionGrid's own coordinate frame, which is DEF
    DBU scaled by `scale_factor` (`route_eval/online_openroad.load_observation`
    uses the same (dbu - shift) * scale convention), so microns are
    length / (scale_factor * dbu_per_micron)."""
    if scale_factor <= 0 or dbu_per_micron <= 0:
        raise ValueError("positive scale_factor and dbu_per_micron required")
    rho_h = track_density(layers, "HORIZONTAL", layer_range)
    rho_v = track_density(layers, "VERTICAL", layer_range)
    microns = table.length / (float(scale_factor) * float(dbu_per_micron))
    rho = np.where(table.orient == ORIENT_V, rho_h, rho_v)
    return (rho * microns).astype(np.float64)


# ------------------------------------------------------------- GCell grid --
def gcell_capacity(table, resources, *, shift_factor=(0.0, 0.0),
                   scale_factor=1.0, chunk=1_000_000):
    """Convert one `dump_online_route.py` resources payload into a per-segment
    capacity (sec 5's overlap-weighted row/column sum). `usage` is ignored by
    construction -- it is never read here."""
    shift = np.asarray(shift_factor, dtype=np.float64).reshape(2)
    scale = float(scale_factor)
    x_edges = (np.asarray(resources["x_edges_dbu"], dtype=np.float64) - shift[0]) * scale
    y_edges = (np.asarray(resources["y_edges_dbu"], dtype=np.float64) - shift[1]) * scale
    hcap = np.asarray(resources["horizontal_capacity"], dtype=np.float64)
    vcap = np.asarray(resources["vertical_capacity"], dtype=np.float64)
    nx, ny = x_edges.size - 1, y_edges.size - 1
    if nx < 2 or ny < 2:
        raise ValueError("invalid OpenDB resource dimensions: need >= 2 GCells "
                         "per axis, got %dx%d" % (nx, ny))
    if hcap.shape != (ny, nx - 1) or vcap.shape != (ny - 1, nx):
        raise ValueError("invalid OpenDB resource dimensions: expected hcap "
                         "%r / vcap %r, got %r / %r"
                         % ((ny, nx - 1), (ny - 1, nx), hcap.shape, vcap.shape))
    if not (np.isfinite(hcap).all() and np.isfinite(vcap).all()):
        raise ValueError("nonfinite OpenDB capacity")
    row_lo, row_hi = y_edges[:-1], y_edges[1:]
    col_lo, col_hi = x_edges[:-1], x_edges[1:]
    out = np.zeros(table.num_segments, dtype=np.float64)

    v_idx = np.nonzero(table.orient == ORIENT_V)[0]
    step = max(1, chunk // max(ny, 1))
    for start in range(0, v_idx.size, step):
        sub = v_idx[start:start + step]
        x = table.box[sub, 0]
        y0, y1 = table.box[sub, 1], table.box[sub, 3]
        col = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, nx - 2)
        overlap = np.clip(np.minimum(row_hi[None, :], y1[:, None])
                          - np.maximum(row_lo[None, :], y0[:, None]), 0.0, None)
        out[sub] = (overlap / (row_hi - row_lo)[None, :] * hcap[:, col].T).sum(axis=1)

    h_idx = np.nonzero(table.orient != ORIENT_V)[0]
    step = max(1, chunk // max(nx, 1))
    for start in range(0, h_idx.size, step):
        sub = h_idx[start:start + step]
        y = table.box[sub, 1]
        x0, x1 = table.box[sub, 0], table.box[sub, 2]
        row = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, ny - 2)
        overlap = np.clip(np.minimum(col_hi[None, :], x1[:, None])
                          - np.maximum(col_lo[None, :], x0[:, None]), 0.0, None)
        out[sub] = (overlap / (col_hi - col_lo)[None, :] * vcap[row, :]).sum(axis=1)

    return out


def run_extraction(def_path, lefs, out_dir, binary, *, congestion_iterations=5,
                   signal_layers="metal2-metal10", threads=4):
    """One OpenROAD process on the input DEF. `-allow_congestion` is mandatory
    and the iteration count is deliberately small: unbounded congestion removal
    cost ~9-10 h on tile/group, while 5 iterations routed tile in 221.9 s
    (`docs/results/2026-09-15-benchmark-router-diagnosis.md:5-18,31-43`). We are
    reading resources, not producing a route, so a congested result is fine.

    Returns (resources_payload, sha256 of the receipt) -- the receipt is the
    provenance record save_capacity stores."""
    from ioplace.route_eval.online_openroad import digest, run_openroad
    out_dir = str(out_dir)
    run_openroad(str(def_path), [str(p) for p in lefs], out_dir, str(binary),
                 congestion_iterations=int(congestion_iterations),
                 allow_congestion=True, threads=int(threads),
                 signal_layers=signal_layers)
    with open(os.path.join(out_dir, "resources.json")) as stream:
        resources = json.load(stream)
    return resources, digest(os.path.join(out_dir, "receipt.json"))


# ------------------------------------------------------------------- I/O ---
def save_capacity(path, table, capacity, *, source, receipt_sha256=None,
                  extra=None):
    """Write capacity.npz: the segment table's defining arrays (so the file is
    self-describing), the per-segment capacity, and the metadata sec 5 asks
    for -- capacity_source, the OpenROAD receipt SHA-256, and the verbatim
    semantics string."""
    if source not in ("openroad", "lef_pitch"):
        raise ValueError("capacity_source must be 'openroad' or 'lef_pitch', "
                         "got %r" % (source,))
    capacity = np.asarray(capacity, dtype=np.float64)
    if capacity.shape != (table.num_segments,):
        raise ValueError("capacity must carry one value per segment (%d), got %r"
                         % (table.num_segments, capacity.shape))
    if not np.isfinite(capacity).all() or (capacity < 0).any():
        raise ValueError("finite nonnegative capacity required")
    metadata = dict(
        schema_version=CAPACITY_SCHEMA_VERSION,
        segment_schema_version=int(SEGMENT_SCHEMA_VERSION),
        capacity_source=source,
        capacity_semantics=CAPACITY_SEMANTICS,
        receipt_sha256=receipt_sha256,
        segments_sha256=segments_digest(table),
        num_segments=int(table.num_segments),
        zero_capacity_segments=int((capacity == 0.0).sum()),
        k=int(table.k), lattice=int(table.lattice), die=list(table.die),
    )
    metadata.update(extra or {})
    arrays = {"seg_" + name: np.ascontiguousarray(getattr(table, name))
              for name in _SEGMENT_ARRAYS}
    arrays["capacity"] = capacity
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    parent = os.path.dirname(os.path.abspath(str(path)))
    os.makedirs(parent, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".capacity-", suffix=".npz",
                                         dir=parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, str(path))
        # mkstemp() creates the temp file at 0600 and os.replace() preserves
        # that mode across the rename, so without this every capacity.npz
        # would come out unreadable by anyone but its writer regardless of
        # the process umask (artifacts._restore_umask_permissions's own
        # docstring names exactly this hazard).
        _restore_umask_permissions(str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def load_capacity(path, table=None):
    """Read capacity.npz. With `table`, verify the segment fingerprint -- a
    capacity file paired with a different geometry is the single most damaging
    silent failure in P-D, so it is a hard error."""
    with np.load(str(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data.pop("metadata")))
    if metadata.get("schema_version") != CAPACITY_SCHEMA_VERSION:
        raise ValueError("unsupported capacity schema %r"
                         % (metadata.get("schema_version"),))
    if metadata.get("capacity_semantics") != CAPACITY_SEMANTICS:
        raise ValueError("capacity_semantics does not match this build")
    capacity = np.asarray(data["capacity"], dtype=np.float64)
    if capacity.shape != (metadata["num_segments"],):
        raise ValueError("capacity array does not match num_segments")
    if table is not None and segments_digest(table) != metadata["segments_sha256"]:
        raise ValueError("capacity.npz was built for a different segment table")
    data["capacity"] = capacity
    data["metadata"] = metadata
    return data


def extract_capacity(table, *, source="auto", def_path=None, lefs=(),
                     openroad_bin=None, or_out=None, tech_lef=None,
                     dbu_per_micron=2000.0, layer_range=("metal2", "metal10"),
                     shift_factor=(0.0, 0.0), scale_factor=1.0,
                     congestion_iterations=5, signal_layers="metal2-metal10",
                     threads=4):
    """Dispatch. `auto` prefers OpenROAD and records why it fell back."""
    if source not in ("auto", "openroad", "lef_pitch"):
        raise ValueError("source must be auto|openroad|lef_pitch")
    can_route = bool(def_path and lefs and openroad_bin and or_out)
    if source in ("auto", "openroad") and can_route:
        try:
            resources, receipt = run_extraction(
                def_path, lefs, or_out, openroad_bin,
                congestion_iterations=congestion_iterations,
                signal_layers=signal_layers, threads=threads)
            capacity = gcell_capacity(table, resources,
                                      shift_factor=shift_factor,
                                      scale_factor=scale_factor)
            extra = {"uint8_backend": bool(resources.get("uint8_backend", False)),
                     "opendb_capacity_semantics": resources.get("capacity_semantics"),
                     "congestion_iterations": int(congestion_iterations),
                     "signal_layers": signal_layers}
            return capacity, "openroad", receipt, extra
        except Exception as error:
            if source == "openroad":
                raise
            fallback_reason = "openroad extraction failed: %s" % (error,)
    elif source == "openroad":
        raise ValueError("source='openroad' needs def_path, lefs, openroad_bin "
                         "and or_out")
    else:
        fallback_reason = "no OpenROAD inputs supplied"
    if tech_lef is None:
        raise ValueError("the lef_pitch fallback needs --tech-lef")
    layers = parse_tech_lef_layers(tech_lef)
    capacity = lef_capacity(table, layers, scale_factor=scale_factor,
                            dbu_per_micron=dbu_per_micron,
                            layer_range=layer_range)
    extra = {"fallback_reason": fallback_reason,
             "tech_lef": os.path.abspath(tech_lef),
             "dbu_per_micron": float(dbu_per_micron),
             "layer_range": list(layer_range),
             "rho_h": track_density(layers, "HORIZONTAL", layer_range),
             "rho_v": track_density(layers, "VERTICAL", layer_range)}
    return capacity, "lef_pitch", None, extra


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m ioplace.capacity.extract",
        description="Extract per-segment boundary IO capacity (v2 design sec 5)")
    parser.add_argument("--regions", required=True, help="regions.json")
    parser.add_argument("--out", required=True, help="capacity.npz to write")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "openroad", "lef_pitch"])
    parser.add_argument("--def", dest="def_path", default=None)
    parser.add_argument("--lef", action="append", default=[])
    parser.add_argument("--openroad-bin", default=os.environ.get("OPENROAD_BIN"))
    parser.add_argument("--or-out", default=None,
                        help="fresh directory for the OpenROAD run (must not exist)")
    parser.add_argument("--congestion-iterations", type=int, default=5)
    parser.add_argument("--signal-layers", default="metal2-metal10")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tech-lef", default=None)
    parser.add_argument("--dbu-per-micron", type=float, default=2000.)
    parser.add_argument("--layer-range", default="metal2-metal10")
    parser.add_argument("--shift-factor", default="0,0")
    parser.add_argument("--scale-factor", type=float, default=1.)
    return parser


def main(argv=None):
    from ioplace.region_grid import RegionGrid
    from ioplace.region_segments import enumerate_segments
    from ioplace.regions import RegionSet
    args = build_parser().parse_args(argv)
    region_set = RegionSet.from_json(args.regions)
    region_set.validate()
    table = enumerate_segments(RegionGrid(region_set))
    shift = tuple(float(v) for v in args.shift_factor.split(","))
    first, _, last = args.layer_range.partition("-")
    capacity, source, receipt, extra = extract_capacity(
        table, source=args.source, def_path=args.def_path, lefs=args.lef,
        openroad_bin=args.openroad_bin, or_out=args.or_out,
        tech_lef=args.tech_lef, dbu_per_micron=args.dbu_per_micron,
        layer_range=(first, last), shift_factor=shift,
        scale_factor=args.scale_factor,
        congestion_iterations=args.congestion_iterations,
        signal_layers=args.signal_layers, threads=args.threads)
    metadata = save_capacity(args.out, table, capacity, source=source,
                             receipt_sha256=receipt, extra=extra)
    print(json.dumps(metadata, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
