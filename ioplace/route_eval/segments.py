"""Stage 2 S2 (`docs/superpowers/specs/2026-08-13-stage2-innovus-calibration-
plan.md` sec 7.1 / sec 10 S2 row): `.venv312`-side loader for the
`.npz`/`.json` pair written by `ioplace/route_eval/or_scripts/dump_segments.py`
(which runs under `openroad -python`, a different Python than this module).

This module is deliberately torch-free -- it only needs `numpy`/`json`/
stdlib -- so it can be imported from a plain `python3 -m pytest` run without
pulling in DREAMPlace. Keep the KIND_* constants and row layout below in
sync with `dump_segments.py`'s copy (the two files intentionally don't share
an import so each stays self-contained in its own Python environment).
"""
from dataclasses import dataclass
import json

import numpy as np

# --- row `kind` values; keep in sync with or_scripts/dump_segments.py ---
KIND_WIRE = 0
KIND_VIA = 1
KIND_RECT = 2
KIND_VWIRE = 3
KIND_SHORT = 4
KIND_NAMES = {KIND_WIRE: "WIRE", KIND_VIA: "VIA", KIND_RECT: "RECT",
              KIND_VWIRE: "VWIRE", KIND_SHORT: "SHORT"}


@dataclass
class Segments:
    """One design's routed-wire dump, column-oriented (numpy arrays, all
    length == number of segment rows except the four `*_names`/`net_has_wire`
    tables, which are indexed by net_id/layer id/via id respectively).
    """
    seg_net_id: np.ndarray    # (S,) int32
    seg_kind: np.ndarray      # (S,) uint8, one of KIND_*
    seg_layer: np.ndarray     # (S,) int16, index into layer_names
    seg_x0: np.ndarray        # (S,) int64, DBU
    seg_y0: np.ndarray        # (S,) int64, DBU
    seg_x1: np.ndarray        # (S,) int64, DBU
    seg_y1: np.ndarray        # (S,) int64, DBU
    seg_width: np.ndarray     # (S,) int64, DBU (WIRE only, 0 otherwise)
    seg_via_id: np.ndarray    # (S,) int32, index into via_names (VIA only, -1 otherwise)
    net_names: np.ndarray     # (N,) <U..., net_names[net_id] = name
    net_has_wire: np.ndarray  # (N,) bool
    layer_names: np.ndarray   # (L,) <U...
    via_names: np.ndarray     # (V,) <U...
    meta: dict                # the JSON sidecar, verbatim

    @property
    def num_segment_rows(self):
        return len(self.seg_net_id)

    @property
    def num_nets(self):
        return len(self.net_names)

    def wire_mask(self):
        return self.seg_kind == KIND_WIRE

    def segment_length(self):
        """Manhattan length of every row (|dx|+|dy|). Exact for WIRE rows
        (axis-aligned by construction of standard-cell routing); 0 for
        VIA/VWIRE/SHORT (single point) and for RECT (patch, spec sec 7.4
        says to ignore it, so its "length" is never summed by wire_length()
        below -- it's exposed here only for callers that want to audit
        patch sizes directly).
        """
        return np.abs(self.seg_x1 - self.seg_x0) + np.abs(self.seg_y1 - self.seg_y0)

    def non_manhattan_wire_count(self):
        """Count of WIRE rows that are neither horizontal nor vertical --
        should be 0 for standard-cell routing; a nonzero count means
        `segment_length()`'s Manhattan-distance assumption is wrong for
        those rows and `wire_length()` is a (mild) underestimate for them.
        """
        m = self.wire_mask()
        return int(np.count_nonzero((self.seg_x0[m] != self.seg_x1[m]) &
                                     (self.seg_y0[m] != self.seg_y1[m])))

    def wire_length(self, net_id=None):
        """Total WIRE-row length (DBU), spec sec 7.4's `route_wl`: RECT
        patches and VWIRE/SHORT markers excluded, matching
        `EvalResult.tree_wl`'s convention of only counting real wire.
        """
        m = self.wire_mask()
        if net_id is not None:
            m = m & (self.seg_net_id == net_id)
        return int(self.segment_length()[m].sum())

    def per_net_wire_length(self):
        """(N,) int64 array, wire_length() broken out per net_id."""
        out = np.zeros(self.num_nets, dtype=np.int64)
        m = self.wire_mask()
        np.add.at(out, self.seg_net_id[m], self.segment_length()[m])
        return out

    def net_index(self, name):
        """Return the net_id for `name`, or None if not present. Sec 7.2's
        key-alignment contract: lookup is by name, never by presumed index
        equality with another net_index space (e.g. `placedb.net_names`).
        """
        hits = np.nonzero(self.net_names == name)[0]
        return int(hits[0]) if len(hits) else None

    def rows_for_net(self, net_id):
        """All segment rows for one net_id, as a list of
        (kind_name, layer_name, x0, y0, x1, y1, width, via_name) tuples --
        convenient for the sampling cross-check against
        `def_text_parser.parse_def_routed()`'s per-net segment lists."""
        idx = np.nonzero(self.seg_net_id == net_id)[0]
        out = []
        for i in idx:
            k = int(self.seg_kind[i])
            layer = self.layer_names[self.seg_layer[i]] if self.seg_layer[i] >= 0 else ""
            via = self.via_names[self.seg_via_id[i]] if self.seg_via_id[i] >= 0 else ""
            out.append((KIND_NAMES[k], str(layer),
                        int(self.seg_x0[i]), int(self.seg_y0[i]),
                        int(self.seg_x1[i]), int(self.seg_y1[i]),
                        int(self.seg_width[i]), str(via)))
        return out


def rect_reconciliation_dbu(segments):
    """Sigma(long_side - short_side) over every KIND_RECT row -- the term
    that reconciles `Segments.wire_length()` (route_wl, RECT excluded, spec
    sec 7.4's "RECT patch 忽略") against `dbWire::getLength()`'s own total,
    which counts each RECT patch as its long side (see
    or_scripts/verify_routed_def.py's check 1). Pure numpy, no odb needed;
    RECT rows never enter `wire_length()` itself.
    """
    m = segments.seg_kind == KIND_RECT
    dx = np.abs(segments.seg_x1[m] - segments.seg_x0[m])
    dy = np.abs(segments.seg_y1[m] - segments.seg_y0[m])
    return int(np.sum(np.maximum(dx, dy) - np.minimum(dx, dy)))


def load_segments(npz_path, json_path=None):
    """Load a `dump_segments.py` output pair. `json_path` defaults to
    `npz_path` with its extension swapped to `.json` (dump_segments.py's own
    default `--json-out` naming).
    """
    with np.load(npz_path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}

    json_path = json_path or (str(npz_path).rsplit(".", 1)[0] + ".json")
    with open(json_path) as f:
        meta = json.load(f)

    return Segments(
        seg_net_id=arrays["seg_net_id"], seg_kind=arrays["seg_kind"],
        seg_layer=arrays["seg_layer"],
        seg_x0=arrays["seg_x0"], seg_y0=arrays["seg_y0"],
        seg_x1=arrays["seg_x1"], seg_y1=arrays["seg_y1"],
        seg_width=arrays["seg_width"], seg_via_id=arrays["seg_via_id"],
        net_names=arrays["net_names"], net_has_wire=arrays["net_has_wire"],
        layer_names=arrays["layer_names"], via_names=arrays["via_names"],
        meta=meta,
    )
