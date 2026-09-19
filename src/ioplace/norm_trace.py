"""`norm_trace.jsonl`: one row per coefficient transaction (>= one per probe;
design sec 4).

One file per run, rewritten on open; flushed after every row, so a run killed
mid-GP keeps everything it logged. Non-finite values are serialised as
`NaN`/`Infinity`; `json.loads` accepts them, strict JSON parsers do not -- this
is a diagnostic log, and failing a multi-hour placement over one NaN would be
worse than logging it.
"""
import json
import os

ROW_FIELDS = ("iteration", "probe_iteration", "overflow", "tau", "gamma",
              "policy", "norm_p", "grad_l1_wl", "cmax", "cap", "cap_binding",
              "cancellation_ratio", "obj_version", "refreshed_version",
              "terms")

#: `grad_l1` holds the `||.||_p` norm with `p` = the row's `norm_p`; the name is
#: kept for continuity with the retired `grad_l1_io`/`grad_l1_ft` trajectory
#: fields. `wt` is policy A's stepped weight, policy B's ramped target share,
#: or (policy "legacy") `rho * activation_ramp(iteration, it_activate, n_ramp)`
#: -- the retired path's own ramped weight, for the same slot.
TERM_FIELDS = ("grad_l1", "ratio_inst", "ratio_ema", "wt", "target_share",
               "lam", "share", "active")


class NormTraceWriter(object):
    def __init__(self, path, validate=True):
        self.path = path
        self.validate = validate
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._stream = open(path, "w")

    def write(self, row):
        if self.validate:
            _check(row, ROW_FIELDS, "row")
            for name, term in row["terms"].items():
                _check(term, TERM_FIELDS, "term %r" % (name,))
        self._stream.write(json.dumps(row) + "\n")
        self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _check(mapping, fields, what):
    got, want = set(mapping), set(fields)
    if got != want:
        raise ValueError("%s fields %r do not match the schema (missing %r, "
                         "unexpected %r)" % (what, sorted(got), sorted(want - got),
                                             sorted(got - want)))


def read_norm_trace(path):
    """Parse a `norm_trace.jsonl` into a list of row dicts."""
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]
