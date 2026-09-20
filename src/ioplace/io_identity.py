"""Verification of design v2 sec 7's accounting identity

    io(final) = io(soft, last GP) + io_delta_at_freeze + lg_loss

against *independently re-measured* IO counts, plus the extractor for spec
sec 9's F "done" criterion.

main_flow_metrics.io_accounting (P-B) defines io_delta_at_freeze and lg_loss as
differences of the same three numbers, so its io_identity_residual is
algebraically zero for any input and detects nothing (P-B pre-flight amendment
D-1). This module closes that hole from the other side: substitute IO counts
re-measured from the saved position arrays, by a freshly built evaluator, into
the identity while keeping the *recorded* deltas. The residual is then zero only
if the recorded deltas really describe the placements on disk.

io_fence_gp is not re-measurable -- the pre-legalisation fence-GP positions are
never written out, which is exactly what io_fence_gp_source exists to vouch for
-- so `measured` may carry any subset of the three counts and only what it
carries is checked. io_soft and io_count alone already constrain the identity
fully.

No torch, no numpy, no I/O: the caller does the re-measurement and hands the
numbers here.
"""

RECORDED_FIELDS = ("io_soft", "io_fence_gp", "io_count", "io_delta_at_freeze",
                   "lg_loss")
MEASURABLE_FIELDS = ("io_soft", "io_fence_gp", "io_count")

# Spec sec 9: "F: five diagnostics reported and the identity closing within +-1
# crossing." Diagnostics 1-3 are P-F's (evaluator side); 4 and 5 are P-B's
# (io_delta_at_freeze from main_flow_metrics.io_accounting, fence_compliance
# from main_flow_metrics.fence_compliance).
P_F_DIAGNOSTICS = ("straddle_cells", "straddle_area_fraction",
                   "straddle_pin_split_nets", "io_delta_at_freeze",
                   "fence_compliance")


def verify_io_identity(recorded, measured, *, tol=1):
    """`recorded`: the run's own numbers (a result.json dict works directly).
    `measured`: independently re-evaluated IO counts, any subset of
    MEASURABLE_FIELDS. Never raises on a bad *number*, only on a missing or
    unknown *name* -- a silent default here would be indistinguishable from
    the stale measurement this function exists to catch.

    Returns a dict with:
      - `residual`: io_final - (io_soft + io_delta_at_freeze + lg_loss), using
        `measured`'s io_soft/io_count where given and `recorded`'s otherwise.
      - `stale`: per-key (measured - recorded) for every key `measured` carries
        -- the disagreement that "detects nothing" (D-1) structurally cannot
        see, itemised so a failure names which recorded number is wrong.
      - `ok`: whether `residual` sits inside the +-tol window spec sec 9
        requires -- the tolerance is deliberately a window, not a free pass
        for any staleness the window happens not to move the residual by
        (pre-flight ruling R-4: the plan's own reading here, `ok = |residual|
        <= tol and not any(stale)`, made `tol` a no-op since any nonzero
        stale value already fails `ok` regardless of the window). **Read
        `ok` alone to ask "does the recorded identity close within tolerance
        against these re-measured counts."**
      - `ok_strict`: `ok and not any(stale.values())` -- the stricter reading
        the plan intended, for a caller that additionally wants every
        re-measured term to match its recorded value exactly. **Read
        `ok_strict` to ask "does it close AND does every re-measured term
        match exactly."** Reaching for `ok` when you meant `ok_strict` (or
        vice versa) is the obvious failure mode this docstring exists to
        head off.
      - `tol`, `recorded`, `measured`: echoed back for logging/provenance.
    """
    for name in RECORDED_FIELDS:
        if name not in recorded:
            raise KeyError("recorded is missing %r; verify_io_identity needs "
                           "every one of %r" % (name, RECORDED_FIELDS))
    unknown = [name for name in measured if name not in MEASURABLE_FIELDS]
    if unknown:
        raise KeyError("measured carries unknown key(s) %r; only %r can be "
                       "re-measured from saved artefacts"
                       % (sorted(unknown), MEASURABLE_FIELDS))

    io_soft = int(measured.get("io_soft", recorded["io_soft"]))
    io_final = int(measured.get("io_count", recorded["io_count"]))
    residual = io_final - (io_soft + int(recorded["io_delta_at_freeze"])
                           + int(recorded["lg_loss"]))
    stale = {name: int(measured[name]) - int(recorded[name]) for name in measured}
    ok = abs(residual) <= int(tol)
    return {"residual": residual, "stale": stale, "tol": int(tol),
            "ok": ok, "ok_strict": ok and not any(stale.values()),
            "recorded": {name: recorded[name] for name in RECORDED_FIELDS},
            "measured": {name: int(value) for name, value in measured.items()}}


def p_f_diagnostics(result):
    """The five diagnostics spec sec 9 requires P-F to report, pulled out of a
    result.json dict in P_F_DIAGNOSTICS order. `fence_compliance` prefers
    `fence_compliance_center` -- the centre anchor is the one sec 3 phase 2 and
    sec 7 both use -- and falls back to the legacy lower-left key that
    run_placement_two_stage.py:252-255 established. A missing diagnostic raises
    rather than defaulting: "not reported" is the failure this checks for.
    """
    out = {}
    for name in P_F_DIAGNOSTICS:
        if name == "fence_compliance":
            for key in ("fence_compliance_center", "fence_compliance"):
                if result.get(key) is not None:
                    out[name] = result[key]
                    break
            else:
                raise KeyError("result reports neither fence_compliance_center "
                               "nor fence_compliance")
        elif name not in result:
            raise KeyError("result is missing diagnostic %r" % (name,))
        else:
            out[name] = result[name]
    return out
