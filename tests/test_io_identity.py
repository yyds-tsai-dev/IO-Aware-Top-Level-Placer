import json
import os

import pytest

from ioplace.io_identity import (P_F_DIAGNOSTICS, RECORDED_FIELDS,
                                 p_f_diagnostics, verify_io_identity)


def _recorded(io_soft=1000, io_fence_gp=1120, io_count=1155):
    return {"io_soft": io_soft, "io_fence_gp": io_fence_gp, "io_count": io_count,
            "io_delta_at_freeze": io_fence_gp - io_soft,
            "lg_loss": io_count - io_fence_gp}


def test_field_name_contracts():
    assert RECORDED_FIELDS == ("io_soft", "io_fence_gp", "io_count",
                               "io_delta_at_freeze", "lg_loss")
    assert P_F_DIAGNOSTICS == ("straddle_cells", "straddle_area_fraction",
                               "straddle_pin_split_nets", "io_delta_at_freeze",
                               "fence_compliance")


def test_identity_closes_when_the_re_measurement_agrees():
    out = verify_io_identity(_recorded(), {"io_soft": 1000, "io_count": 1155})
    assert out["residual"] == 0
    assert out["stale"] == {"io_soft": 0, "io_count": 0}
    assert out["ok"] is True and out["tol"] == 1


def test_a_stale_recorded_io_soft_breaks_the_identity():
    """The failure mode main_flow_metrics.io_accounting structurally cannot see:
    result.json's io_soft was taken at a different iteration than soft.npz."""
    out = verify_io_identity(_recorded(), {"io_soft": 1040, "io_count": 1155})
    assert out["residual"] == -40
    assert out["stale"]["io_soft"] == 40
    assert out["ok"] is False


def test_a_stale_recorded_io_count_breaks_the_identity():
    out = verify_io_identity(_recorded(), {"io_soft": 1000, "io_count": 1160})
    assert out["residual"] == 5
    assert out["stale"]["io_count"] == 5
    assert out["ok"] is False


def test_the_tolerance_is_a_window_not_a_free_pass():
    """Spec sec 9 allows the identity to close within +-1 crossing."""
    assert verify_io_identity(_recorded(), {"io_soft": 1000, "io_count": 1156})["ok"] is True
    assert verify_io_identity(_recorded(), {"io_soft": 1000, "io_count": 1157})["ok"] is False
    assert verify_io_identity(_recorded(), {"io_soft": 1000, "io_count": 1157},
                              tol=2)["ok"] is True


def test_io_fence_gp_may_be_absent_from_the_re_measurement():
    out = verify_io_identity(_recorded(), {"io_count": 1155, "io_soft": 1000})
    assert "io_fence_gp" not in out["stale"]
    out = verify_io_identity(_recorded(), {"io_soft": 1000, "io_fence_gp": 1120,
                                           "io_count": 1155})
    assert out["stale"]["io_fence_gp"] == 0


def test_a_missing_recorded_field_is_an_error_not_a_default():
    broken = _recorded()
    del broken["lg_loss"]
    with pytest.raises(KeyError, match="lg_loss"):
        verify_io_identity(broken, {"io_soft": 1000, "io_count": 1155})


def test_an_unknown_measured_key_is_rejected():
    with pytest.raises(KeyError, match="io_total"):
        verify_io_identity(_recorded(), {"io_total": 1155})


def test_p_f_diagnostics_extracts_the_five_and_prefers_the_centre_compliance():
    result = {"straddle_cells": 12, "straddle_area_fraction": 0.004,
              "straddle_pin_split_nets": 3, "io_delta_at_freeze": 120,
              "fence_compliance": 0.98, "fence_compliance_center": 1.0}
    got = p_f_diagnostics(result)
    assert tuple(got) == P_F_DIAGNOSTICS
    assert got["fence_compliance"] == 1.0      # the centre anchor, per sec 3/sec 7


def test_p_f_diagnostics_falls_back_to_the_legacy_compliance_key():
    result = {"straddle_cells": 0, "straddle_area_fraction": 0.0,
              "straddle_pin_split_nets": 0, "io_delta_at_freeze": 0,
              "fence_compliance": 0.97}
    assert p_f_diagnostics(result)["fence_compliance"] == 0.97


def test_p_f_diagnostics_names_the_missing_diagnostic():
    with pytest.raises(KeyError, match="straddle_pin_split_nets"):
        p_f_diagnostics({"straddle_cells": 0, "straddle_area_fraction": 0.0,
                         "io_delta_at_freeze": 0, "fence_compliance": 1.0})
    with pytest.raises(KeyError, match="fence_compliance"):
        p_f_diagnostics({"straddle_cells": 0, "straddle_area_fraction": 0.0,
                         "straddle_pin_split_nets": 0, "io_delta_at_freeze": 0})


@pytest.mark.slow
@pytest.mark.gpu
def test_identity_closes_on_a_real_gcd_main_flow_run(tmp_path):
    """Spec sec 9's F exit criterion, end to end: run the main flow on GCD,
    re-evaluate soft.npz and placement.npz from disk with a fresh evaluator, and
    require the identity to close within +-1 crossing.

    R-5 (pre-flight ruling): the plan's own version of this test called
    run_main_flow(..., init="region_center", ...) without --membership, which
    run_main_flow's own validation rejects (init_pos.apply_init raises before
    any placement runs) -- the test never reached the identity assertion it
    exists to check. Fixed by switching to init="die_center": which init mode
    the soft phase starts from is incidental to what this test checks (the
    accounting identity closing), and die_center is the only mode that needs
    no extra membership artefact, so it is the smallest change that lets the
    test reach its assertion.
    """
    pytest.importorskip("torch")
    run_main_flow = pytest.importorskip(
        "ioplace.drivers.run_main_flow",
        reason="P-B has not landed; the identity's end-to-end gate is blocked "
               "on run_main_flow.py -- the arithmetic above still runs").run_main_flow
    from ioplace.artifacts import load_positions
    from ioplace.evaluator_gpu import GpuEvalContext
    from ioplace.io_identity import verify_io_identity
    from ioplace.netlist import load_netlist
    from ioplace.region_grid import RegionGrid
    from ioplace.regions import RegionSet
    from ioplace.artifacts import scaled_region_set

    config = os.path.abspath("results/route_feedback_20260914/gcd.json")
    out_dir = str(tmp_path / "gcd")
    result = run_main_flow(config, out_dir, k=4, rtype="grid", seed=0,
                           init="die_center", node_anchor="center",
                           dp_seed=1000, deterministic=1)

    nl, placedb, params = load_netlist(config)
    rs_native = RegionSet.from_json(os.path.join(out_dir, "regions.json"))
    rg = RegionGrid(scaled_region_set(rs_native, params.shift_factor,
                                      params.scale_factor))
    ctx = GpuEvalContext(nl, rg, device="cuda")

    def _io(npz_name):
        p = load_positions(os.path.join(out_dir, npz_name))
        sx = (p.node_x - p.shift_factor[0]) * p.scale_factor
        sy = (p.node_y - p.shift_factor[1]) * p.scale_factor
        return int(ctx.evaluate(sx, sy).io_count)

    check = verify_io_identity(result,
                               {"io_soft": _io("soft.npz"),
                                "io_count": _io("placement.npz")}, tol=1)
    assert check["ok"], check
    five = p_f_diagnostics(result)
    assert all(value is not None for value in five.values()), five
    with open(os.path.join(out_dir, "result.json")) as handle:
        assert p_f_diagnostics(json.load(handle)) == five
