import json

import pytest

from ioplace.norm import TermNormalizer
from ioplace.norm_trace import (ROW_FIELDS, TERM_FIELDS, NormTraceWriter,
                                read_norm_trace)


def _row(**overrides):
    row = {"iteration": 100, "probe_iteration": 100, "overflow": 0.42,
           "tau": 53.4, "gamma": 14.3, "policy": "grandplan", "norm_p": 1,
           "grad_l1_wl": 1000.0, "cmax": 3.5, "cap": 285.7142857142857,
           "cap_binding": "io", "cancellation_ratio": 0.83, "obj_version": 3,
           "refreshed_version": 3,
           "terms": {"io": {"grad_l1": 10.0, "ratio_inst": 100.0,
                            "ratio_ema": 95.0, "wt": 0.1, "wt_max": 1.0,
                            "target_share": 0.3, "lam": 9.5,
                            "lam_applied": 4.75, "share": 0.087,
                            "kappa_clamped": False, "active": True}}}
    row.update(overrides)
    return row


def test_row_fields_match_the_design_logging_list():
    assert ROW_FIELDS == ("iteration", "probe_iteration", "overflow", "tau",
                          "gamma", "policy", "norm_p", "grad_l1_wl", "cmax",
                          "cap", "cap_binding", "cancellation_ratio",
                          "obj_version", "refreshed_version", "terms")
    assert TERM_FIELDS == ("grad_l1", "ratio_inst", "ratio_ema", "wt",
                           "wt_max", "target_share", "lam", "lam_applied",
                           "share", "kappa_clamped", "active")


def test_writer_appends_one_json_object_per_row(tmp_path):
    path = tmp_path / "sub" / "norm_trace.jsonl"
    writer = NormTraceWriter(str(path))
    writer.write(_row(iteration=0))
    writer.write(_row(iteration=50))
    writer.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["iteration"] for line in lines] == [0, 50]
    assert read_norm_trace(str(path)) == [_row(iteration=0), _row(iteration=50)]


def test_writer_flushes_so_a_killed_run_keeps_its_rows(tmp_path):
    path = tmp_path / "norm_trace.jsonl"
    writer = NormTraceWriter(str(path))
    writer.write(_row())
    assert len(path.read_text().splitlines()) == 1     # readable before close()
    writer.close()


def test_writer_rejects_an_unknown_or_missing_field(tmp_path):
    writer = NormTraceWriter(str(tmp_path / "t.jsonl"))
    with pytest.raises(ValueError):
        writer.write(_row(extra=1))
    bad = _row()
    del bad["cmax"]
    with pytest.raises(ValueError):
        writer.write(bad)
    bad = _row()
    bad["terms"]["io"]["surprise"] = 1
    with pytest.raises(ValueError):
        writer.write(bad)
    writer.close()


def test_writer_is_a_context_manager(tmp_path):
    path = tmp_path / "t.jsonl"
    with NormTraceWriter(str(path)) as writer:
        writer.write(_row())
    assert len(read_norm_trace(str(path))) == 1


def test_normalizer_emits_one_row_per_refreshed_transaction(tmp_path):
    path = tmp_path / "norm_trace.jsonl"
    with NormTraceWriter(str(path)) as writer:
        n = TermNormalizer(policy="grandplan", trace=writer)
        n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
        for iteration in (0, 50, 100):
            n.transaction(iteration, 0.85, tau=1000.0, gamma=1e-12,
                          grad_norms={"wl": 1000.0, "io": 10.0})
            n.mark_refreshed()
    rows = read_norm_trace(str(path))
    assert [r["iteration"] for r in rows] == [0, 50, 100]
    assert [r["obj_version"] for r in rows] == [1, 2, 3]
    # written at mark_refreshed(), so the row always records a live objective
    assert all(r["refreshed_version"] == r["obj_version"] for r in rows)
    assert [r["terms"]["io"]["wt"] for r in rows] == [0.05, 0.05, 0.10]


def test_write_after_close_raises(tmp_path):
    """Review M9 (deferred at progress.md:51): this used to raise
    `AttributeError: 'NoneType' object has no attribute 'write'`."""
    writer = NormTraceWriter(str(tmp_path / "t.jsonl"))
    writer.write(_row())
    writer.close()
    with pytest.raises(RuntimeError) as excinfo:
        writer.write(_row())
    assert "closed" in str(excinfo.value)
    writer.close()                     # close() stays idempotent


def test_no_row_is_written_for_a_transaction_that_is_never_refreshed(tmp_path):
    path = tmp_path / "norm_trace.jsonl"
    with NormTraceWriter(str(path)) as writer:
        n = TermNormalizer(policy="grandplan", trace=writer)
        n.register("io", object(), 1.0, activate_overflow=0.90, n_ramp=0)
        n.transaction(0, 0.85, tau=1000.0, gamma=1e-12,
                      grad_norms={"wl": 1000.0, "io": 10.0})
    assert read_norm_trace(str(path)) == []
