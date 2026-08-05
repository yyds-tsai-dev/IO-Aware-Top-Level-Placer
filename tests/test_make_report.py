import json
from ioplace.drivers.make_report import collect_results, render_markdown

def test_collect_and_render(tmp_path):
    r = {"mode": "flat", "config": "x/adaptec1.json", "k": 16, "rtype": "grid",
         "seed": 0, "io_count": 1234, "ft_count": 56, "tree_wl": 1.5e7,
         "hpwl": 1.2e7, "runtime_s": 100.0, "peak_mem_mb": 2048.0}
    (tmp_path / "a.json").write_text(json.dumps(r))
    rows = collect_results(str(tmp_path))
    assert len(rows) == 1 and rows[0]["case"] == "adaptec1"
    md = render_markdown(rows)
    assert "adaptec1" in md and "1,234" in md and "| flat |" in md

def test_num_reweights_column_present_and_blank_when_missing(tmp_path):
    """Task 13: COLS gains num_reweights so the M1 report can show reweight's
    debug count alongside flat/two_stage rows, which never set that key."""
    r_flat = {"mode": "flat", "config": "x/adaptec1.json", "k": 16, "rtype": "grid",
              "seed": 0, "io_count": 1234, "ft_count": 56, "tree_wl": 1.5e7,
              "hpwl": 1.2e7, "runtime_s": 100.0, "peak_mem_mb": 2048.0}
    r_reweight = {"mode": "reweight", "config": "x/adaptec1.json", "k": 16,
                  "rtype": "grid", "seed": 0, "io_count": 1000, "ft_count": 50,
                  "tree_wl": 1.4e7, "hpwl": 1.1e7, "runtime_s": 120.0,
                  "peak_mem_mb": 2200.0, "reweight_every": 100, "alpha": 0.5,
                  "num_reweights": 7}
    (tmp_path / "a_flat.json").write_text(json.dumps(r_flat))
    (tmp_path / "b_reweight.json").write_text(json.dumps(r_reweight))
    rows = collect_results(str(tmp_path))
    md = render_markdown(rows)
    assert "num_reweights" in md
    reweight_row = [ln for ln in md.splitlines() if "| reweight |" in ln][0]
    assert reweight_row.strip().endswith("| 7 |")
    flat_row = [ln for ln in md.splitlines() if "| flat |" in ln][0]
    # flat rows never set num_reweights -- _fmt(r.get(c, "")) renders "", so the
    # last cell is blank rather than "0" or some other placeholder.
    assert flat_row.strip().endswith("|  |")
