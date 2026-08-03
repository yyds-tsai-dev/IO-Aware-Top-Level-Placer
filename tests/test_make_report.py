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
