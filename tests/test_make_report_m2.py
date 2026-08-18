import json
from ioplace.drivers.make_report import build_m2_table, M2_TABLE_SPEC

def _io_row(**overrides):
    """A representative mode="io" run row, matching the schema actually
    written by run_placement_io.run_io (results/m2/sweep/*.json,
    results/m2/noise/*.json) -- including the fields flat baselines carry too,
    since T6/T7 generate their flat baselines via rho_max=0 observer-mode
    `io` runs, not run_flat."""
    r = {"mode": "io", "config": "x/adaptec1.json", "k": 16, "rtype": "grid",
         "seed": 0, "dp_seed": 1000, "det": 1, "io_count": 24647, "io_gp": 23554,
         "ft_count": 3525, "hard_lambda_sum": 19337, "tree_wl": 85635781.0,
         "hpwl": 74895086.0, "lg_loss": 1093, "runtime_s": 84.1,
         "peak_mem_mb": 2032.3, "rho_max": 0.4, "tau_hi": 0.3, "tau_lo": 0.03,
         "alpha_io": 0.0, "w_mode": "unit", "d_max": 100, "rho_margin": 0.0,
         "margin_m": 24.0, "lambda_io_final": 340.7, "spearman_rho": 0.516}
    r.update(overrides)
    return r

def test_header_matches_design_sec_8_2_column_order(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_io_row()))
    md = build_m2_table([str(p)], {})
    header = md.splitlines()[0]
    for label, _ in M2_TABLE_SPEC:
        assert label in header
    # exact order, not just presence
    assert header == "| " + " | ".join(h for h, _ in M2_TABLE_SPEC) + " |"

def test_case_derived_from_config_basename(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_io_row(config="x/y/bigblue4.json")))
    md = build_m2_table([str(p)], {})
    row = md.splitlines()[2]
    assert row.startswith("| bigblue4 |")

def test_margin_column_always_dash_no_such_key_in_schema(tmp_path):
    """No run schema ever writes a field literally named "margin" (only
    rho_margin/margin_m) -- build_m2_table must not guess a mapping for it."""
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_io_row()))
    md = build_m2_table([str(p)], {})
    cells = [c.strip() for c in md.splitlines()[2].split("|")]
    margin_idx = [h for h, _ in M2_TABLE_SPEC].index("margin") + 1  # +1: leading ""
    assert cells[margin_idx] == "—"

def test_flat_row_missing_io_fields_renders_dash(tmp_path):
    """run_flat's schema has no io_gp/hard_lambda_sum/rho_max/... -- those
    cells must render "—", not raise or show a placeholder like 0."""
    flat_row = {"mode": "flat", "config": "x/adaptec1.json", "k": 16,
                "rtype": "grid", "seed": 0, "dp_seed": 1000, "det": 1,
                "io_count": 29992, "ft_count": 4000, "tree_wl": 9e7,
                "hpwl": 7.9e7, "runtime_s": 60.0, "peak_mem_mb": 1800.0}
    p = tmp_path / "flat.json"
    p.write_text(json.dumps(flat_row))
    md = build_m2_table([str(p)], {})
    cells = [c.strip() for c in md.splitlines()[2].split("|")]
    for label in ("io_gp", "hard_lambda_sum", "rho_max", "tau_hi", "tau_lo",
                  "alpha_io", "w_mode", "d_max", "lambda_io_final", "spearman_rho"):
        idx = [h for h, _ in M2_TABLE_SPEC].index(label) + 1
        assert cells[idx] == "—", f"{label} should be — , got {cells[idx]!r}"

def test_delta_pct_computed_against_matching_flat_baseline(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_io_row(io_count=9500, io_gp=9000, hpwl=1.02e8)))
    baselines = {("adaptec1", 16, "grid"): {"io_count": 10000, "io_gp": 10000, "hpwl": 1.0e8}}
    md = build_m2_table([str(p)], baselines)
    cells = [c.strip() for c in md.splitlines()[2].split("|")]
    d_io = cells[[h for h, _ in M2_TABLE_SPEC].index("Δio%") + 1]
    d_io_gp = cells[[h for h, _ in M2_TABLE_SPEC].index("Δio_gp%") + 1]
    d_hpwl = cells[[h for h, _ in M2_TABLE_SPEC].index("Δhpwl%") + 1]
    assert d_io == "-5.0"
    assert d_io_gp == "-10.0"
    assert d_hpwl == "2.0"

def test_delta_pct_dash_when_no_baseline_for_case_k_rtype(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_io_row()))
    md = build_m2_table([str(p)], {})   # no baseline at all
    cells = [c.strip() for c in md.splitlines()[2].split("|")]
    for label in ("Δio%", "Δio_gp%", "Δhpwl%"):
        idx = [h for h, _ in M2_TABLE_SPEC].index(label) + 1
        assert cells[idx] == "—"

def test_delta_pct_dash_when_baseline_metric_is_zero(tmp_path):
    """A flat baseline of 0 must not raise ZeroDivisionError or render inf."""
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_io_row()))
    baselines = {("adaptec1", 16, "grid"): {"io_count": 0, "io_gp": 23554, "hpwl": 7.4e7}}
    md = build_m2_table([str(p)], baselines)
    cells = [c.strip() for c in md.splitlines()[2].split("|")]
    idx = [h for h, _ in M2_TABLE_SPEC].index("Δio%") + 1
    assert cells[idx] == "—"

def test_multiple_rows_preserve_input_order(tmp_path):
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps(_io_row(k=8)))
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps(_io_row(k=32)))
    md = build_m2_table([str(p2), str(p1)], {})
    body = md.splitlines()[2:]
    assert len(body) == 2
    assert "| 32 |" in body[0]
    assert "| 8 |" in body[1]
