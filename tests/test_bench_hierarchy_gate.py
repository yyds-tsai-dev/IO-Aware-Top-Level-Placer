"""Unit tests for M4 T3a's hierarchy gate (`ioplace/bench/hierarchy_gate.py`)
against small synthetic DEF fixtures -- the real `mempool_cluster.def` (9.7
GB) is exercised separately by the T3a probe run, not by pytest (design
draft `docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec
3.1a)."""
import os

from ioplace.bench import hierarchy_gate as hg

_DEF_HEADER = (
    "VERSION 5.8 ;\n"
    'DIVIDERCHAR "/" ;\n'
    'BUSBITCHARS "[]" ;\n'
    "DESIGN toy ;\n"
)


def _write_def(path, groups, residual, nets):
    """`groups`: {prefix: [(name, cellname, x, y), ...]}.
    `residual`: [(name, cellname, x, y), ...] (no `/` in name).
    `nets`: [(netname, [(inst_or_PIN, pinname), ...]), ...]."""
    lines = [_DEF_HEADER]
    n_comp = sum(len(cells) for cells in groups.values()) + len(residual)
    lines.append(f"COMPONENTS {n_comp} ;\n")
    for prefix, cells in groups.items():
        for name, cellname, x, y in cells:
            lines.append(f"- {prefix}/{name} {cellname} + PLACED ( {x} {y} ) N\n")
            lines.append(" ;\n")
    for name, cellname, x, y in residual:
        lines.append(f"- {name} {cellname} + PLACED ( {x} {y} ) N\n")
        lines.append(" ;\n")
    lines.append("END COMPONENTS\n")
    lines.append(f"NETS {len(nets)} ;\n")
    for netname, pins in nets:
        lines.append(f"- {netname}\n")
        lines.append("  " + " ".join(f"( {inst} {pin} )" for inst, pin in pins) + "\n")
        lines.append(" ;\n")
    lines.append("END NETS\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)


def _quadrant_cells(n_bufx2=5, n_invx1=5):
    """10 cells (5 BUFX2 + 5 INVX1) named n0..n9, all at the group's own
    origin -- bbox/centroid collapse to that single point, which is enough
    to drive `_classify_arrangement`."""
    cells = []
    idx = 0
    for _ in range(n_bufx2):
        cells.append((f"n{idx}", "BUFX2", 0, 0)); idx += 1
    for _ in range(n_invx1):
        cells.append((f"n{idx}", "INVX1", 0, 0)); idx += 1
    return cells


def _place_at(cells, x, y):
    return [(name, cell, x, y) for name, cell, _, _ in cells]


def _make_2x2_fixture(tmp_path, *, group_cells=10, perturb_group0=0):
    """4 groups g0..g3 at the corners of a 100x100 square (2x2 arrangement,
    4 adjacent pairs at d=100, 2 diagonal pairs at d=141.42), 1 top-level
    residual cell, and a handful of nets (internal/crossing/top-only) to
    smoke-test the NETS-section state machine. `perturb_group0` adds extra
    cells to g0 only, to push G-B's size-tolerance check."""
    base = _quadrant_cells()
    g0 = _place_at(base, 0, 0) + [(f"extra{i}", "BUFX2", 0, 0) for i in range(perturb_group0)]
    groups = {
        "g0": g0,
        "g1": _place_at(base, 100, 0),
        "g2": _place_at(base, 0, 100),
        "g3": _place_at(base, 100, 100),
    }
    residual = [("top0", "BUFX2", 50, 50)]
    nets = [
        ("net_internal_g0", [("g0/n0", "O"), ("g0/n1", "I")]),
        ("net_cross_g0_g1", [("g0/n0", "O"), ("g1/n0", "I")]),
        ("net_top_port_only", [("PIN", "clk")]),
        ("net_touches_residual", [("top0", "A"), ("g2/n0", "B")]),
    ]
    cluster_def = str(tmp_path / "cluster.def")
    _write_def(cluster_def, groups, residual, nets)

    # Standalone reference: single design-level bucket, same 10-cell
    # BUFX2/INVX1 histogram as an unperturbed quadrant group.
    group_def = str(tmp_path / "group.def")
    _write_def(group_def, {}, _place_at(_quadrant_cells(), 0, 0), nets=[])
    return cluster_def, group_def


def test_2x2_pass():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cluster_def, group_def = _make_2x2_fixture(__import__("pathlib").Path(tmp))
        buckets, meta = hg.scan_def(cluster_def, hg.top_level_prefix)

        assert meta["declared_components"] == 41  # 4*10 + 1 residual
        assert meta["scanned_components"] == 41
        assert meta["declared_nets"] == 4
        assert meta["scanned_nets"] == 4

        assert set(buckets) == {"g0", "g1", "g2", "g3", hg.TOP_LEVEL_RESIDUAL_KEY}
        assert buckets["g0"].cells == 10
        assert buckets["g0"].celltype == {"BUFX2": 5, "INVX1": 5}
        assert buckets[hg.TOP_LEVEL_RESIDUAL_KEY].cells == 1
        # net_internal_g0 -> g0 internal; net_cross_g0_g1 -> g0/g1 crossing;
        # net_top_port_only touches no instance at all; net_touches_residual
        # -> __TOP__/g2 crossing.
        assert buckets["g0"].nets_internal == 1
        assert buckets["g0"].nets_crossing_or_top == 1
        assert buckets["g2"].nets_crossing_or_top == 1
        assert buckets[hg.TOP_LEVEL_RESIDUAL_KEY].nets_crossing_or_top == 1

        result = hg.run(cluster_def, group_def)

    assert result["verdict"] == "PASS"
    assert result["arrangement"] == "2x2"
    assert result["protocol"] == "2x2"

    g_a, g_b, g_c, g_d, g_e = (result[k] for k in ("g_a", "g_b", "g_c", "g_d", "g_e"))
    assert g_a["pass_"] is True
    assert g_a["n_distinct_prefixes"] == 5  # g0..g3 + __TOP__
    assert g_a["coverage"] == 40 / 41

    assert g_b["pass_"] is True
    assert g_b["identity_holds"] is True
    assert g_b["standalone_cells"] == 10
    assert g_b["top4_cells"] == 40
    assert g_b["n_top"] == 1

    assert g_c["pass_"] is True
    for pg in result["per_group"]:
        assert pg["rel_diff_vs_standalone"] == 0.0
        assert abs(pg["celltype_cosine_vs_standalone"] - 1.0) < 1e-12

    assert g_d["pass_"] is True
    assert g_d["bbox_available"] is True
    dist_pair_counts = sorted(lvl["pair_count"] for lvl in g_d["distance_levels"])
    assert dist_pair_counts == [2, 4]

    assert g_e["n_top"] == 1
    assert g_e["residual"]["cells"] == 1


def test_g_b_fails_when_a_group_is_oversized():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # +30% extra cells in g0 alone blows the 15% size tolerance (G-B)
        # while G-A's 95% coverage and G-D's arrangement still hold.
        cluster_def, group_def = _make_2x2_fixture(
            __import__("pathlib").Path(tmp), perturb_group0=3)
        result = hg.run(cluster_def, group_def)

    assert result["verdict"] == "FAIL"
    assert result["g_b"]["pass_"] is False
    g0 = next(pg for pg in result["per_group"] if pg["prefix"] == "g0")
    assert g0["cells"] == 13
    assert g0["size_within_tolerance"] is False
    assert abs(g0["rel_diff_vs_standalone"] - 0.3) < 1e-9
    # G-A (coverage) and G-D (arrangement determinable) are unaffected by a
    # cell-count perturbation that doesn't touch prefix count or geometry.
    assert result["g_a"]["pass_"] is True
    assert result["g_d"]["pass_"] is True


def test_below_95_percent_coverage_fails_g_a():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = __import__("pathlib").Path(tmp)
        base = _quadrant_cells()
        groups = {
            "g0": _place_at(base, 0, 0), "g1": _place_at(base, 100, 0),
            "g2": _place_at(base, 0, 100), "g3": _place_at(base, 100, 100),
        }
        # A large residual (10 cells, same size as one group) pushes top4
        # coverage down to 40/50 = 80% < 95%.
        residual = [(f"top{i}", "BUFX2", 50, 50) for i in range(10)]
        cluster_def = str(tmp_path / "cluster.def")
        _write_def(cluster_def, groups, residual, nets=[])
        group_def = str(tmp_path / "group.def")
        _write_def(group_def, {}, _place_at(_quadrant_cells(), 0, 0), nets=[])

        result = hg.run(cluster_def, group_def)

    assert result["g_a"]["pass_"] is False
    assert result["g_a"]["coverage"] == 40 / 50
    assert result["verdict"] == "FAIL"


def test_top_level_prefix():
    assert hg.top_level_prefix("gen_groups\\[0\\].i_group/gen_tiles\\[0\\].i_tile/x") == \
        "gen_groups\\[0\\].i_group"
    assert hg.top_level_prefix("wake_up_q_reg\\[212\\]") == hg.TOP_LEVEL_RESIDUAL_KEY


def test_unescape():
    assert hg._unescape("gen_groups\\[0\\].i_group") == "gen_groups[0].i_group"
