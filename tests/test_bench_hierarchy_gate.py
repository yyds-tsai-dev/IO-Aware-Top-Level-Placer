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


def _write_def(path, groups, residual, nets, die=(0, 0, 100, 100)):
    """`groups`: {prefix: [(name, cellname, x, y), ...]}.
    `residual`: [(name, cellname, x, y), ...] (no `/` in name).
    `nets`: [(netname, [(inst_or_PIN, pinname), ...])] or [(netname, pins,
    plus_clause), ...] -- `plus_clause` (if given), a literal `+ ...` line
    written after the pin tuples and before the record's terminating ` ;`,
    for exercising the NETS `+`-clause truncation (Bug B).
    `die`: `(xl, yl, xh, yh)` for the `DIEAREA` line, or `None` to omit it
    (G-D's classifier needs one; G-A/B/C/E fixtures that don't touch G-D
    can pass `None`)."""
    lines = [_DEF_HEADER]
    if die is not None:
        xl, yl, xh, yh = die
        lines.append(f"DIEAREA ( {xl} {yl} ) ( {xh} {yh} ) ;\n")
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
    for net in nets:
        netname, pins = net[0], net[1]
        plus_clause = net[2] if len(net) > 2 else None
        lines.append(f"- {netname}\n")
        lines.append("  " + " ".join(f"( {inst} {pin} )" for inst, pin in pins) + "\n")
        if plus_clause:
            lines.append("  " + plus_clause + "\n")
        lines.append(" ;\n")
    lines.append("END NETS\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)


def _quadrant_cells(n_bufx2=5, n_invx1=5):
    """10 cells (5 BUFX2 + 5 INVX1) named n0..n9, all at the group's own
    origin -- bbox/centroid collapse to that single point, which is enough
    to drive `classify_arrangement`'s mass-weighted assignment scorer."""
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
        buckets, meta, _ = hg.scan_def(cluster_def, hg.top_level_prefix)

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

    assert result["verdict"] == "PASS_2x2"
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

    # 4 groups at exact corners of the die -> score_2x2 finds a perfect
    # (score=1.0) split for any cx/cy in the search range, while 1x4/4x1's
    # fixed quartiles can only place 2 of the 4 groups' mass uniquely
    # (score=0.5 each) -- see module docstring for the general argument.
    assert g_d["pass_"] is True
    assert g_d["arrangement"] == "2x2"
    assert g_d["scores"] == {"2x2": 1.0, "1x4": 0.5, "4x1": 0.5}
    assert g_d["margin"] == 0.5
    assert g_d["score_2x2"]["assignment"] == {"g0": "LL", "g1": "RL", "g2": "LU", "g3": "RU"}
    assert g_d["left_right_purity"] == 1.0
    assert g_d["top_bottom_purity"] == 1.0
    assert g_d["bin_dominance"]["mean_max_share"] == 1.0

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


def test_plus_clause_truncates_pin_scan():
    """Bug B (2026-08-14 T6 adjudication doc sec 4 前置 1): a NETS record's
    `+ ROUTED ...` clause is not pin tuples -- routing-coordinate tuples
    like `( 10 20 )` and a `( * pinname )` wildcard tuple are both 4-token
    parenthesized groups too, and must not be scanned as instance pins
    once the record's first `+` token is seen (routing coords), or at all
    (wildcard)."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        groups = {"g0": _place_at(_quadrant_cells(), 0, 0),
                  "g1": _place_at(_quadrant_cells(), 100, 0)}
        nets = [
            ("net_route",
             [("g0/n0", "O"), ("g1/n0", "I"), ("*", "wild_pin")],
             "+ ROUTED metal1 ( 10 20 ) ( 30 40 )"),
        ]
        cluster_def = str(tmp_path / "cluster.def")
        _write_def(cluster_def, groups, residual=[], nets=nets, die=None)
        buckets, meta, _ = hg.scan_def(cluster_def, hg.top_level_prefix)

    # 2 real pin tuples (g0/n0, g1/n0) before the `+`; the wildcard tuple
    # before `+` is excluded and separately counted; the 2 routing-
    # coordinate tuples after `+` must not be scanned as pins at all.
    assert meta["scanned_pin_tuples"] == 2
    assert meta["n_wildcard_tuples"] == 1
    assert meta["n_post_plus_tuples_skipped"] == 2
    assert buckets["g0"].pins == 1
    assert buckets["g1"].pins == 1


def test_mass_centroid_differs_from_bbox_center():
    """Bug A (2026-08-14 T6 adjudication doc sec 4 前置 2): a placed
    group's bbox midpoint can be dominated by a couple of outlier cells and
    say nothing about where the group's mass actually sits -- the mass
    centroid (`sum_x/n_placed`, `sum_y/n_placed`) must differ from it
    here."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        bulk = [(f"n{i}", "BUFX2", 10, 10) for i in range(9)]
        outlier = [("n_outlier", "BUFX2", 1000, 1000)]
        groups = {"g0": bulk + outlier}
        cluster_def = str(tmp_path / "cluster.def")
        _write_def(cluster_def, groups, residual=[], nets=[], die=None)
        buckets, meta, _ = hg.scan_def(cluster_def, hg.top_level_prefix)

    d = buckets["g0"].to_dict()
    assert d["bbox_outlier_extent"] == [10.0, 10.0, 1000.0, 1000.0]
    bbox_center = [(10.0 + 1000.0) / 2, (10.0 + 1000.0) / 2]  # (505, 505)
    assert d["centroid"] == [109.0, 109.0]  # (9 * 10 + 1000) / 10
    assert d["centroid"] != bbox_center
    assert abs(d["centroid"][0] - bbox_center[0]) > 300


def test_arrangement_scoring_classifies_1x4_fixture_correctly():
    """A row arrangement (not a grid) must score highest under the 1x4
    hypothesis, not 2x2 -- `classify_arrangement`'s mass-weighted
    assignment scorer (2026-08-14 T6 adjudication doc sec 4 前置 3) must
    discriminate arrangement *shape*, not just "is there SOME spatial
    separation" (companion to `test_2x2_pass`'s 2x2 fixture, which already
    asserts `scores == {"2x2": 1.0, "1x4": 0.5, "4x1": 0.5}`)."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        base = _quadrant_cells()
        groups = {
            "g0": _place_at(base, 0, 0), "g1": _place_at(base, 100, 0),
            "g2": _place_at(base, 200, 0), "g3": _place_at(base, 300, 0),
        }
        cluster_def = str(tmp_path / "cluster.def")
        _write_def(cluster_def, groups, residual=[], nets=[], die=(0, 0, 400, 100))
        group_def = str(tmp_path / "group.def")
        _write_def(group_def, {}, _place_at(_quadrant_cells(), 0, 0), nets=[])

        result = hg.run(cluster_def, group_def)

    g_d = result["g_d"]
    assert g_d["scores"] == {"2x2": 0.5, "1x4": 1.0, "4x1": 0.25}
    assert g_d["arrangement"] == "1x4"
    assert g_d["pass_"] is True
    assert result["verdict"] == "PASS_1x4"
