# v2 Subproject P-F — Straddling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three-way anchor inconsistency between the differentiable IO term (node lower-left), the evaluator (pins) and the fence (whole cell) by anchoring the soft assignment at the cell centre, and instrument the residual straddling with five reported diagnostics whose accounting identity `io(final) = io(soft) + io_delta_at_freeze + lg_loss` closes to within ±1 crossing.

**Architecture:** Three independent seams. (1) A per-node constant offset applied once, where `x`/`y` are sliced out of `pos`, turns `lower_left` into `center` without touching `region_sdf_l1`, without changing any tensor shape, and without changing any gradient (the offset is constant, so `d/dx == d/d(x+h)`); the `pin` arm is a structurally different, node→pin re-indexed forward path that exists only in `IoTermRef`. (2) A new numpy module `src/ioplace/straddle.py` fixes the geometric conventions once; `evaluator_ref` calls it and `evaluator_gpu` mirrors it in torch, bound together by bit-exact parity tests in the established `tests/test_evaluator_gpu.py` style. (3) `evaluation.npz` grows a schema-version-2 straddle block and `result.json` grows six straddle fields plus `node_anchor`.

**Tech Stack:** Python 3.12 (`$DREAMPLACE_ROOT/.venv312/bin/python`), torch 2.8.0+cu128, numpy, pytest, DREAMPlace 4.3.1 (`$DREAMPLACE_ROOT/install`). No new DREAMPlace patch.

**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` — §0 "Straddling" row, all of §7, §9 (parity + the F "done" criterion), §1 (`evaluation.npz` schema). Read the spec before starting; this plan argues from it and the two travel together.

**Dependency plans — read, do not duplicate:**

- `docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md` (HEAD version). P-B owns, and P-F must **not** re-implement: Task 3 `src/ioplace/freeze.py` (cell-centre argmax `argmax_region`/`cell_centers`, the freeze criterion, `region_cell_stats`); Task 6 `src/ioplace/main_flow_metrics.py` (`io_accounting` → `io_delta_at_freeze`/`lg_loss`/`io_identity_residual`, `fence_compliance` with its `lower_left` and `center` keys, `region_area_balance`); Task 7 `result.json` and `artifacts.MAIN_FLOW_RESULT_FIELDS` (`io_fence_gp_source`, the identity fields). P-F consumes those names and adds fields next to them.
- `docs/superpowers/plans/2026-09-19-v2-p-h-normalisation.md` — P-H owns `src/ioplace/norm.py` and the `--norm-*` flags already present in `run_placement.build_parser`. P-F adds one flag in the same parser and touches nothing else there.

**What P-F owns:** the `--node-anchor` flag end to end; the `pin` reference arm; the three evaluator-side diagnostics `straddle_cells` / `straddle_area_fraction` / `straddle_pin_split_nets`; their persistence and reporting; the anchor-comparison experiment; the accounting-identity checker.

---

## Global Constraints

Every task's requirements implicitly include this section.

**Run protocol.** From the repo root `/ldaphome/yyds-tsai-dev/IO-Aware-Top-Level-Placer`, branch `v2/redesign`:

```bash
source src/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3
"$IOPLACE_PYTHON" -m pytest
```

`src/scripts/env.sh` exports `DREAMPLACE_ROOT=/ldaphome/yyds-tsai-dev/DREAMPlace` and `IOPLACE_PYTHON=$DREAMPLACE_ROOT/.venv312/bin/python`. Use `-m "not slow"` while iterating; run the full suite before declaring a task done. This is a shared H100 NVL host: run `nvidia-smi` before any GPU work and honour `CUDA_VISIBLE_DEVICES`. At the time this plan was written GPUs 0–2 were at 100% utilisation and GPU 3 was carrying a long P-B/P-C acceptance run (22 GB resident) — check again, and if GPU 3 is still busy run only `-m "not slow and not gpu"` and defer Tasks 6/7's slow tests.

**No new DREAMPlace patch.** DREAMPlace source is off-limits. `src/ioplace/dp_patch/m2-extra-obj-terms.patch`, `iteration-callback.patch` and `shapely2-compat.patch` are the only modifications and none of them changes. Nothing in this plan needs a hook that does not already exist.

**Python >= 3.9.** No `match`, no PEP-604 `X | Y` annotations, no PEP-585 `tuple[int, ...]` annotations in runtime code (the interpreter is 3.12, but the floor is 3.9 and the rest of `src/ioplace` honours it).

**`--node-anchor` default is `center`; `pin` is IoTermRef-only.** The flag is `--node-anchor {lower_left,center,pin}` with default `center` (spec §0/§7). Every driver rejects `pin` with a `ValueError` before it touches CUDA. `IoTerm` (the production chunked term) rejects `pin` in its constructor. `IoTermRef` is the only place `pin` is implemented, and it is a small-scale bias probe, never a driver path. The *class-level* default of `node_anchor` on `IoTermRef`/`IoTerm` stays `"lower_left"` — a deliberate, recorded interpretation of the spec: the spec sets the default of the **flag**, and fourteen existing construction sites (`tests/test_io_term.py`, `tests/test_io_term_chunked.py`, `tests/test_ft_term.py`, `src/ioplace/bench/spike_30m_child.py`, `src/ioplace/diagnostics/spike_10m.py`, `src/scripts/run_component_models.py`) supply no node sizes, which `center` requires. The two defaults are wired together in exactly one place per driver and Task 1 pins that with a test.

**Evaluator parity = integer fields bit-exact.** Extending `tests/test_evaluator_gpu.py`'s existing contract (module docstring, `evaluator_gpu.py:16-26`; `_assert_batch_invariant_fields`, `tests/test_evaluator_gpu.py:426-445`): the **integer** straddle fields `straddle_cells`, `straddle_pin_split_nets`, `straddle_wide_cells`, `per_node_straddle`, `per_net_pin_split` must be **bit-exact** between `evaluator_ref` and `evaluator_gpu` and across construction parameters (`mst_chunk_budget`, `seg_chunk_budget`, `edge_batch_size`). The **float** fields `straddle_area_fraction`, `straddle_out_area`, `straddle_movable_area` carry the same `rel <= 1e-12` contract `tree_wl`/`hpwl` already carry — float64 addition is not associative and the numpy and torch reductions do not share an order.

**No overlap/straddle penalty term.** Spec §0 and §7: containment comes from the fence LG. Nothing in this plan adds an objective term, a Lagrangian, or a gradient. Every quantity P-F introduces is measured, never optimised.

**Commit trailer.** Every commit message in this plan ends with:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```

**"Done" for F (spec §9).** The five diagnostics `straddle_cells`, `straddle_area_fraction`, `straddle_pin_split_nets`, `io_delta_at_freeze`, `fence_compliance` are all reported in one `result.json`, and the accounting identity closes within ±1 crossing. Task 6 makes that machine-checkable; Task 7 produces the anchor-comparison evidence §7 asks for.

---

## File Structure

**New modules**

| File | Responsibility |
|---|---|
| `src/ioplace/straddle.py` | The numpy reference for the three geometric/attribution diagnostics and, in its module docstring, the five conventions (closed four-corner box, centre owner, movable-only, quadrant area split, pin re-attribution) that `evaluator_gpu`'s torch mirror must reproduce bit-for-bit. Pure numpy; imports only `numpy` and `ioplace.netlist.pin_positions`. |
| `src/ioplace/io_identity.py` | Pure-arithmetic verification of spec §7's accounting identity against *independently re-measured* IO counts, plus the `p_f_diagnostics` extractor that asserts F's five-diagnostic "done" criterion over a `result.json` dict. No torch, no numpy-heavy work, no I/O. |
| `src/scripts/run_anchor_comparison.py` | The §7 verification experiment: one soft solution, three anchors, GP-surrogate IO vs post-fence-LG evaluator truth. Writes `anchor_comparison.json` and a markdown table. |

**Modified**

| File | Change |
|---|---|
| `src/ioplace/ops/soft_assign.py:19-31` (append after) | `NODE_ANCHORS`, `anchor_offsets()` — the one place the centre offset is defined. `region_sdf_l1` itself is untouched. |
| `src/ioplace/ops/io_term.py:60-127,291-346,356-423` | `PinCsr`/`build_net_pin_csr`; `node_anchor`/`node_size_x`/`node_size_y`/`pin_csr` on `IoTermRef`; `_anchor_xy` on both classes; the `pin` forward arm in `IoTermRef`; `IoTerm`'s `pin` rejection; anchoring inside `IoTerm.diagnostics`. |
| `src/ioplace/ops/ft_term.py:149-156,167-199` | `FtTerm._evaluate_parts` and `FtTermRef._values` anchor through the shared `IoTerm`; `FtTermRef` rejects a pin-anchored term. |
| `src/ioplace/evaluator_ref.py:6-25,83-172` | Eight new `EvalResult` fields; `evaluate(..., straddle=True)`. |
| `src/ioplace/evaluator_gpu.py:93-283,531-769` | The torch mirror: `straddle=True` constructor flag, size tensors, `_straddle_stats`, `evaluate(..., straddle=None)`. |
| `src/ioplace/export/evaluation.py:14-18,68-141` | `SCHEMA_VERSION = 2`, `SUPPORTED_SCHEMA_VERSIONS`, the straddle arrays + metadata block, reader validation. |
| `src/ioplace/drivers/run_placement.py:114-119,505-573` | `_pack_straddle_metrics`; `--node-anchor` in `build_parser`; pass-through in `main`. |
| `src/ioplace/drivers/run_placement_io.py:33-58,139-178,291-293,518,784-785,839-860` | `node_anchor` parameter + validation; anchor threaded into `IoTerm`; `straddle=False` on the in-loop evaluation; straddle metrics + `node_anchor` in `result`; seven new `RESULT_FIELDS`. |
| `docs/dev-env.md` | Driver-flag table gains `--node-anchor`; evaluator section gains the straddle fields and the npz schema bump. |

**New tests** — `tests/test_node_anchor.py`, `tests/test_straddle.py`, `tests/test_io_identity.py`, `tests/test_anchor_comparison.py`.
**Modified tests** — `tests/test_evaluator_gpu.py`, `tests/test_evaluation_export.py`.
**New docs** — `docs/results/2026-09-19-p-f-anchor-comparison.md`.

**Reused unchanged — do not edit:** `regions.py`, `region_grid.py`, `region_graph.py`, `fence_inject.py`, `netlist.py`, `schedules.py`, `dp_hook.py`, `drivers/run_placement_two_stage.py`, and (if they have landed) `freeze.py`, `main_flow_metrics.py`, `artifacts.py` — except for the single guarded field/flag addition Task 5 makes to `artifacts.MAIN_FLOW_RESULT_FIELDS` and `drivers/run_main_flow.py`.

**Test inputs.** GCD is the smallest real LEF/DEF case on this host: config `results/route_feedback_20260914/gcd.json` (508 movable nodes, 168 terminals, 579 nets). The §7 experiment case is `mempool_tile_wrap`: use `benchmarks/ispd25/h100/mempool_tile_wrap.json` if P-C Task 11 has landed it, otherwise `results/route_gp_20260914/mempool_tile_wrap.json` (host-local, present, 129,033 physical nodes, `target_density 0.399`, 2000 GP iterations — `.superpowers/sdd/2026-09-19-v2-p-c-region-producer/preflight.md` §E6). The checked-in `benchmarks/ispd25/mempool_tile_wrap.json` points at `/nashome/NVL4/...`, which does not exist here — never use it, and do not repair it (P-C Task 11 owns that).

---

### Task 1: The `center` anchor through `soft_assign` / `IoTerm` / `FtTerm` and the `--node-anchor` flag

**Files:**
- Modify: `src/ioplace/ops/soft_assign.py` (append after line 31)
- Modify: `src/ioplace/ops/io_term.py:60-127` (`IoTermRef`), `:291-346` (`IoTerm.__init__`/`forward`), `:373-380` (`IoTerm.diagnostics`)
- Modify: `src/ioplace/ops/ft_term.py:149-156` (`FtTerm._evaluate_parts`), `:167-199` (`FtTermRef`)
- Modify: `src/ioplace/drivers/run_placement.py:505-573` (`build_parser`, `main`)
- Modify: `src/ioplace/drivers/run_placement_io.py:33-58` (`RESULT_FIELDS`), `:139-178` (`run_io` signature + validation), `:291-293` (`IoTerm(...)`), `:839-860` (`result`)
- Test: `tests/test_node_anchor.py` (new)

**Interfaces:**
- Consumes: `ioplace.ops.soft_assign.region_sdf_l1`/`softmax_stats`/`chunk_p_ell` (`src/ioplace/ops/soft_assign.py:19-58`), unchanged; `tests.test_io_term._nl`/`_pos`/`DIE` (`tests/test_io_term.py:17-37`).
- Produces:
  - `ioplace.ops.soft_assign.NODE_ANCHORS = ("lower_left", "center", "pin")`
  - `ioplace.ops.soft_assign.anchor_offsets(node_anchor, node_size_x, node_size_y, num_physical, *, device, dtype=torch.float64) -> (torch.Tensor | None, torch.Tensor | None)`
  - `IoTermRef(..., *, node_anchor="lower_left", node_size_x=None, node_size_y=None, pin_csr=None)` and `IoTerm(..., *, node_anchor="lower_left", node_size_x=None, node_size_y=None)`, both with `.node_anchor` (str) and `._anchor_xy(x, y) -> (Tensor, Tensor)`
  - `run_io(..., node_anchor="center")`; `RESULT_FIELDS` gains `"node_anchor"`; `build_parser()` gains `--node-anchor`

**Why this is a one-line change per call site.** `region_sdf_l1` is a pure function of a point; the anchor only decides *which* point. Adding a per-node constant `h` before it leaves every tensor shape identical and every gradient identical, because `d f(x+h)/dx == f'(x+h)` — the same value `_IoFn.backward` already computes by running `torch.autograd.grad` on `region_sdf_l1` at the evaluated point (`io_term.py:261-283`). So the anchor is applied exactly where `x`/`y` leave `pos`: `IoTermRef._split_xy`, `IoTerm.forward`, `IoTerm.diagnostics`, `FtTerm._evaluate_parts`, `FtTermRef._values`. `pin` cannot be expressed this way at all — it re-indexes the accumulation from `(N,K)` over nodes to `(P,K)` over pins — which is precisely why spec §7 confines it to `IoTermRef`.

- [x] **Step 1: Write the failing test**

Create `tests/test_node_anchor.py`:

```python
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ioplace.ops.io_term import IoTerm, IoTermRef, build_net_node_csr
from ioplace.ops.soft_assign import NODE_ANCHORS, anchor_offsets, rect_table
from ioplace.regions import make_grid_regions
from tests.test_io_term import DIE, _nl, _pos

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _straddler():
    """One 2-pin net. Cell 0 sits inside region 0. Cell 1's lower-left corner is
    in region 0 (x=44 < 50) but its centre is in region 1 (44 + 14/2 = 51), so
    the anchor alone decides whether this net crosses a boundary."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)          # boundaries at x=50, y=50
    nl = _nl([(10., 10.), (44., 10.)], [[0, 1]])
    nl.node_size_x = np.array([4., 14.])
    nl.node_size_y = np.array([4., 4.])
    return rs, nl


def _ref(nl, rs, anchor, cls=IoTermRef, **extra):
    rects, r2k = rect_table(rs)
    kw = dict(csr=build_net_node_csr(nl, 100), rects=rects, rect2region=r2k, K=rs.k,
              num_movable=nl.num_movable, num_physical=nl.num_physical,
              num_nodes=nl.num_physical, device=DEV, node_anchor=anchor)
    if anchor == "center":
        kw.update(node_size_x=nl.node_size_x, node_size_y=nl.node_size_y)
    kw.update(extra)
    return cls(**kw)


def test_node_anchors_tuple_is_the_three_spec_values():
    assert NODE_ANCHORS == ("lower_left", "center", "pin")


def test_anchor_offsets_lower_left_is_none_center_is_half_the_cell():
    assert anchor_offsets("lower_left", None, None, 3, device="cpu") == (None, None)
    dx, dy = anchor_offsets("center", np.array([4., 6., 8.]), np.array([2., 2., 2.]),
                            3, device="cpu")
    assert dx.tolist() == [2., 3., 4.] and dy.tolist() == [1., 1., 1.]
    assert dx.dtype == torch.float64


def test_anchor_offsets_rejects_pin_and_bad_sizes():
    with pytest.raises(ValueError, match="only in IoTermRef"):
        anchor_offsets("pin", np.ones(3), np.ones(3), 3, device="cpu")
    with pytest.raises(ValueError, match="requires node_size_x"):
        anchor_offsets("center", None, None, 3, device="cpu")
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        anchor_offsets("center", np.ones(2), np.ones(3), 3, device="cpu")


def test_center_anchor_makes_a_straddling_cell_cross_the_boundary():
    """The whole point of spec sec 7: at the same positions, the lower-left
    anchor scores this net as fully contained (L_IO = 0) while the centre
    anchor scores the one real crossing it will have after fence LG (L_IO = 1)."""
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    tau = 0.05
    ll = _ref(nl, rs, "lower_left")(pos, tau, 1.0)
    ce = _ref(nl, rs, "center")(pos, tau, 1.0)
    assert float(ll.detach()) == pytest.approx(0.0, abs=1e-9)
    assert float(ce.detach()) == pytest.approx(1.0, abs=1e-9)


def test_center_anchor_leaves_the_gradient_on_the_movable_nodes_only():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    term = _ref(nl, rs, "center")
    term(pos, 0.3, 1.0).backward()
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().sum()) > 0.0


def test_io_term_matches_io_term_ref_under_the_center_anchor():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    for chunk_budget in (10 ** 9, 1):
        prod = _ref(nl, rs, "center", cls=IoTerm, chunk_budget=chunk_budget)
        ref = _ref(nl, rs, "center")
        assert prod.node_anchor == "center"
        a = prod(pos, 0.3, 1.0)
        b = ref(pos, 0.3, 1.0)
        assert float(a.detach()) == pytest.approx(float(b.detach()), rel=1e-12)
        ga, = torch.autograd.grad(a, pos, retain_graph=False)
        gb, = torch.autograd.grad(b, pos, retain_graph=False)
        assert torch.allclose(ga, gb, rtol=1e-10, atol=1e-12)


def test_io_term_diagnostics_use_the_same_anchor_as_forward():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    prod = _ref(nl, rs, "center", cls=IoTerm, chunk_budget=10 ** 9)
    assert prod.diagnostics(pos.detach(), 0.05)["l_io"] == pytest.approx(1.0, abs=1e-9)
    plain = _ref(nl, rs, "lower_left", cls=IoTerm, chunk_budget=10 ** 9)
    assert plain.diagnostics(pos.detach(), 0.05)["l_io"] == pytest.approx(0.0, abs=1e-9)


def test_io_term_rejects_the_pin_anchor():
    rs, nl = _straddler()
    with pytest.raises(ValueError, match="only in IoTermRef"):
        _ref(nl, rs, "pin", cls=IoTerm, chunk_budget=10 ** 9)


def test_center_anchor_requires_sizes_over_all_physical_nodes():
    rs, nl = _straddler()
    rects, r2k = rect_table(rs)
    with pytest.raises(ValueError, match="requires node_size_x"):
        IoTermRef(csr=build_net_node_csr(nl, 100), rects=rects, rect2region=r2k,
                  K=rs.k, num_movable=2, num_physical=2, num_nodes=2, device=DEV,
                  node_anchor="center")


def test_unknown_anchor_is_rejected_by_both_classes():
    rs, nl = _straddler()
    for cls, extra in ((IoTermRef, {}), (IoTerm, {"chunk_budget": 10 ** 9})):
        with pytest.raises(ValueError, match="node_anchor"):
            _ref(nl, rs, "centre", cls=cls, **extra)


def test_ft_term_inherits_the_anchor_from_its_io_term():
    from ioplace.ops.ft_term import FtTerm
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    D = np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]], dtype=np.float64)
    values = {}
    for anchor in ("lower_left", "center"):
        io = _ref(nl, rs, anchor, cls=IoTerm, chunk_budget=10 ** 9)
        ft = FtTerm(io, D)
        ft.set_home(torch.zeros(io.n_active, dtype=torch.int64, device=DEV))
        values[anchor] = float(ft.ft_only(pos, 0.05).detach())
    assert values["lower_left"] != pytest.approx(values["center"], abs=1e-9)


def test_io_term_ref_needs_a_pin_csr_for_the_pin_anchor():
    """Task 2 supplies the arm itself; Task 1 only guarantees the constructor
    cannot silently fall back to the node-level accumulation."""
    rs, nl = _straddler()
    with pytest.raises(ValueError, match="pin_csr"):
        _ref(nl, rs, "pin")


def test_run_io_rejects_the_pin_anchor_before_touching_cuda():
    from ioplace.drivers.run_placement_io import run_io
    with pytest.raises(ValueError, match="IoTermRef-only"):
        run_io("nonexistent.json", 16, "grid", 0, "out.json", node_anchor="pin")


def test_run_io_rejects_an_unknown_anchor():
    from ioplace.drivers.run_placement_io import run_io
    with pytest.raises(ValueError, match="node_anchor must be"):
        run_io("nonexistent.json", 16, "grid", 0, "out.json", node_anchor="centre")


def test_cli_defaults_node_anchor_to_center_and_offers_all_three():
    from ioplace.drivers.run_placement import build_parser
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--mode", "io", "--out", "o.json"])
    assert args.node_anchor == "center"
    action, = [a for a in parser._actions if a.dest == "node_anchor"]
    assert tuple(action.choices) == ("lower_left", "center", "pin")


def test_result_fields_carry_the_anchor():
    from ioplace.drivers.run_placement_io import RESULT_FIELDS
    assert "node_anchor" in RESULT_FIELDS
```

- [x] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_node_anchor.py -v`
Expected: FAIL — `ImportError: cannot import name 'NODE_ANCHORS' from 'ioplace.ops.soft_assign'` at collection.

- [x] **Step 3: Add `anchor_offsets` to `soft_assign.py`**

Append to `src/ioplace/ops/soft_assign.py`, immediately after `region_sdf_l1` (line 31):

```python
NODE_ANCHORS = ("lower_left", "center", "pin")


def anchor_offsets(node_anchor, node_size_x, node_size_y, num_physical, *,
                   device, dtype=torch.float64):
    """Per-node constant (dx, dy) added to the lower-left position before any
    region SDF is taken (design v2 sec 7).

    `lower_left` returns (None, None) so callers skip the add entirely and the
    legacy path stays bit-identical rather than merely numerically equal.
    `center` returns half the cell size: the anchor that agrees with the freeze
    membership (sec 3 phase 2) and with whole-cell fence ownership.

    `pin` never reaches here. It is not an offset at all -- it re-indexes the
    whole accumulation from nodes to pins, reintroducing the (P,K) cost and the
    multi-pin double counting that
    docs/superpowers/specs/2026-08-06-m2-differentiable-io-design.md:105 chose
    node anchoring to avoid -- and is implemented only in IoTermRef.
    """
    if node_anchor == "lower_left":
        return None, None
    if node_anchor != "center":
        raise ValueError(
            "anchor_offsets handles 'lower_left' and 'center'; got %r "
            "(anchor 'pin' is not an offset and is implemented only in IoTermRef)"
            % (node_anchor,))
    if node_size_x is None or node_size_y is None:
        raise ValueError("node_anchor='center' requires node_size_x and node_size_y")
    dx = torch.as_tensor(node_size_x, dtype=dtype, device=device).reshape(-1)
    dy = torch.as_tensor(node_size_y, dtype=dtype, device=device).reshape(-1)
    if dx.shape != (num_physical,) or dy.shape != (num_physical,):
        raise ValueError(
            "node_size_x/node_size_y must both have shape (%d,), got %s and %s"
            % (num_physical, tuple(dx.shape), tuple(dy.shape)))
    return 0.5 * dx, 0.5 * dy
```

- [x] **Step 4: Thread the anchor through `IoTermRef` and `IoTerm`**

In `src/ioplace/ops/io_term.py`, extend the import at line 13-14 to
`from ioplace.ops.soft_assign import (rect_table, region_sdf_l1, softmax_stats, chunk_p_ell, d_star_from_m, _chunks, NODE_ANCHORS, anchor_offsets)`.

Add this shared method to **both** `IoTermRef` and `IoTerm` (identical text in both classes — they share no base class today and this plan does not introduce one):

```python
    def _anchor_xy(self, x, y):
        """Apply the sec 7 anchor. A per-node *constant* offset, so every
        gradient w.r.t. pos is unchanged; only the point at which the region
        SDF is evaluated moves."""
        if self.anchor_dx is None:
            return x, y
        return x + self.anchor_dx.to(dtype=x.dtype), y + self.anchor_dy.to(dtype=x.dtype)
```

In `IoTermRef.__init__` (line 70), change the signature to

```python
    def __init__(self, csr, rects, rect2region, K, num_movable, num_physical,
                 num_nodes, device="cuda", w_mode="unit", *,
                 node_anchor="lower_left", node_size_x=None, node_size_y=None,
                 pin_csr=None):
```

and insert, right after the existing `w_mode` validation:

```python
        if node_anchor not in NODE_ANCHORS:
            raise ValueError("node_anchor must be one of %r, got %r"
                             % (NODE_ANCHORS, node_anchor))
        self.node_anchor = node_anchor
```

then, immediately after the existing `register_buffer("rect2region", ...)` call, add:

```python
        if node_anchor == "pin":
            # The pin arm re-indexes the accumulation from nodes to pins; it is
            # not an offset. Task 2 registers its buffers and forward branch --
            # until then, and forever after if the caller forgets the CSR, this
            # must never silently fall back to the node-level accumulation.
            if pin_csr is None:
                raise ValueError("node_anchor='pin' requires pin_csr "
                                 "(ioplace.ops.io_term.build_net_pin_csr)")
            dx = dy = None
        else:
            dx, dy = anchor_offsets(node_anchor, node_size_x, node_size_y,
                                    self.num_physical, device=device)
        self.register_buffer("anchor_dx", dx)
        self.register_buffer("anchor_dy", dy)
```

Change the last line of `IoTermRef._split_xy` (line 110) from `return x, y` to `return self._anchor_xy(x, y)`, and extend its docstring with one sentence: `"The sec 7 anchor is applied here, after the terminal detach, so every caller (forward, diagnostics) sees the same anchored coordinates."`

In `IoTerm.__init__` (line 307), change the signature to

```python
    def __init__(self, csr, rects, rect2region, K, num_movable, num_physical,
                 num_nodes, device="cuda", w_mode="unit", chunk_budget: int = 8_000_000,
                 *, node_anchor="lower_left", node_size_x=None, node_size_y=None):
```

and insert, right after the existing `w_mode` validation:

```python
        if node_anchor == "pin":
            raise ValueError(
                "IoTerm supports node_anchor 'lower_left' or 'center'; 'pin' is "
                "implemented only in IoTermRef (design v2 sec 7: the pin arm "
                "reintroduces (P,K) cost and multi-pin double counting, so it is "
                "a small-scale bias probe, never a production path)")
        if node_anchor not in NODE_ANCHORS:
            raise ValueError("node_anchor must be one of %r, got %r"
                             % (NODE_ANCHORS, node_anchor))
        self.node_anchor = node_anchor
```

then, after `register_buffer("rect2region", ...)`, add:

```python
        dx, dy = anchor_offsets(node_anchor, node_size_x, node_size_y,
                                self.num_physical, device=device)
        self.register_buffer("anchor_dx", dx)
        self.register_buffer("anchor_dy", dy)
```

In `IoTerm.forward` (lines 343-346), insert the anchor between the slices and the dispatch:

```python
    def forward(self, pos, tau, lambda_io, lambda_margin=0.0, margin_m=0.0, margin_tau=1.0):
        x = pos[:self.num_physical]
        y = pos[self.num_nodes:self.num_nodes + self.num_physical]
        x, y = self._anchor_xy(x, y)
        return _IoFn.apply(x, y, self, tau, lambda_io, lambda_margin, margin_m, margin_tau)
```

In `IoTerm.diagnostics`, after the two slices at lines 374-375 and before `rects = ...`, insert `x, y = self._anchor_xy(x, y)`. (The `grad_share` loop already routes through `self.forward`, which anchors itself.)

- [x] **Step 5: Thread the anchor through `FtTerm` and `FtTermRef`**

In `src/ioplace/ops/ft_term.py`, `FtTerm._evaluate_parts` (lines 152-156) becomes:

```python
        meta = self.io_term
        x = pos[:meta.num_physical]
        y = pos[meta.num_nodes:meta.num_nodes + meta.num_physical]
        x, y = meta._anchor_xy(x, y)          # design v2 sec 7: same anchor as IoTerm
        return _FtFn.apply(x, y, meta, self.D, self.home, tau, lambda_io,
                           io_scale, ft_scale, lambda_margin, margin_m, margin_tau)
```

In `FtTermRef.__init__` (line 169), after `super().__init__()`, insert:

```python
        if getattr(io_term, "node_anchor", "lower_left") == "pin":
            raise ValueError(
                "FtTermRef reads io_term's node-level CSR (node_idx/net_idx); a "
                "pin-anchored IoTermRef exposes a pin-level one, so the two cannot "
                "be combined (design v2 sec 7)")
```

In `FtTermRef._values`, after the existing detach-and-concatenate of `x`/`y` (lines 188-189), insert `x, y = meta._anchor_xy(x, y)`.

- [x] **Step 6: Add the flag to the CLI and the driver**

In `src/ioplace/drivers/run_placement.py`, add to `build_parser()` immediately before `return ap` (line 532):

```python
    # v2 P-F (design sec 7): the anchor at which the soft region assignment is
    # evaluated. mode=io only; no-op for the other modes.
    ap.add_argument("--node-anchor", choices=["lower_left", "center", "pin"],
                    default="center",
                    help="anchor for the soft region assignment: 'center' "
                         "(default) evaluates the SDF at x+0.5*w, y+0.5*h, "
                         "matching the freeze rule and whole-cell fence "
                         "ownership; 'lower_left' is the legacy anchor; 'pin' "
                         "is rejected by every driver -- it exists only in "
                         "IoTermRef as a small-scale bias probe")
```

and add `node_anchor=args.node_anchor,` to the `run_io(...)` call in `main()` (line 552).

In `src/ioplace/drivers/run_placement_io.py`:

1. Add `"node_anchor"` to `RESULT_FIELDS` (line 57, in the P-H block, as `# v2 P-F (design sec 7): soft-assign anchor`).
2. Add `node_anchor="center",` to `run_io`'s keyword list (after `w_mode="unit", every=50,` at line 145).
3. Insert this validation right after the `callback_order` check (line 181), so it raises before `import torch` and before any file is opened:

```python
    if node_anchor not in ("lower_left", "center", "pin"):
        raise ValueError("node_anchor must be lower_left, center or pin, got %r"
                         % (node_anchor,))
    if node_anchor == "pin":
        raise ValueError(
            "node_anchor='pin' is an IoTermRef-only bias probe (design v2 sec 7); "
            "no driver may run it -- use src/scripts/run_anchor_comparison.py")
```

4. Pass the anchor to the term (line 291-293):

```python
        io_term = IoTerm(csr=csr, rects=rects, rect2region=r2k, K=k,
                         num_movable=nl.num_movable, num_physical=nl.num_physical,
                         num_nodes=placedb.num_nodes, device="cuda", w_mode=w_mode,
                         node_anchor=node_anchor,
                         node_size_x=nl.node_size_x, node_size_y=nl.node_size_y)
```

`nl` is built at line 266 from the *initialized* PlaceDB, so `node_size_x`/`node_size_y` are already in the same scaled frame as `pos` — the units contract `anchor_offsets` depends on.

5. Add `"node_anchor": node_anchor,` to the `result` dict, next to `"w_mode": w_mode,` (line 851).

- [x] **Step 7: Run the new tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_node_anchor.py -v`
Expected: PASS — 16 passed.

- [x] **Step 8: Run every test that touches the terms or the driver**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_io_term.py tests/test_io_term_chunked.py tests/test_ft_term.py tests/test_soft_assign.py tests/test_driver_io.py tests/test_driver_t8a.py tests/test_norm_driver.py tests/test_ft_reweight.py -m "not slow" -v`
Expected: PASS. These construct `IoTerm`/`IoTermRef` without `node_anchor` and must be unaffected — the class default is `lower_left` and `_anchor_xy` short-circuits on `anchor_dx is None`, so their objective values are bit-identical, not merely close. If any of them fails, the anchor leaked into the default path; fix that rather than the test.

- [x] **Step 9: Run the full fast suite**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow"`
Expected: PASS, no new failures against the pre-task baseline. `run_io`'s default changed from the (previously implicit) lower-left anchor to `center`, so any *slow* driver test that pins an absolute `io_count` may move; record the before/after numbers in the commit message rather than re-pinning silently.

- [x] **Step 10: Commit**

```bash
git add src/ioplace/ops/soft_assign.py src/ioplace/ops/io_term.py \
        src/ioplace/ops/ft_term.py src/ioplace/drivers/run_placement.py \
        src/ioplace/drivers/run_placement_io.py tests/test_node_anchor.py
git commit -m "feat(anchor): evaluate the soft region assignment at the cell centre

design v2 sec 7: the differentiable term anchored at the node lower-left while
the evaluator counted at pins and the fence owned the whole cell, so GP
optimised a quantity the evaluator did not score. anchor_offsets() adds a
per-node constant half-size before region_sdf_l1 -- same tensor shapes, same
gradients, no new DREAMPlace hook -- behind --node-anchor {lower_left,center,pin}
with default center. 'pin' is rejected by IoTerm and by every driver.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The `pin` reference arm in `IoTermRef` and the bias measurement

**Files:**
- Modify: `src/ioplace/ops/io_term.py` — new `PinCsr`/`build_net_pin_csr` after `build_net_node_csr` (line 52), new buffers and forward branch in `IoTermRef`
- Test: `tests/test_node_anchor.py` (append)

**Interfaces:**
- Consumes: Task 1's `IoTermRef(..., node_anchor=..., pin_csr=...)` constructor and its `pin_csr is None` guard; `NetCsr` (`src/ioplace/ops/io_term.py:20-28`), whose `net_ids` is ascending (it comes from `np.unique`, line 45).
- Produces:
  - `@dataclass PinCsr(pin_node: np.ndarray, pin_offset_x: np.ndarray, pin_offset_y: np.ndarray, net_pos: np.ndarray)` — all `(P',)`, `net_pos` indexing into `csr.net_ids` (i.e. into `[0, n_active)`), so it lines up with `IoTermRef.w` and `IoTermRef.n_active`
  - `build_net_pin_csr(nl, csr) -> PinCsr`
  - `IoTermRef.node_anchor == "pin"` forward/backward; `IoTermRef.diagnostics` raises `NotImplementedError` under it

**Why the pin arm exists and why only here.** Spec §7: "The `pin` arm is implemented only in `IoTermRef`, so the bias is quantified once at small scale rather than paid for at 11M cells." Structurally it is a different term: `S_{e,k}` accumulates over the net's *pins* rather than its deduped *nodes*, so the tensor goes from `(N,K)` to `(P,K)` and a cell carrying two pins of one net is counted twice — the two costs `docs/superpowers/specs/2026-08-06-m2-differentiable-io-design.md:105` chose node anchoring to avoid. Task 7 measures what that buys and what it costs; nothing else may use it.

**Margin and diagnostics under `pin`.** `d_star` becomes per-pin, so `margin_penalty` would sum over `P'` instead of `N` and its scale would silently change. The arm therefore rejects a non-zero `lambda_margin`, and `diagnostics()` — whose `frac_soft` denominator and `grad_share` attribution are both node-based — raises rather than returning a number that looks comparable and is not.

- [x] **Step 1: Write the failing test**

Append to `tests/test_node_anchor.py`:

```python
def _pin_ref(nl, rs):
    from ioplace.ops.io_term import build_net_pin_csr
    rects, r2k = rect_table(rs)
    csr = build_net_node_csr(nl, 100)
    return IoTermRef(csr=csr, rects=rects, rect2region=r2k, K=rs.k,
                     num_movable=nl.num_movable, num_physical=nl.num_physical,
                     num_nodes=nl.num_physical, device=DEV,
                     node_anchor="pin", pin_csr=build_net_pin_csr(nl, csr))


def test_build_net_pin_csr_aligns_with_the_node_csr():
    from ioplace.ops.io_term import build_net_pin_csr
    rs, nl = _straddler()
    csr = build_net_node_csr(nl, 100)
    pc = build_net_pin_csr(nl, csr)
    assert pc.pin_node.shape == pc.net_pos.shape == (len(nl.pin2node),)
    assert pc.net_pos.min() >= 0 and pc.net_pos.max() < len(csr.net_ids)
    assert np.array_equal(csr.net_ids[pc.net_pos], nl.pin2net[: len(pc.net_pos)])


def test_build_net_pin_csr_drops_pins_of_nets_the_node_csr_dropped():
    """Degree-1 nets and nets above ignore_net_degree are absent from NetCsr;
    their pins must be absent here too or net_pos would not index w."""
    nl = _nl([(10., 10.), (90., 10.), (10., 90.)], [[0], [0, 1], [0, 1, 2]])
    nl.node_size_x = np.ones(3)
    nl.node_size_y = np.ones(3)
    from ioplace.ops.io_term import build_net_pin_csr
    csr = build_net_node_csr(nl, ignore_net_degree=3)      # keeps only net 1
    pc = build_net_pin_csr(nl, csr)
    assert len(csr.net_ids) == 1 and csr.net_ids.tolist() == [1]
    assert pc.net_pos.tolist() == [0, 0]
    assert sorted(pc.pin_node.tolist()) == [0, 1]


def test_pin_arm_equals_the_lower_left_node_arm_when_every_node_has_one_pin_at_offset_zero():
    """The strongest available correctness statement: with one zero-offset pin
    per node the pin-level and node-level accumulations are the same sum, so
    the two arms must agree to the last bit of the fp64 accumulator."""
    rng = np.random.default_rng(3)
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    xy = [(float(a), float(b)) for a, b in zip(rng.uniform(2, 98, 12),
                                               rng.uniform(2, 98, 12))]
    nets = []
    for d in (2, 3, 2, 3, 2):
        nets.append(sorted(rng.choice(12, d, replace=False).tolist()))
    nl = _nl(xy, nets)
    nl.node_size_x = np.full(12, 3.)
    nl.node_size_y = np.full(12, 2.)
    pos = _pos(nl, device=DEV)
    node_arm = _ref(nl, rs, "lower_left")(pos, 0.5, 1.0)
    pin_arm = _pin_ref(nl, rs)(pos, 0.5, 1.0)
    assert float(pin_arm.detach()) == float(node_arm.detach())
    g_node, = torch.autograd.grad(node_arm, pos, retain_graph=True)
    g_pin, = torch.autograd.grad(pin_arm, pos)
    assert torch.allclose(g_node, g_pin, rtol=1e-12, atol=1e-14)


def test_pin_arm_double_counts_a_node_carrying_two_pins_of_one_net():
    """The defect m2-differentiable-io-design.md:105 rejected, made visible:
    the node arm dedups (net, node), the pin arm does not."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    nl = _nl([(10., 10.), (90., 10.)], [[0, 1, 1]])   # node 1 carries two pins
    nl.node_size_x = np.ones(2)
    nl.node_size_y = np.ones(2)
    csr = build_net_node_csr(nl, 100)
    assert csr.degrees.tolist() == [2] and csr.pin_degrees.tolist() == [3]
    from ioplace.ops.io_term import build_net_pin_csr
    assert len(build_net_pin_csr(nl, csr).pin_node) == 3


def test_pin_arm_moves_with_the_pin_offsets_not_the_cell_corner():
    """A cell whose lower-left corner and centre are both in region 0 but whose
    one pin sits past the boundary: only the pin arm sees the crossing."""
    rs = make_grid_regions(DIE, 2, 2, lattice=10)
    nl = _nl([(10., 10.), (44., 10.)], [[0, 1]])
    nl.node_size_x = np.array([2., 2.])
    nl.node_size_y = np.array([2., 2.])
    nl.pin_offset_x = np.array([0., 10.])            # pin of node 1 at x = 54
    pos = _pos(nl, device=DEV)
    assert float(_ref(nl, rs, "lower_left")(pos, 0.05, 1.0).detach()) == pytest.approx(0., abs=1e-9)
    assert float(_ref(nl, rs, "center")(pos, 0.05, 1.0).detach()) == pytest.approx(0., abs=1e-9)
    assert float(_pin_ref(nl, rs)(pos, 0.05, 1.0).detach()) == pytest.approx(1., abs=1e-9)


def test_pin_arm_rejects_a_margin_and_diagnostics():
    rs, nl = _straddler()
    pos = _pos(nl, device=DEV)
    term = _pin_ref(nl, rs)
    with pytest.raises(ValueError, match="lambda_margin"):
        term(pos, 0.3, 1.0, lambda_margin=1.0, margin_m=1.0)
    with pytest.raises(NotImplementedError, match="pin"):
        term.diagnostics(pos.detach(), 0.3)


def test_ft_term_ref_rejects_a_pin_anchored_io_term():
    from ioplace.ops.ft_term import FtTermRef
    rs, nl = _straddler()
    with pytest.raises(ValueError, match="pin"):
        FtTermRef(_pin_ref(nl, rs), np.zeros((4, 4)))
```

- [x] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_node_anchor.py -k pin -v`
Expected: FAIL — `ImportError: cannot import name 'build_net_pin_csr' from 'ioplace.ops.io_term'`.

- [x] **Step 3: Add `PinCsr` and `build_net_pin_csr`**

In `src/ioplace/ops/io_term.py`, after `build_net_node_csr` (line 52):

```python
@dataclass
class PinCsr:
    """Pin-level companion to NetCsr, for IoTermRef's `pin` anchor only
    (design v2 sec 7). `net_pos` indexes NetCsr.net_ids, not the raw net id,
    so it lines up with IoTermRef.w / n_active without a second lookup."""
    pin_node: np.ndarray
    pin_offset_x: np.ndarray
    pin_offset_y: np.ndarray
    net_pos: np.ndarray


def build_net_pin_csr(nl, csr):
    """Every pin of every net NetCsr kept, in the netlist's own pin order.

    Nets NetCsr dropped -- degree < 2, degree >= ignore_net_degree, or collapsed
    to a single node -- contribute no pins, or `net_pos` would not be a valid
    index into `w`. NetCsr.net_ids is ascending (np.unique), so membership is a
    searchsorted, not a hash set.
    """
    net_of_pin = np.asarray(nl.pin2net, dtype=np.int64)
    if len(csr.net_ids) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return PinCsr(empty_i, empty_f, empty_f, empty_i)
    pos = np.searchsorted(csr.net_ids, net_of_pin)
    pos = np.clip(pos, 0, len(csr.net_ids) - 1)
    keep = csr.net_ids[pos] == net_of_pin
    return PinCsr(np.asarray(nl.pin2node, dtype=np.int64)[keep],
                  np.asarray(nl.pin_offset_x, dtype=np.float64)[keep],
                  np.asarray(nl.pin_offset_y, dtype=np.float64)[keep],
                  pos[keep])
```

- [x] **Step 4: Register the pin buffers**

In `IoTermRef.__init__`, replace the `dx = dy = None` line inside the `if node_anchor == "pin":` branch Task 1 added with:

```python
            if pin_csr is None:
                raise ValueError("node_anchor='pin' requires pin_csr "
                                 "(ioplace.ops.io_term.build_net_pin_csr)")
            self.register_buffer("pin_node", torch.as_tensor(pin_csr.pin_node,
                                                             dtype=torch.int64, device=device))
            self.register_buffer("pin_dx", torch.as_tensor(pin_csr.pin_offset_x,
                                                           dtype=torch.float64, device=device))
            self.register_buffer("pin_dy", torch.as_tensor(pin_csr.pin_offset_y,
                                                           dtype=torch.float64, device=device))
            self.register_buffer("pin_net_idx", torch.as_tensor(pin_csr.net_pos,
                                                                dtype=torch.int64, device=device))
            dx = dy = None
```

- [x] **Step 5: Add the pin forward branch**

Replace `IoTermRef._forward_io` (lines 112-120) with:

```python
    def _forward_io(self, x, y, tau):
        """-> (L_io scalar, lam (n_active,), d_star (M,)) where M is N for the
        node anchors and P' for the pin anchor."""
        if self.node_anchor == "pin":
            ax = x[self.pin_node] + self.pin_dx.to(dtype=x.dtype)
            ay = y[self.pin_node] + self.pin_dy.to(dtype=y.dtype)
            unit_idx = self.pin_net_idx
        else:
            ax, ay, unit_idx = x, y, None
        m, t, am = softmax_stats(ax, ay, self.rects, self.rect2region, self.K, tau)
        sdf = region_sdf_l1(ax, ay, self.rects, self.rect2region, 0, self.K)
        p, ell = chunk_p_ell(sdf, m, t, am, 0, tau)
        if unit_idx is None:
            unit_idx, contrib = self.net_idx, ell[self.node_idx]
        else:
            contrib = ell
        S = torch.zeros((self.n_active, self.K), dtype=torch.float64,
                        device=x.device).index_add_(0, unit_idx, contrib.double())
        lam = (1.0 - torch.exp(S)).sum(dim=1)
        return (self.w * (lam - 1.0).clamp(min=0)).sum(), lam, d_star_from_m(m, tau)
```

In `IoTermRef.forward` (lines 122-126), reject the margin under the pin anchor before anything else:

```python
    def forward(self, pos, tau, lambda_io, lambda_margin=0.0, margin_m=0.0, margin_tau=1.0):
        if self.node_anchor == "pin" and lambda_margin != 0.0:
            raise ValueError(
                "the pin anchor makes d_star per-pin, so lambda_margin would "
                "silently change the margin term's scale; pass lambda_margin=0 "
                "(design v2 sec 7: the pin arm is an L_IO bias probe only)")
        x, y = self._split_xy(pos)
        L_io, lam, d_star = self._forward_io(x, y, tau)
        L_margin = margin_penalty(d_star, margin_m, margin_tau)
        return lambda_io * L_io + lambda_margin * L_margin
```

At the top of `IoTermRef.diagnostics` (line 136), insert:

```python
        if self.node_anchor == "pin":
            raise NotImplementedError(
                "frac_soft's denominator and grad_share's attribution are both "
                "node-based; under the pin anchor they would return numbers that "
                "look comparable with the node arms and are not (design v2 sec 7)")
```

- [x] **Step 6: Run the new tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_node_anchor.py -v`
Expected: PASS — 23 passed (Task 1's 16 plus these 7).

- [x] **Step 7: Run the term suites**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_io_term.py tests/test_io_term_chunked.py tests/test_ft_term.py -m "not slow" -v`
Expected: PASS, unchanged counts.

- [x] **Step 8: Commit**

```bash
git add src/ioplace/ops/io_term.py tests/test_node_anchor.py
git commit -m "feat(anchor): IoTermRef-only pin arm for the sec 7 bias measurement

build_net_pin_csr re-indexes the accumulation from deduped nodes to pins, so the
(P,K) cost and the multi-pin double counting m2-differentiable-io-design.md:105
rejected are paid once, at small scale, to measure the bias -- never in
production. With one zero-offset pin per node the pin arm reproduces the node
arm bit-for-bit; the margin term and diagnostics() are refused under it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `straddle.py` — the numpy reference and `evaluator_ref` integration

**Files:**
- Create: `src/ioplace/straddle.py`
- Modify: `src/ioplace/evaluator_ref.py:6-25` (`EvalResult`), `:83-172` (`evaluate`)
- Test: `tests/test_straddle.py` (new)

**Interfaces:**
- Consumes: `ioplace.region_grid.RegionGrid.region_of_points`/`to_idx`/`grid`/`die`/`cell_w`/`cell_h`/`k` (`src/ioplace/region_grid.py:25-40`); `ioplace.netlist.pin_positions` (`src/ioplace/netlist.py:26-29`); `Netlist.node_size_x`/`node_size_y`/`num_movable`/`num_physical`/`num_nets`/`net_degrees`/`pin2node`/`pin2net`.
- Produces:
  - `@dataclass StraddleStats(straddle_cells: int, straddle_area_fraction: float, straddle_pin_split_nets: int, straddle_out_area: float, straddle_movable_area: float, straddle_wide_cells: int, per_node_straddle: np.ndarray, per_net_pin_split: np.ndarray)`
  - `STRADDLE_SCALARS: tuple` and `STRADDLE_ARRAYS: tuple` — the field-name contracts `export/evaluation.py` and the drivers quote
  - `straddle_geometry(rg, x, y, w, h) -> (straddle, owner, out_area, wide)`, all `(M,)`
  - `distinct_regions_per_net(nl, pin_rid) -> np.ndarray` `(num_nets,)` int64
  - `straddle_diagnostics(nl, node_x, node_y, rg, *, pin_rid=None) -> StraddleStats`
  - `EvalResult` gains the six scalars (defaults `0`/`0.0`) and `per_node_straddle=None`, `per_net_pin_split=None`
  - `evaluator_ref.evaluate(nl, node_x, node_y, rg, max_degree=256, *, route_wirelength_budget=None, straddle=True)`

**The five conventions, fixed once.** They live in `straddle.py`'s module docstring because `evaluator_gpu` must mirror them bit-for-bit (Task 4) and a convention that lives in two places drifts:

1. **Closed box, four corners.** A movable cell occupies `[x, x+w] x [y, y+h]` and is judged by the region ids of its four corners, exactly as spec §7 prescribes. The box is closed, so a cell whose right edge lands *exactly* on a region boundary counts as straddling. Rejected: a half-open `[x, x+w)` box, which needs an epsilon that `RegionGrid.to_idx`'s truncate-and-clamp cannot express identically in numpy and torch. The closed box over-counts in the conservative direction, which is the right direction for a diagnostic.
2. **Owner = the cell centre's region.** `region_of_points(x + w/2, y + h/2)` — the freeze membership (spec §3 phase 2), the new soft-assign anchor (§7) and whole-cell fence ownership all agree on it, which is the whole point of P-F. Rejected: maximum-overlap ownership, which is more expensive and disagrees with the freeze.
3. **Movable cells only**, `i < nl.num_movable`. Terminals do not move and are not fence-assigned. `per_node_straddle` is still shaped `(num_physical,)`, zero on the tail, so it aligns with `node_region` in `evaluation.npz`.
4. **Quadrant area split.** Out-of-owner area is accumulated over the (at most) four rectangles the *first* lattice line in each axis cuts the box into, each attributed to its own corner's region. The four areas sum to `w*h` exactly. For a cell spanning at most two lattice cells per axis — every standard cell on a 512-lattice — this is exact. Wider cells are approximated and counted in `straddle_wide_cells`, so the approximation is measured, not silent. Rejected: exact rasterisation over the full lattice span, which needs the pow2 bucketing machinery `_process_segments` carries and buys nothing for a diagnostic whose straddle test is already four-corner.
5. **Pin re-attribution by distinct-region count.** `straddle_pin_split_nets` counts nets whose number of distinct pin regions *drops* when every pin of a straddling cell moves to that cell's owner. `per_net_lambda` is the only crossing measure that is a function of pin→region attribution alone: the MST-geometry `per_net_crossings` is a function of coordinates, which re-attribution does not change, and `per_net_steiner` would need the whole Λ≥4 Steiner pass re-run (`evaluator_gpu.py:634-642`) for no extra information. `λ_e − 1` is exactly the per-net crossing lower bound realised after fence LG, when no cell straddles — which is why it is the right target for "the direct attribution of `lg_loss`". `per_net_pin_split` is kept *signed*: re-attribution can also raise a net's λ, and that number is evidence too.

- [x] **Step 1: Write the failing test**

Create `tests/test_straddle.py`:

```python
import numpy as np
import pytest

from ioplace.evaluator_ref import evaluate
from ioplace.netlist import Netlist
from ioplace.region_grid import RegionGrid
from ioplace.regions import make_grid_regions
from ioplace.straddle import (STRADDLE_ARRAYS, STRADDLE_SCALARS,
                              distinct_regions_per_net, straddle_diagnostics,
                              straddle_geometry)

DIE = (0., 0., 100., 100.)


def _grid():
    """2x2 grid on a 10x10 lattice: cells are 10x10, region boundaries at
    x=50 and y=50, regions numbered row-major (P0 lower-left, P1 lower-right,
    P2 upper-left, P3 upper-right)."""
    return RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))


def _corner_case():
    """Four movable-plus-one-fixed layout with every interesting case present.

      node 0 (10,10) 4x4  -- wholly inside region 0
      node 1 (48,10) 4x4  -- crosses x=50 only; centre (50,12) -> owner 1
      node 2 (48,48) 4x4  -- crosses both lines; corners hit all four regions,
                             centre (50,50) -> owner 3
      node 3 (90,90) 4x4  -- FIXED (num_movable = 3), must be ignored

    Nets, all zero pin offsets, so pin positions are the cells' lower-left
    corners and every pin of nodes 1 and 2 sits in region 0 before
    re-attribution:
      net 0 {0,1}: {0,0} -> lambda 1;  re {0,1} -> lambda 2   (split rises)
      net 1 {1,3}: {0,3} -> lambda 2;  re {1,3} -> lambda 2   (no change)
      net 2 {2,3}: {0,3} -> lambda 2;  re {3,3} -> lambda 1   (split drops)
      net 3 {0,2}: {0,0} -> lambda 1;  re {0,3} -> lambda 2   (split rises)
    """
    pin2node = np.array([0, 1, 1, 3, 2, 3, 0, 2], np.int32)
    pin2net = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int32)
    return Netlist(node_x=np.array([10., 48., 48., 90.]),
                   node_y=np.array([10., 10., 48., 90.]),
                   node_size_x=np.full(4, 4.), node_size_y=np.full(4, 4.),
                   num_movable=3, num_terminals=1, num_terminal_NIs=0,
                   pin_offset_x=np.zeros(8), pin_offset_y=np.zeros(8),
                   pin2node=pin2node, pin2net=pin2net,
                   flat_net2pin=np.arange(8, dtype=np.int32),
                   flat_net2pin_start=np.array([0, 2, 4, 6, 8], np.int32),
                   xl=0., yl=0., xh=100., yh=100.)


def test_field_name_contracts():
    assert STRADDLE_SCALARS == ("straddle_cells", "straddle_area_fraction",
                                "straddle_pin_split_nets", "straddle_out_area",
                                "straddle_movable_area", "straddle_wide_cells")
    assert STRADDLE_ARRAYS == ("per_node_straddle", "per_net_pin_split")


def test_geometry_reads_the_four_corners_and_the_centre_owner():
    rg = _grid()
    x = np.array([10., 48., 48.])
    y = np.array([10., 10., 48.])
    w = np.full(3, 4.)
    h = np.full(3, 4.)
    straddle, owner, out_area, wide = straddle_geometry(rg, x, y, w, h)
    assert straddle.tolist() == [False, True, True]
    assert owner.tolist() == [0, 1, 3]
    # node 1: the 2x4 strip left of x=50 is region 0, the owner is 1 -> 8.
    # node 2: three of the four 2x2 quadrants are not region 3 -> 12.
    assert out_area.tolist() == [0., 8., 12.]
    assert wide.tolist() == [False, False, False]


def test_quadrant_areas_always_sum_to_the_cell_area():
    rg = _grid()
    rng = np.random.default_rng(0)
    x = rng.uniform(0., 95.,  400)
    y = rng.uniform(0., 95.,  400)
    w = rng.uniform(0.5, 4.5, 400)
    h = rng.uniform(0.5, 4.5, 400)
    _, owner, out_area, _ = straddle_geometry(rg, x, y, w, h)
    # out_area is a sub-sum of the same four quadrants, so it can never exceed
    # the cell area and is zero exactly when every quadrant belongs to the owner
    assert np.all(out_area <= w * h + 1e-9)
    assert np.all(out_area >= 0.)


def test_wide_cells_are_flagged_and_approximated_not_silently_wrong():
    """A cell spanning four lattice columns: the quadrant split attributes the
    whole 25-wide right part to the right corner's region, so it reports 100
    where the exact out-of-owner area is 60. The point of straddle_wide_cells
    is that this is counted, not hidden."""
    rg = _grid()
    straddle, owner, out_area, wide = straddle_geometry(
        rg, np.array([30.]), np.array([10.]), np.array([35.]), np.array([4.]))
    assert straddle.tolist() == [True] and owner.tolist() == [0]
    assert wide.tolist() == [True]
    assert out_area.tolist() == [100.]


def test_cell_touching_a_boundary_exactly_counts_as_straddling():
    """Closed-box convention: the right edge landing exactly on x=50 puts the
    (x+w) corner in region 1."""
    rg = _grid()
    straddle, owner, out_area, _ = straddle_geometry(
        rg, np.array([46.]), np.array([10.]), np.array([4.]), np.array([4.]))
    assert straddle.tolist() == [True] and owner.tolist() == [0]
    assert out_area.tolist() == [0.]     # every quadrant with area is region 0


def test_distinct_regions_per_net_is_zero_for_degree_one_nets():
    nl = _corner_case()
    rg = _grid()
    px, py = nl.node_x[nl.pin2node], nl.node_y[nl.pin2node]
    lam = distinct_regions_per_net(nl, rg.region_of_points(px, py))
    assert lam.tolist() == [1, 2, 2, 1]


def test_diagnostics_on_the_corner_case():
    nl = _corner_case()
    st = straddle_diagnostics(nl, nl.node_x, nl.node_y, _grid())
    assert st.straddle_cells == 2
    assert st.straddle_out_area == pytest.approx(20.0)
    assert st.straddle_movable_area == pytest.approx(48.0)      # 3 movable 4x4 cells
    assert st.straddle_area_fraction == pytest.approx(20.0 / 48.0)
    assert st.straddle_wide_cells == 0
    assert st.per_node_straddle.tolist() == [0, 1, 1, 0]         # the terminal is excluded
    assert st.per_net_pin_split.tolist() == [-1, 0, 1, -1]
    assert st.straddle_pin_split_nets == 1


def test_diagnostics_are_all_zero_when_nothing_straddles():
    nl = _corner_case()
    nl.node_x = np.array([10., 10., 60., 90.])
    nl.node_y = np.array([10., 20., 60., 90.])
    st = straddle_diagnostics(nl, nl.node_x, nl.node_y, _grid())
    assert st.straddle_cells == 0
    assert st.straddle_area_fraction == 0.0
    assert st.straddle_pin_split_nets == 0
    assert st.per_net_pin_split.tolist() == [0, 0, 0, 0]


def test_pin_offsets_move_the_attribution_not_the_geometry():
    """A cell wholly inside region 0 whose pin offset puts its pin in region 1
    does not straddle, so its pins are not re-attributed."""
    nl = _corner_case()
    nl.pin_offset_x = np.array([0., 0., 0., 0., 0., 0., 40., 0.])   # pin 6 (node 0)
    st = straddle_diagnostics(nl, nl.node_x, nl.node_y, _grid())
    assert st.per_node_straddle[0] == 0
    assert st.straddle_cells == 2


def test_evaluate_reports_the_diagnostics_and_leaves_the_legacy_fields_alone():
    nl = _corner_case()
    rg = _grid()
    off = evaluate(nl, nl.node_x, nl.node_y, rg, straddle=False)
    on = evaluate(nl, nl.node_x, nl.node_y, rg)
    for field in ("io_count", "ft_count", "hard_lambda_sum", "io_rg", "ft_rg",
                  "large_net_lb", "tree_wl", "hpwl", "boundary_pair_demand"):
        assert getattr(on, field) == getattr(off, field)
    assert np.array_equal(on.per_net_lambda, off.per_net_lambda)
    assert off.straddle_cells == 0 and off.per_node_straddle is None
    assert on.straddle_cells == 2
    assert on.straddle_pin_split_nets == 1
    assert on.straddle_area_fraction == pytest.approx(20.0 / 48.0)
    assert on.per_node_straddle.dtype == np.uint8
    assert on.per_net_pin_split.dtype == np.int32


def test_lambda_matches_the_evaluators_own_per_net_lambda():
    """distinct_regions_per_net must agree with the loop evaluator_ref already
    runs, or per_net_pin_split would be measured against a different baseline."""
    nl = _corner_case()
    rg = _grid()
    res = evaluate(nl, nl.node_x, nl.node_y, rg)
    px, py = nl.node_x[nl.pin2node], nl.node_y[nl.pin2node]
    assert distinct_regions_per_net(nl, rg.region_of_points(px, py)).tolist() \
        == res.per_net_lambda.tolist()
```

- [x] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_straddle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.straddle'`.

- [x] **Step 3: Write `src/ioplace/straddle.py`**

```python
"""Straddling diagnostics (design v2 sec 7, diagnostics 1-3).

This is the numpy reference. `evaluator_ref.evaluate` calls it;
`evaluator_gpu.GpuEvalContext` mirrors it in torch and
`tests/test_evaluator_gpu.py` pins the two together -- every integer field
bit-exact, the float fields at rel <= 1e-12, across batch sizes and chunk
budgets (spec sec 9).

Five conventions, fixed here because a convention that lives in two places
drifts:

1. Closed box, four corners. A movable cell occupies [x, x+w] x [y, y+h] and is
   judged by the region ids of its four corners, as sec 7 prescribes. The box is
   closed, so a cell whose right edge lands exactly on a region boundary counts
   as straddling -- the conservative direction, and the only convention
   RegionGrid.to_idx's truncate-and-clamp reproduces identically in numpy and
   torch without an epsilon.
2. Owner = the cell centre's region. That is the freeze membership (sec 3 phase
   2), the soft-assign anchor (sec 7) and whole-cell fence ownership, all three
   agreeing -- which is the point of P-F.
3. Movable cells only (i < nl.num_movable). per_node_straddle is still shaped
   (num_physical,), zero on the terminal tail, so it aligns with
   evaluation.npz's node_region.
4. Quadrant area split. Out-of-owner area is summed over the at most four
   rectangles the FIRST lattice line in each axis cuts the box into, each
   attributed to its own corner's region; the four areas sum to w*h exactly.
   Exact for a cell spanning at most two lattice cells per axis (every standard
   cell on a 512-lattice); wider cells are approximated and counted in
   straddle_wide_cells, so the approximation is measured, not silent.
5. Pin re-attribution by distinct-region count. per_net_lambda is the only
   crossing measure that is a function of pin->region attribution alone: the
   MST-geometry per_net_crossings depends on coordinates, which re-attribution
   does not change, and per_net_steiner would need the whole Lambda>=4 Steiner
   pass re-run for no extra information. lambda_e - 1 is exactly the per-net
   crossing lower bound realised after fence LG, when no cell straddles.
   per_net_pin_split is signed: re-attribution can also raise a net's lambda.
"""
from dataclasses import dataclass

import numpy as np

from ioplace.netlist import pin_positions

STRADDLE_SCALARS = ("straddle_cells", "straddle_area_fraction",
                    "straddle_pin_split_nets", "straddle_out_area",
                    "straddle_movable_area", "straddle_wide_cells")
STRADDLE_ARRAYS = ("per_node_straddle", "per_net_pin_split")

# (net_id, region_id) are packed into one int64 as net*_REGION_STRIDE + region,
# the same packing evaluator_gpu.py:575 uses for pin_bm -- keep the two equal so
# the CPU and GPU unique/bincount passes are the same arithmetic.
_REGION_STRIDE = np.int64(64)


@dataclass
class StraddleStats:
    straddle_cells: int
    straddle_area_fraction: float
    straddle_pin_split_nets: int
    straddle_out_area: float
    straddle_movable_area: float
    straddle_wide_cells: int
    per_node_straddle: np.ndarray        # (num_physical,) uint8
    per_net_pin_split: np.ndarray        # (num_nets,) int32, signed


def straddle_geometry(rg, x, y, w, h):
    """(M,) lower-left positions and sizes -> (straddle, owner, out_area, wide).

    straddle: bool, the four corners do not all agree.
    owner:    int64, the region of the cell centre.
    out_area: float64, quadrant area not belonging to the owner.
    wide:     bool, the box spans more than two lattice cells in some axis, so
              out_area is approximate (convention 4).
    """
    xr = x + w
    yt = y + h
    r00 = rg.region_of_points(x, y).astype(np.int64)
    r10 = rg.region_of_points(xr, y).astype(np.int64)
    r01 = rg.region_of_points(x, yt).astype(np.int64)
    r11 = rg.region_of_points(xr, yt).astype(np.int64)
    owner = rg.region_of_points(x + 0.5 * w, y + 0.5 * h).astype(np.int64)
    straddle = (r10 != r00) | (r01 != r00) | (r11 != r00)

    ix0, iy0 = rg.to_idx(x, y)
    ix1, iy1 = rg.to_idx(xr, yt)
    xm = np.minimum(xr, rg.die[0] + (ix0 + 1) * rg.cell_w)
    ym = np.minimum(yt, rg.die[1] + (iy0 + 1) * rg.cell_h)
    lw = np.maximum(xm - x, 0.0)
    rw = np.maximum(xr - xm, 0.0)
    bh = np.maximum(ym - y, 0.0)
    th = np.maximum(yt - ym, 0.0)
    out_area = (lw * bh * (r00 != owner) + rw * bh * (r10 != owner)
                + lw * th * (r01 != owner) + rw * th * (r11 != owner))
    wide = ((ix1 - ix0) > 1) | ((iy1 - iy0) > 1)
    return straddle, owner, out_area, wide


def distinct_regions_per_net(nl, pin_rid):
    """(num_nets,) int64 count of distinct regions each net's pins touch, 0 for
    degree < 2 nets -- the same convention EvalResult.per_net_lambda uses."""
    net_of_pin = np.asarray(nl.pin2net, dtype=np.int64)
    key = np.unique(net_of_pin * _REGION_STRIDE
                    + np.asarray(pin_rid, dtype=np.int64))
    counts = np.bincount(key // _REGION_STRIDE,
                         minlength=nl.num_nets).astype(np.int64)
    return np.where(np.asarray(nl.net_degrees, dtype=np.int64) >= 2, counts, 0)


def straddle_diagnostics(nl, node_x, node_y, rg, *, pin_rid=None):
    """Design v2 sec 7 diagnostics 1-3. `pin_rid` lets a caller that has already
    computed the per-pin region ids (evaluator_ref.evaluate has) pass them in
    rather than paying for a second region_of_points over every pin."""
    if rg.k > int(_REGION_STRIDE):
        raise ValueError("straddle diagnostics pack (net, region) into one "
                         "int64 with stride %d; got k=%d"
                         % (int(_REGION_STRIDE), rg.k))
    m = int(nl.num_movable)
    n_physical = int(nl.num_physical)
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)
    w = np.asarray(nl.node_size_x, dtype=np.float64)[:m]
    h = np.asarray(nl.node_size_y, dtype=np.float64)[:m]
    straddle, owner, out_area, wide = straddle_geometry(
        rg, node_x[:m], node_y[:m], w, h)

    total_area = float((w * h).sum())
    out_total = float(out_area.sum())
    per_node = np.zeros(n_physical, dtype=np.uint8)
    per_node[:m] = straddle.astype(np.uint8)

    if pin_rid is None:
        px, py = pin_positions(nl, node_x, node_y)
        pin_rid = rg.region_of_points(px, py)
    pin_rid = np.asarray(pin_rid, dtype=np.int64)
    node_of_pin = np.asarray(nl.pin2node, dtype=np.int64)
    owner_full = np.zeros(n_physical, dtype=np.int64)
    owner_full[:m] = owner
    pin_rid_re = np.where(per_node[node_of_pin].astype(bool),
                          owner_full[node_of_pin], pin_rid)

    split = (distinct_regions_per_net(nl, pin_rid)
             - distinct_regions_per_net(nl, pin_rid_re)).astype(np.int32)
    return StraddleStats(
        straddle_cells=int(straddle.sum()),
        straddle_area_fraction=(out_total / total_area) if total_area > 0.0 else 0.0,
        straddle_pin_split_nets=int((split > 0).sum()),
        straddle_out_area=out_total,
        straddle_movable_area=total_area,
        straddle_wide_cells=int(wide.sum()),
        per_node_straddle=per_node,
        per_net_pin_split=split)
```

- [x] **Step 4: Extend `EvalResult` and `evaluate`**

In `src/ioplace/evaluator_ref.py`, append to the `EvalResult` dataclass (after `per_net_home`, line 25):

```python
    # v2 P-F (design v2 sec 7 diagnostics 1-3). Defaults are the "not computed"
    # state, which evaluate(straddle=False) and any pre-P-F caller both land on.
    straddle_cells: int = 0
    straddle_area_fraction: float = 0.0
    straddle_pin_split_nets: int = 0
    straddle_out_area: float = 0.0
    straddle_movable_area: float = 0.0
    straddle_wide_cells: int = 0
    per_node_straddle: np.ndarray = None   # (num_physical,) uint8
    per_net_pin_split: np.ndarray = None   # (num_nets,) int32, signed
```

Add `from ioplace.straddle import straddle_diagnostics` to the imports (line 4).

Change `evaluate`'s signature (line 83) to
`def evaluate(nl, node_x, node_y, rg, max_degree=256, *, route_wirelength_budget=None, straddle=True):`
and add to its docstring: `"straddle=False skips the sec 7 diagnostics; the legacy fields are bit-identical either way."`

Immediately before the `return EvalResult(...)` (line 163), insert:

```python
    st = straddle_diagnostics(nl, node_x, node_y, rg,
                              pin_rid=pin_rid_all) if straddle else None
```

and add to the `EvalResult(...)` call:

```python
        straddle_cells=st.straddle_cells if st else 0,
        straddle_area_fraction=st.straddle_area_fraction if st else 0.0,
        straddle_pin_split_nets=st.straddle_pin_split_nets if st else 0,
        straddle_out_area=st.straddle_out_area if st else 0.0,
        straddle_movable_area=st.straddle_movable_area if st else 0.0,
        straddle_wide_cells=st.straddle_wide_cells if st else 0,
        per_node_straddle=st.per_node_straddle if st else None,
        per_net_pin_split=st.per_net_pin_split if st else None,
```

`pin_rid_all` is already computed at line 113 from the same `px`/`py`, so no pin position is gathered twice.

- [x] **Step 5: Run the new tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_straddle.py -v`
Expected: PASS — 11 passed.

- [x] **Step 6: Run the reference-evaluator suite**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_ref.py tests/test_region_grid.py tests/test_budgeted_routing.py tests/test_route_crossings_s3.py -v`
Expected: PASS, unchanged counts. `EvalResult` gained only defaulted fields, so every positional construction elsewhere still works.

- [x] **Step 7: Commit**

```bash
git add src/ioplace/straddle.py src/ioplace/evaluator_ref.py tests/test_straddle.py
git commit -m "feat(eval): straddle_cells / straddle_area_fraction / straddle_pin_split_nets

design v2 sec 7 diagnostics 1-3, as a numpy reference whose module docstring
fixes the five conventions evaluator_gpu must mirror: closed four-corner box,
cell-centre owner, movable-only, quadrant area split with wide cells counted
rather than hidden, and pin re-attribution scored by distinct-region count --
the only crossing measure that is a function of attribution alone.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The `evaluator_gpu` mirror and bit-exact parity

**Files:**
- Modify: `src/ioplace/evaluator_gpu.py:93-283` (`__init__`), `:284-306` (helpers), `:531-769` (`evaluate`), `:772-773` (`evaluate_gpu`)
- Test: `tests/test_evaluator_gpu.py` (append + extend `_assert_batch_invariant_fields`)

**Interfaces:**
- Consumes: Task 3's `StraddleStats` field names and the five conventions in `src/ioplace/straddle.py`'s module docstring; `GpuEvalContext._to_idx` (`src/ioplace/evaluator_gpu.py:287-301`) and the `_cell_w_t`/`_cell_h_t` 0-dim tensors (`:178-180`) — the anti-mis-rounding contract from C1 applies to the corner lookups exactly as it does to pin lookups.
- Produces:
  - `GpuEvalContext(nl, rg, device="cuda", max_degree=256, mst_chunk_budget=..., seg_chunk_budget=..., edge_batch_size=..., straddle=True)` with `.straddle`, `.num_movable`, `.num_physical`
  - `GpuEvalContext.evaluate(node_x, node_y, *, straddle=None)`
  - `evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda", straddle=True)`

**Why the mirror can be bit-exact on the integers.** Every integer the diagnostics produce comes from `_to_idx` + a `grid_t` gather + integer comparisons, which `evaluator_gpu.py:10-26` already argues are bit-identical between numpy float64 and torch float64 as long as the lattice division goes through the 0-dim `_cell_w_t`/`_cell_h_t` tensors rather than python floats (the C1 reciprocal-multiply hazard). The new lattice-line coordinate `xl + (ix0 + 1) * cell_w` must use the same tensors for the same reason. The float fields are reductions and therefore only carry the `rel <= 1e-12` contract.

- [x] **Step 1: Write the failing test**

Append to `tests/test_evaluator_gpu.py`:

```python
# ---------------------------------------------------------------------------
# v2 P-F (design sec 7 / spec sec 9): straddle diagnostics. The three integer
# fields and the two integer arrays are bit-exact ref-vs-GPU and across batch
# sizes / chunk budgets; the three float fields carry tree_wl/hpwl's rel<=1e-12
# contract, because they are float64 reductions whose order numpy and torch do
# not share.
# ---------------------------------------------------------------------------

_STRADDLE_INT_SCALARS = ("straddle_cells", "straddle_pin_split_nets",
                         "straddle_wide_cells")
_STRADDLE_FLOAT_SCALARS = ("straddle_area_fraction", "straddle_out_area",
                           "straddle_movable_area")


def _assert_straddle_equal(a, b, float_rel=1e-12):
    for field in _STRADDLE_INT_SCALARS:
        assert getattr(a, field) == getattr(b, field), field
    for field in _STRADDLE_FLOAT_SCALARS:
        assert getattr(a, field) == pytest.approx(getattr(b, field), rel=float_rel), field
    assert np.array_equal(a.per_node_straddle, b.per_node_straddle)
    assert np.array_equal(a.per_net_pin_split, b.per_net_pin_split)


def _sized_case(rng, n_cells=60, n_nets=40, max_d=10, size=6.0):
    """_random_case with cells big enough (6.0 on a 100-wide die cut at
    25/50/75) that a solid fraction of them straddles -- the default unit-size
    cells only straddle by accident."""
    nl = _random_case(rng, n_cells=n_cells, n_nets=n_nets, max_d=max_d)
    nl.node_size_x = np.full(n_cells, size)
    nl.node_size_y = np.full(n_cells, size)
    return nl


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_gpu_straddle_matches_reference(seed):
    rng = np.random.default_rng(seed)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.straddle_cells > 0          # the case actually exercises the path
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_on_the_hand_built_corner_case():
    """The same layout tests/test_straddle.py pins by hand, so ref and GPU are
    both nailed to known numbers rather than only to each other."""
    from tests.test_straddle import _corner_case
    nl = _corner_case()
    rg = RegionGrid(make_grid_regions(DIE, 2, 2, lattice=10))
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.straddle_cells == 2 and ref.straddle_pin_split_nets == 1
    assert ref.per_net_pin_split.tolist() == [-1, 0, 1, -1]
    assert ref.straddle_area_fraction == pytest.approx(20.0 / 48.0)
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_with_fixed_nodes_and_pin_offsets():
    """num_movable < num_physical and non-zero pin offsets together: the
    terminal tail must stay out of both the geometry and the re-attribution."""
    rng = np.random.default_rng(7)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng, n_cells=50, n_nets=30)
    nl.num_movable = 35
    nl.num_terminals = 15
    nl.pin_offset_x = rng.uniform(0., 6., len(nl.pin2node))
    nl.pin_offset_y = rng.uniform(0., 6., len(nl.pin2node))
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.per_node_straddle[35:].sum() == 0
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_matches_reference_on_lattice_boundaries():
    """C1 again, for the corner lookups: the lattice-line coordinate
    xl + (ix0+1)*cell_w must be built from the 0-dim cell-size tensors, or the
    quadrant split lands one ULP off exactly at a region boundary."""
    from ioplace.netlist import Netlist
    DIE_ND = (0., 0., 10692., 10680.)
    rg = RegionGrid(make_grid_regions(DIE_ND, 4, 4, lattice=512))
    # cell_w = 10692/512 = 20.8828125; region columns break at lattice 128/256/384,
    # i.e. x = 2673.0 / 5346.0 / 8019.0. Each cell straddles one of them.
    node_x = np.array([2670.0, 5340.0, 8010.0, 1000.0])
    node_y = np.array([5000.0, 5000.0, 5000.0, 5000.0])
    nl = Netlist(node_x=node_x, node_y=node_y,
                 node_size_x=np.full(4, 20.0), node_size_y=np.full(4, 20.0),
                 num_movable=4, num_terminals=0, num_terminal_NIs=0,
                 pin_offset_x=np.zeros(4), pin_offset_y=np.zeros(4),
                 pin2node=np.array([0, 1, 2, 3], np.int32),
                 pin2net=np.array([0, 0, 1, 1], np.int32),
                 flat_net2pin=np.arange(4, dtype=np.int32),
                 flat_net2pin_start=np.array([0, 2, 4], np.int32),
                 xl=0., yl=0., xh=10692., yh=10680.)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    assert ref.straddle_cells == 3
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_bit_exact_across_batch_sizes_and_chunk_budgets():
    from ioplace.evaluator_gpu import GpuEvalContext
    rg = RegionGrid(make_grid_regions(DIE, 8, 4, lattice=32))
    rng = np.random.default_rng(123)
    nl = _lambda_ge4_netlist(rng, rg.k)
    nl.node_size_x = np.full(len(nl.node_x), 5.0)
    nl.node_size_y = np.full(len(nl.node_x), 5.0)
    ref_ctx = GpuEvalContext(nl, rg, device="cuda", mst_chunk_budget=10**9,
                             seg_chunk_budget=10**9, edge_batch_size=10**9)
    ref = ref_ctx.evaluate(nl.node_x, nl.node_y)
    assert ref.straddle_cells > 0
    cpu = evaluate(nl, nl.node_x, nl.node_y, rg)
    for mst_b, seg_b, edge_b in [(3, 3, 3), (7, 11, 5), (1, 4, 2)]:
        got = GpuEvalContext(nl, rg, device="cuda", mst_chunk_budget=mst_b,
                             seg_chunk_budget=seg_b,
                             edge_batch_size=edge_b).evaluate(nl.node_x, nl.node_y)
        _assert_batch_invariant_fields(ref, got)     # incl. the straddle block
        _assert_straddle_equal(cpu, got)             # and against evaluator_ref


@pytest.mark.parametrize("k_shape", [(1, 1, 20), (4, 2, 20), (8, 4, 32)])
def test_gpu_straddle_matches_reference_for_k1_k8_k32(k_shape):
    rows, cols, lattice = k_shape
    rng = np.random.default_rng(5)
    if rows == 1 and cols == 1:
        from ioplace.regions import RegionSet, RegionSpec
        rs = RegionSet(die=DIE, lattice=lattice,
                       regions=[RegionSpec("P0", np.array([[0., 0., 100., 100.]]))])
    else:
        rs = make_grid_regions(DIE, rows, cols, lattice=lattice)
    rg = RegionGrid(rs)
    nl = _sized_case(rng)
    ref = evaluate(nl, nl.node_x, nl.node_y, rg)
    gpu = evaluate_gpu(nl, nl.node_x, nl.node_y, rg)
    if rg.k == 1:
        assert ref.straddle_cells == 0     # one region: nothing can straddle
    _assert_straddle_equal(ref, gpu)


def test_gpu_straddle_off_leaves_the_legacy_fields_bit_identical():
    from ioplace.evaluator_gpu import GpuEvalContext
    rng = np.random.default_rng(1)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng)
    on = GpuEvalContext(nl, rg, device="cuda").evaluate(nl.node_x, nl.node_y)
    off = GpuEvalContext(nl, rg, device="cuda",
                         straddle=False).evaluate(nl.node_x, nl.node_y)
    _assert_batch_invariant_fields(on, off, straddle=False)
    assert off.straddle_cells == 0 and off.per_node_straddle is None
    assert on.straddle_cells > 0


def test_gpu_per_call_straddle_switch_matches_the_context_default():
    from ioplace.evaluator_gpu import GpuEvalContext
    rng = np.random.default_rng(2)
    rg = RegionGrid(make_grid_regions(DIE, 4, 4, lattice=20))
    nl = _sized_case(rng)
    ctx = GpuEvalContext(nl, rg, device="cuda")
    _assert_straddle_equal(ctx.evaluate(nl.node_x, nl.node_y),
                           ctx.evaluate(nl.node_x, nl.node_y, straddle=True))
    assert ctx.evaluate(nl.node_x, nl.node_y, straddle=False).per_node_straddle is None
    lean = GpuEvalContext(nl, rg, device="cuda", straddle=False)
    with pytest.raises(ValueError, match="straddle=False"):
        lean.evaluate(nl.node_x, nl.node_y, straddle=True)
```

Also extend `_assert_batch_invariant_fields` (`tests/test_evaluator_gpu.py:426-445`) so every *existing* batch-invariance caller covers the new fields too. Change its signature to

```python
def _assert_batch_invariant_fields(a, b, float_rel=1e-12, straddle=True):
```

and append at the end of its body:

```python
    if straddle:
        # v2 P-F: same split as everything above -- integers bit-exact, the
        # three float64 reductions at float_rel. `straddle=False` is only for
        # the one test that deliberately compares a straddle-on run against a
        # straddle-off one.
        for field in _STRADDLE_INT_SCALARS:
            assert getattr(a, field) == getattr(b, field), field
        assert np.array_equal(a.per_node_straddle, b.per_node_straddle)
        assert np.array_equal(a.per_net_pin_split, b.per_net_pin_split)
        for field in _STRADDLE_FLOAT_SCALARS:
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=float_rel), field
```

Because `_STRADDLE_INT_SCALARS`/`_STRADDLE_FLOAT_SCALARS` are referenced from inside it, move those two module-level tuples (and `_assert_straddle_equal`) **above** `_assert_batch_invariant_fields` rather than appending them at the end of the file.

- [x] **Step 2: Run test to verify it fails**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_gpu.py -k straddle -v`
Expected: FAIL — `AssertionError` on `straddle_cells` (ref reports a positive count, the GPU still returns the dataclass default 0).

- [x] **Step 3: Add the constructor flag and the size tensors**

In `GpuEvalContext.__init__` (line 94), change the signature to

```python
    def __init__(self, nl, rg, device="cuda", max_degree=256,
                 mst_chunk_budget=8_000_000, seg_chunk_budget=8_000_000,
                 edge_batch_size=1_000_000, straddle=True):
```

and add to its docstring:

```
        straddle (v2 P-F, design sec 7): compute the straddle diagnostics.
        Costs two persistent (num_physical,) float64 size tensors and, per
        evaluate(), an O(N) geometry pass plus one extra torch.unique over the
        pin keys -- small next to the MST, but not free at 30M cells, so it is
        switchable both here (skip the allocation) and per call
        (evaluate(..., straddle=False), which the in-loop diagnostic callback in
        run_placement_io uses).
```

Right after `self.pin_offset_y_t = ...` (line 225), insert:

```python
        # v2 P-F: movable/physical counts and the sizes the sec 7 diagnostics
        # need. float64 for the same reason every other geometry tensor here is
        # float64 -- these feed _to_idx, whose truncation decides integers.
        self.num_movable = int(nl.num_movable)
        self.num_physical = int(n_physical)
        self.straddle = bool(straddle)
        if self.straddle:
            self.node_size_x_t = torch.from_numpy(
                np.asarray(nl.node_size_x[:n_physical], dtype=np.float64)).to(self.device)
            self.node_size_y_t = torch.from_numpy(
                np.asarray(nl.node_size_y[:n_physical], dtype=np.float64)).to(self.device)
        else:
            self.node_size_x_t = self.node_size_y_t = None
```

- [x] **Step 4: Add the torch mirror**

Insert after `_pin_positions` (line 306):

```python
    def _distinct_regions_per_net(self, pin_rid):
        """(n_nets,) int64 distinct-region count per net, 0 for degree<2 -- the
        torch mirror of straddle.distinct_regions_per_net, using the same
        net*64+region packing evaluate() already uses for pin_bm."""
        key = torch.unique(self.pin2net_t.to(torch.int64) * 64
                           + pin_rid.to(torch.int64))
        counts = torch.bincount(key // 64, minlength=self.n_nets)
        return torch.where(self.degrees_t >= 2, counts, torch.zeros_like(counts))

    def _straddle_stats(self, node_x, node_y, pin_rid, per_net_lambda):
        """Torch mirror of ioplace.straddle.straddle_diagnostics. Every
        convention is documented there; this must not invent one.

        The lattice-line coordinate below divides/multiplies through the 0-dim
        self._cell_w_t / self._cell_h_t tensors, never the python floats
        self.cell_w / self.cell_h -- same C1 reciprocal-multiply hazard as
        _to_idx (see its comment); a python float here mis-rounds the quadrant
        split exactly at region boundaries.
        """
        dev = self.device
        m = self.num_movable
        x = node_x[:m]
        y = node_y[:m]
        w = self.node_size_x_t[:m]
        h = self.node_size_y_t[:m]
        xr = x + w
        yt = y + h

        def _rid(ax, ay):
            ix, iy = self._to_idx(ax, ay)
            return self.grid_t[iy, ix].to(torch.int64)

        r00 = _rid(x, y)
        r10 = _rid(xr, y)
        r01 = _rid(x, yt)
        r11 = _rid(xr, yt)
        owner = _rid(x + 0.5 * w, y + 0.5 * h)
        straddle = (r10 != r00) | (r01 != r00) | (r11 != r00)

        ix0, iy0 = self._to_idx(x, y)
        ix1, iy1 = self._to_idx(xr, yt)
        xm = torch.minimum(xr, self.xl + (ix0.to(torch.float64) + 1.0) * self._cell_w_t)
        ym = torch.minimum(yt, self.yl + (iy0.to(torch.float64) + 1.0) * self._cell_h_t)
        lw = (xm - x).clamp(min=0.0)
        rw = (xr - xm).clamp(min=0.0)
        bh = (ym - y).clamp(min=0.0)
        th = (yt - ym).clamp(min=0.0)
        out_area = (lw * bh * (r00 != owner).to(torch.float64)
                    + rw * bh * (r10 != owner).to(torch.float64)
                    + lw * th * (r01 != owner).to(torch.float64)
                    + rw * th * (r11 != owner).to(torch.float64))
        wide = ((ix1 - ix0) > 1) | ((iy1 - iy0) > 1)

        per_node = torch.zeros(self.num_physical, dtype=torch.uint8, device=dev)
        per_node[:m] = straddle.to(torch.uint8)
        owner_full = torch.zeros(self.num_physical, dtype=torch.int64, device=dev)
        owner_full[:m] = owner
        node_of_pin = self.pin2node_t.to(torch.int64)
        pin_rid_re = torch.where(per_node[node_of_pin].bool(),
                                 owner_full[node_of_pin], pin_rid.to(torch.int64))
        split = (per_net_lambda - self._distinct_regions_per_net(pin_rid_re))

        total_area = float((w * h).sum().item())
        out_total = float(out_area.sum().item())
        return {
            "straddle_cells": int(straddle.sum().item()),
            "straddle_area_fraction": (out_total / total_area) if total_area > 0.0 else 0.0,
            "straddle_pin_split_nets": int((split > 0).sum().item()),
            "straddle_out_area": out_total,
            "straddle_movable_area": total_area,
            "straddle_wide_cells": int(wide.sum().item()),
            "per_node_straddle": per_node.cpu().numpy(),
            "per_net_pin_split": split.to(torch.int32).cpu().numpy(),
        }
```

`per_net_lambda` is passed in rather than recomputed: `evaluate` already derives it from `pin_bm`'s popcount (line 602), and it is by construction the same quantity `_distinct_regions_per_net(pin_rid)` would return — reusing it saves a `torch.unique` and removes any chance of the two disagreeing.

- [x] **Step 5: Wire it into `evaluate`**

Change the signature (line 531) to `def evaluate(self, node_x, node_y, *, straddle=None):` and insert at the top of the body:

```python
        want_straddle = self.straddle if straddle is None else bool(straddle)
        if want_straddle and not self.straddle:
            raise ValueError("this GpuEvalContext was built with straddle=False, "
                             "so it holds no node-size tensors; rebuild it with "
                             "straddle=True to ask for the sec 7 diagnostics")
```

Replace line 576 (`del pin_ix, pin_iy, pin_rid  # only needed to build net_region_key above`) with:

```python
        del pin_ix, pin_iy
        if not want_straddle:
            pin_rid = None      # only net_region_key above needed it
```

Immediately after `hard_lambda_sum = ...` (line 604), insert:

```python
        # v2 P-F (design sec 7): per_net_lambda is in hand and pin_rid is still
        # alive, which is the only point in evaluate() where both are true.
        if want_straddle:
            straddle_stats = self._straddle_stats(node_x, node_y, pin_rid,
                                                  per_net_lambda)
            pin_rid = None
        else:
            straddle_stats = {}
```

and add `**straddle_stats,` as the last entry of the `return EvalResult(...)` call (after `per_net_home=...`, line 768).

Change `evaluate_gpu` (line 772) to:

```python
def evaluate_gpu(nl, node_x, node_y, rg, max_degree=256, device="cuda", straddle=True):
    return GpuEvalContext(nl, rg, device=device, max_degree=max_degree,
                          straddle=straddle).evaluate(node_x, node_y)
```

- [x] **Step 6: Run the new tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_gpu.py -k straddle -v`
Expected: PASS — 14 passed (5 seeds + 3 K shapes + 6 singles).

- [x] **Step 7: Run the whole evaluator parity suite**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_gpu.py tests/test_evaluator_ref.py tests/test_straddle.py -m "not slow" -v`
Expected: PASS, no regressions. Every pre-existing test now also runs `_assert_batch_invariant_fields`' new straddle clauses.

- [x] **Step 8: Run the slow evaluator regressions (GPU permitting)**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluator_gpu.py -m slow -v`
Expected: PASS — `test_legacy_fields_bit_exact_regression_adaptec1_k16_grid_flat`, the bigblue4 batch-invariance run and the two memory-budget tests. The memory tests are the ones at risk: they assert bigblue4 K=32 peak reduction and mempool_group K=32 under 2 GB, and this task adds two persistent `(num_physical,)` float64 tensors (bigblue4 ≈ 2.2M nodes → 35 MB; mempool_group ≈ 3M → 48 MB). If either budget now binds, do **not** loosen the assertion: pass `straddle=False` in that test's context construction and record why in the commit message.

- [x] **Step 9: Commit**

```bash
git add src/ioplace/evaluator_gpu.py tests/test_evaluator_gpu.py
git commit -m "feat(eval): GPU mirror of the sec 7 straddle diagnostics, bit-exact on integers

Mirrors ioplace/straddle.py under spec sec 9's parity contract: straddle_cells,
straddle_pin_split_nets, straddle_wide_cells, per_node_straddle and
per_net_pin_split bit-exact ref-vs-GPU and across mst_chunk_budget /
seg_chunk_budget / edge_batch_size; the three float fields keep tree_wl's
rel<=1e-12. The lattice-line coordinate goes through the 0-dim cell-size
tensors, same C1 hazard as _to_idx. straddle=False is available per context and
per call for the 30M-cell path and for the in-loop diagnostic callback.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Persist the diagnostics — `evaluation.npz` schema 2 and `result.json`

**Files:**
- Modify: `src/ioplace/export/evaluation.py:14-18` (constants), `:68-117` (`save_evaluation`), `:120-141` (`load_evaluation`)
- Modify: `src/ioplace/drivers/run_placement.py:114-119` (add `_pack_straddle_metrics`)
- Modify: `src/ioplace/drivers/run_placement_io.py:18` (import), `:33-58` (`RESULT_FIELDS`), `:518` (in-loop call), `:784-785` (final metrics)
- Modify **if and only if P-B has landed**: `src/ioplace/artifacts.py` (`MAIN_FLOW_RESULT_FIELDS`), `src/ioplace/drivers/run_main_flow.py`
- Test: `tests/test_evaluation_export.py` (append), `tests/test_straddle.py` (append)

**Interfaces:**
- Consumes: Task 3's `STRADDLE_SCALARS`/`STRADDLE_ARRAYS` and the `EvalResult` fields; Task 4's GPU result.
- Produces:
  - `export/evaluation.SCHEMA_VERSION = 2`, `SUPPORTED_SCHEMA_VERSIONS = (1, 2)`
  - `evaluation.npz` arrays `per_node_straddle` `(num_physical,)` uint8 and `per_net_pin_split` `(num_nets,)` int32, plus `metadata["straddle"]` = the six scalars + `"anchor": "center"` + `"box": "closed_four_corner"`, or `None`
  - `run_placement._pack_straddle_metrics(res) -> dict` with exactly `STRADDLE_SCALARS` as keys
  - `run_placement_io.RESULT_FIELDS` gains the six straddle names
  - (guarded) `artifacts.MAIN_FLOW_RESULT_FIELDS` gains `"node_anchor"` + the six straddle names; `run_main_flow` gains `--node-anchor`

**Why a version bump and not a silent extra array.** `load_evaluation` refuses anything whose `schema_version` is not the current one (`export/evaluation.py:124-125`), and there are historical `evaluation.npz` files under `results/` that the route-calibration tooling still reads. Bumping to 2 while accepting `(1, 2)` keeps those readable and still lets a consumer tell whether the straddle block should be there. The block is absent — and `metadata["straddle"]` is `None` — whenever the result came from `straddle=False`, so "computed" and "not computed" are distinguishable rather than both looking like zero.

**Why `_pack_straddle_metrics` is separate from `_pack_eval_metrics`.** `_pack_eval_metrics` (`run_placement.py:114-119`) feeds `run_flat`, `run_two_stage` and `run_reweight` as well, none of which has a `RESULT_FIELDS` gate; widening it would silently add keys to three other drivers' `result.json`. A separate packer spread only into `run_io` (and, guarded, `run_main_flow`) keeps the blast radius at zero.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_evaluation_export.py`:

```python
def _straddling_evidence(tmp_path, straddle=True):
    from tests.test_straddle import _corner_case
    nl = _corner_case()
    rg = RegionGrid(make_grid_regions((0., 0., 100., 100.), 2, 2, lattice=10))
    result = evaluate(nl, nl.node_x, nl.node_y, rg, straddle=straddle)
    path = tmp_path / "evaluation.npz"
    save_evaluation(path, nl, rg, result, nl.node_x, nl.node_y,
                    ["n0", "n1", "n2", "n3"])
    return path, nl, rg, result


def test_schema_2_round_trips_the_straddle_block(tmp_path):
    path, nl, rg, result = _straddling_evidence(tmp_path)
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["schema_version"] == 2
    straddle = data["metadata"]["straddle"]
    assert straddle["straddle_cells"] == 2
    assert straddle["straddle_pin_split_nets"] == 1
    assert straddle["straddle_area_fraction"] == pytest.approx(20. / 48.)
    assert straddle["straddle_out_area"] == pytest.approx(20.)
    assert straddle["straddle_movable_area"] == pytest.approx(48.)
    assert straddle["straddle_wide_cells"] == 0
    assert straddle["anchor"] == "center" and straddle["box"] == "closed_four_corner"
    np.testing.assert_array_equal(data["per_node_straddle"], [0, 1, 1, 0])
    np.testing.assert_array_equal(data["per_net_pin_split"], [-1, 0, 1, -1])


def test_evidence_without_diagnostics_records_that_fact(tmp_path):
    path, _, rg, _ = _straddling_evidence(tmp_path, straddle=False)
    data = load_evaluation(path, rg=rg)
    assert data["metadata"]["schema_version"] == 2
    assert data["metadata"]["straddle"] is None
    assert "per_node_straddle" not in data and "per_net_pin_split" not in data


def test_corrupted_straddle_arrays_fail_total_validation(tmp_path):
    path, _, _, _ = _straddling_evidence(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["per_node_straddle"][0] = 1          # now 3 straddlers, metadata says 2
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="total mismatch: per_node_straddle"):
        load_evaluation(path)


def test_a_schema_1_archive_still_loads(tmp_path):
    path, _, _, _ = _straddling_evidence(tmp_path)
    import json
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["schema_version"] = 1
    metadata.pop("straddle")
    arrays.pop("per_node_straddle")
    arrays.pop("per_net_pin_split")
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    data = load_evaluation(path)
    assert data["metadata"]["schema_version"] == 1
    assert data["metadata"].get("straddle") is None
```

Append to `tests/test_straddle.py`:

```python
def test_pack_straddle_metrics_has_exactly_the_six_scalar_names():
    from ioplace.drivers.run_placement import _pack_straddle_metrics
    nl = _corner_case()
    res = evaluate(nl, nl.node_x, nl.node_y, _grid())
    packed = _pack_straddle_metrics(res)
    assert tuple(packed) == STRADDLE_SCALARS
    assert packed["straddle_cells"] == 2
    assert packed["straddle_pin_split_nets"] == 1
    assert isinstance(packed["straddle_area_fraction"], float)
    assert isinstance(packed["straddle_cells"], int)


def test_run_io_result_fields_carry_every_straddle_scalar():
    from ioplace.drivers.run_placement_io import RESULT_FIELDS
    for name in STRADDLE_SCALARS:
        assert name in RESULT_FIELDS, name
```

- [x] **Step 2: Run tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluation_export.py tests/test_straddle.py -v`
Expected: FAIL — `assert data["metadata"]["schema_version"] == 2` gets `1`, and `ImportError: cannot import name '_pack_straddle_metrics'`.

- [x] **Step 3: Bump the `evaluation.npz` schema**

In `src/ioplace/export/evaluation.py`, replace lines 14-18 with:

```python
SCHEMA_VERSION = 2
# v2 P-F: schema 2 adds the sec 7 straddle block. Schema 1 archives stay
# readable -- results/ holds historical evidence the route-calibration tooling
# still pairs against, and a diagnostics addition is no reason to orphan it.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
PER_NET_FIELDS = (
    "per_net_crossings", "per_net_ft", "per_net_lambda",
    "per_net_steiner", "per_net_home",
)
```

Add `from ioplace.straddle import STRADDLE_SCALARS` to the imports.

In `save_evaluation`, after the `arrays.update(...)` block (line 85-92) and before `metadata = dict(...)`, insert:

```python
    # v2 P-F (design sec 7). Present iff the evaluator actually computed them,
    # so "not measured" and "measured as zero" stay distinguishable.
    straddle = None
    if getattr(result, "per_node_straddle", None) is not None:
        node_straddle = np.asarray(result.per_node_straddle, dtype=np.uint8)
        pin_split = np.asarray(result.per_net_pin_split, dtype=np.int32)
        if node_straddle.shape != (nl.num_physical,):
            raise ValueError("per_node_straddle must cover every physical node")
        if pin_split.shape != (nl.num_nets,):
            raise ValueError("per_net_pin_split must cover the entire netlist")
        arrays.update(per_node_straddle=node_straddle, per_net_pin_split=pin_split)
        straddle = {key: getattr(result, key) for key in STRADDLE_SCALARS}
        straddle = {key: (int(value) if isinstance(value, (int, np.integer))
                          else float(value)) for key, value in straddle.items()}
        straddle["anchor"] = "center"
        straddle["box"] = "closed_four_corner"
```

and add `straddle=straddle,` to the `metadata = dict(...)` literal, next to `node_region_convention=...`.

In `load_evaluation`, replace lines 124-125 with:

```python
    if metadata.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported evaluator evidence schema")
```

and insert after the existing per-net total checks (line 135):

```python
    straddle = metadata.get("straddle")
    if straddle is not None:
        for key, shape in (("per_node_straddle", (metadata["num_physical"],)),
                           ("per_net_pin_split", (n,))):
            if key not in data or data[key].shape != shape:
                raise ValueError(f"invalid evaluator evidence array: {key}")
        if int(data["per_node_straddle"].sum(dtype=np.int64)) != straddle["straddle_cells"]:
            raise ValueError("evaluator total mismatch: per_node_straddle")
        if int((data["per_net_pin_split"] > 0).sum()) != straddle["straddle_pin_split_nets"]:
            raise ValueError("evaluator total mismatch: per_net_pin_split")
```

- [x] **Step 4: Add the metric packer and wire `run_io`**

In `src/ioplace/drivers/run_placement.py`, after `_pack_eval_metrics` (line 119):

```python
def _pack_straddle_metrics(res):
    """v2 P-F (design sec 7 diagnostics 1-3). Deliberately separate from
    _pack_eval_metrics: that one also feeds run_flat / run_two_stage /
    run_reweight, none of which has a RESULT_FIELDS gate, so widening it would
    silently change three other drivers' result.json."""
    from ioplace.straddle import STRADDLE_SCALARS
    out = {}
    for name in STRADDLE_SCALARS:
        value = getattr(res, name)
        out[name] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return out
```

In `src/ioplace/drivers/run_placement_io.py`:

1. Add `_pack_straddle_metrics` to the `from ioplace.drivers.run_placement import (...)` list at line 18.
2. Add to `RESULT_FIELDS`, after the `node_anchor` entry Task 1 added:

```python
                 # v2 P-F (design sec 7 diagnostics 1-3)
                 "straddle_cells", "straddle_area_fraction",
                 "straddle_pin_split_nets", "straddle_out_area",
                 "straddle_movable_area", "straddle_wide_cells",
```

3. Line 518, the in-loop diagnostic evaluation, becomes
`res = ctx.evaluate(node_x, node_y, straddle=False)`
with the comment `# sec 7 diagnostics are a final-placement report, not a per-callback cost`.
4. After line 785 (`metrics = _pack_eval_metrics(res)`), add `metrics.update(_pack_straddle_metrics(res))`.

- [x] **Step 5: Run the fast tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_evaluation_export.py tests/test_straddle.py tests/test_node_anchor.py -v`
Expected: PASS — 7 + 13 + 23 passed.

- [x] **Step 6: Run everything that reads `evaluation.npz`**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_route_eval_s2.py tests/test_stage2_calibration.py tests/test_stage2_identity.py tests/test_evaluate_placement.py tests/test_route_feedback.py -m "not slow" -v`
Expected: PASS. These are the `load_evaluation`/`pin_regions_from_evaluation` consumers; `SUPPORTED_SCHEMA_VERSIONS` is what keeps them green.

- [x] **Step 7: P-B-guarded wiring**

Check whether P-B has landed:

```bash
test -f src/ioplace/drivers/run_main_flow.py && echo LANDED || echo PENDING
```

**If `LANDED`** — make exactly these three edits, then re-run `"$IOPLACE_PYTHON" -m pytest tests/test_main_flow_driver.py tests/test_artifacts.py -v`:

1. In `src/ioplace/artifacts.py`, add to `MAIN_FLOW_RESULT_FIELDS`, immediately after the `"io_identity_residual", "io_fence_gp_source",` line:

```python
    # v2 P-F (design sec 7): the anchor the soft phase ran with, and the three
    # evaluator-side diagnostics plus their two components. Together with
    # io_delta_at_freeze and fence_compliance above, these are F's five.
    "node_anchor", "straddle_cells", "straddle_area_fraction",
    "straddle_pin_split_nets", "straddle_out_area", "straddle_movable_area",
    "straddle_wide_cells",
```

2. In `src/ioplace/drivers/run_main_flow.py`:
   - `build_parser()` gains, next to `--argmax-chunk`:
     ```python
     parser.add_argument("--node-anchor", choices=["lower_left", "center", "pin"],
                         default="center",
                         help="anchor for the soft region assignment (design v2 "
                              "sec 7); 'center' matches the freeze rule and "
                              "whole-cell fence ownership, 'pin' is rejected")
     ```
   - `run_soft_phase` gains the keyword `node_anchor="center"` and threads it into its `IoTerm(...)` construction together with `node_size_x=nl.node_size_x, node_size_y=nl.node_size_y` — exactly the four extra arguments Task 1 added to `run_placement_io.py:291-293`.
   - `run_main_flow`'s option validation raises the same two `ValueError`s Task 1 added to `run_io` (unknown anchor; `pin` is IoTermRef-only), and the assembled `result` dict gains `"node_anchor": node_anchor` plus `**_pack_straddle_metrics(res)` on the final evaluator result.

3. In `tests/test_main_flow_driver.py`, add:

```python
def test_parser_defaults_the_node_anchor_to_center():
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.node_anchor == "center"


def test_main_flow_result_fields_carry_the_p_f_diagnostics():
    from ioplace.artifacts import MAIN_FLOW_RESULT_FIELDS
    from ioplace.straddle import STRADDLE_SCALARS
    for name in ("node_anchor",) + STRADDLE_SCALARS:
        assert name in MAIN_FLOW_RESULT_FIELDS, name
```

**If `PENDING`** — make no edit under `src/ioplace/artifacts.py` or `run_main_flow.py` (they do not exist), and instead append this line to the plan's own Task 8 doc-sync note: *"P-F's seven `result.json` fields (`node_anchor` + the six straddle scalars) are wired into `run_placement_io.RESULT_FIELDS` only; P-B must add the identical seven to `artifacts.MAIN_FLOW_RESULT_FIELDS` and to `run_main_flow`'s result dict, and `--node-anchor` to its parser, when it lands."* Then open the follow-up by adding the same sentence to `docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md` Task 7's Interfaces block as a one-line **P-F dependency** note. That single cross-plan line is the only edit P-F makes to another plan.

- [x] **Step 8: Run the full fast suite**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow"`
Expected: PASS, no new failures.

- [x] **Step 9: Commit**

```bash
git add src/ioplace/export/evaluation.py src/ioplace/drivers/run_placement.py \
        src/ioplace/drivers/run_placement_io.py tests/test_evaluation_export.py \
        tests/test_straddle.py
# plus src/ioplace/artifacts.py src/ioplace/drivers/run_main_flow.py
# tests/test_main_flow_driver.py docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md
# depending on which branch of Step 7 ran
git commit -m "feat(export): persist the sec 7 straddle diagnostics

evaluation.npz goes to schema 2 -- per_node_straddle, per_net_pin_split and a
metadata straddle block -- while still reading schema 1, so the historical
evidence under results/ stays pairable. result.json gains the six scalars
through _pack_straddle_metrics, deliberately not through _pack_eval_metrics,
which three ungated drivers also use. The in-loop diagnostic evaluation opts out
with straddle=False: these are a final-placement report, not a per-callback cost.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: The accounting identity, checked against re-measured IO counts

**Files:**
- Create: `src/ioplace/io_identity.py`
- Test: `tests/test_io_identity.py` (new)

**Interfaces:**
- Consumes: nothing from P-B at import time. It reads *field names* P-B Task 6/7 produce (`io_soft`, `io_fence_gp`, `io_count`, `io_delta_at_freeze`, `lg_loss`, `fence_compliance_center`/`fence_compliance`) and Task 5's six straddle names, but imports no module that might not exist. The slow GCD test is the only place `ioplace.drivers.run_main_flow` is touched, and it skips if that module is absent.
- Produces:
  - `RECORDED_FIELDS = ("io_soft", "io_fence_gp", "io_count", "io_delta_at_freeze", "lg_loss")`
  - `P_F_DIAGNOSTICS = ("straddle_cells", "straddle_area_fraction", "straddle_pin_split_nets", "io_delta_at_freeze", "fence_compliance")`
  - `verify_io_identity(recorded, measured, *, tol=1) -> dict` with keys `residual`, `stale`, `ok`, `tol`, `recorded`, `measured`
  - `p_f_diagnostics(result) -> dict` — the five names of spec §9's F "done" criterion, or `KeyError`

**Why this is not the tautology P-B already has.** `main_flow_metrics.io_accounting` defines `io_delta_at_freeze = io_fence_gp − io_soft` and `lg_loss = io_count − io_fence_gp`, so its `io_identity_residual` is algebraically zero for any three inputs and "detects nothing" (P-B plan Task 6, pre-flight amendment D-1). `verify_io_identity` substitutes **independently re-measured** `io_soft` and `io_count` — re-evaluated from the saved position arrays, in a separate process, by a freshly built evaluator context — into the identity while keeping the **recorded** `io_delta_at_freeze` and `lg_loss`. The residual is then zero only if the recorded deltas really describe the placements on disk. A stale, mismatched or wrong-frame measurement moves it. `stale` reports each per-term disagreement separately so the failure names itself.

`io_fence_gp` is *not* re-measurable: P-B writes `soft.npz` and the final `placement.npz`, but the pre-legalisation fence-GP positions are never persisted (that is what `io_fence_gp_source` exists to vouch for). `verify_io_identity` therefore accepts a `measured` dict containing any subset of the three and only checks what it was given — with `io_soft` and `io_count` alone, the identity is already fully constrained.

- [x] **Step 1: Write the failing test**

Create `tests/test_io_identity.py`:

```python
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
    require the identity to close within +-1 crossing."""
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
                           init="region_center", node_anchor="center",
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
```

- [x] **Step 2: Run the fast tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_io_identity.py -m "not slow" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ioplace.io_identity'`.

- [x] **Step 3: Write `src/ioplace/io_identity.py`**

```python
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
    MEASURABLE_FIELDS. Returns the residual, the per-term staleness and a
    verdict; never raises on a bad *number*, only on a missing or unknown
    *name* -- a silent default here would be indistinguishable from the stale
    measurement this function exists to catch.
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
    return {"residual": residual, "stale": stale, "tol": int(tol),
            "ok": abs(residual) <= int(tol) and not any(stale.values()),
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
```

- [x] **Step 4: Run the fast tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_io_identity.py -m "not slow" -v`
Expected: PASS — 11 passed.

- [x] **Step 5: Run the GCD gate (only if GPU 3 is free)**

Run: `nvidia-smi` first. If GPU 3 is idle:
`CUDA_VISIBLE_DEVICES=3 "$IOPLACE_PYTHON" -m pytest tests/test_io_identity.py -m slow -v`
Expected: PASS if P-B has landed; `SKIPPED [1] ... P-B has not landed` otherwise. A skip is an acceptable state for this task but **not** for F's exit criterion — record it in the commit message and re-run this single test the day P-B lands.

- [x] **Step 6: Commit**

```bash
git add src/ioplace/io_identity.py tests/test_io_identity.py
git commit -m "test(identity): check sec 7's identity against re-measured IO counts

main_flow_metrics.io_accounting's residual is algebraically zero for any three
inputs (P-B amendment D-1). verify_io_identity substitutes io_soft and io_count
re-evaluated from soft.npz and placement.npz by a fresh evaluator context and
keeps the recorded deltas, so a stale or wrong-frame measurement moves the
residual and names itself in `stale`. p_f_diagnostics is spec sec 9's F exit
criterion as an assertion.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The anchor-comparison experiment and its result doc

**Files:**
- Create: `src/scripts/run_anchor_comparison.py`
- Create: `docs/results/2026-09-19-p-f-anchor-comparison.md`
- Test: `tests/test_anchor_comparison.py` (new)

**Interfaces:**
- Consumes: Task 1's `IoTerm(..., node_anchor=...)`; Task 2's `IoTermRef(..., node_anchor="pin", pin_csr=...)` and `build_net_pin_csr`; `ioplace.netlist.load_netlist`; `ioplace.regions.RegionSet.from_json`; `ioplace.ops.soft_assign.rect_table`; `ioplace.ops.io_term.build_net_node_csr`. Guarded: `ioplace.artifacts.load_positions`/`scaled_region_set`, `ioplace.drivers.run_main_flow`.
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `surrogate_io(nl, placedb, params, rs_scaled, node_x_scaled, node_y_scaled, anchor, tau, *, k, ignore_net_degree, device="cuda") -> dict` with keys `anchor`, `l_io_soft`, `lambda_sum_soft`, `n_active`, `tau`
  - `ANCHOR_TABLE_COLUMNS: tuple` — the exact column order the result doc uses
  - `compare(rows, truth) -> list` — each row gains `io_lb_final`, `abs_err`, `rel_err`, `io_count_final`, `closest`
  - `main(argv=None) -> dict`

**What the experiment asks and what it compares.** Spec §7: "On `mempool_tile`, from the same soft solution, compute the GP surrogate IO with all three anchors and compare against the evaluator after fence LG; expect centre closest. If pin is closer, LG displacement exceeds half a cell and LG must be examined."

The surrogate is `L_IO = Σ_e w_e·max(λ_e − 1, 0)` with `w_mode="unit"` — literally the term the GP minimises, read off the *same* soft solution with each anchor in turn. The truth is `hard_lambda_sum = Σ_e max(λ_e^hard − 1, 0)` on the post-fence-LG placement: the *same functional* of the *same* per-net distinct-region counts, evaluated where the soft assignment has become hard. That is the only apples-to-apples pairing. `io_count` (MST-geometry crossings) is reported alongside as context but is not the comparison target — the surrogate has no MST in it and comparing against it would confound anchor bias with routing geometry.

**Producing the two placements.**

- **Preferred (P-B landed).** One soft solution, then one fence LG, both from `run_main_flow.py`:
  ```bash
  "$IOPLACE_PYTHON" -m ioplace.drivers.run_main_flow \
      --config benchmarks/ispd25/h100/mempool_tile_wrap.json \
      --out-dir runs/anchor_cmp/mempool_tile_wrap --phase soft \
      --k 16 --rtype grid --init die_center --node-anchor center --dp-seed 1000
  "$IOPLACE_PYTHON" -m ioplace.drivers.run_main_flow \
      --config benchmarks/ispd25/h100/mempool_tile_wrap.json \
      --out-dir runs/anchor_cmp/mempool_tile_wrap --phase fence \
      --k 16 --rtype grid --dp-seed 1000
  ```
  The `--phase soft` run writes `soft.npz`, `frozen_membership.npz`, `regions.json` and `freeze.json`; `--phase fence` consumes exactly those from `--out-dir` and writes `placement.npz`, `evaluation.npz` and `result.json`. The script then reads `soft.npz` + `freeze.json["tau"]` for the surrogate and `result.json` for the truth. If `benchmarks/ispd25/h100/mempool_tile_wrap.json` does not exist (P-C Task 11 pending), use `results/route_gp_20260914/mempool_tile_wrap.json`.
- **Fallback (P-B pending).** There is no driver on this branch that fence-legalises a *given* soft solution: `run_placement_two_stage.run_two_stage(config, k, rtype, seed, out_json)` builds its own Mt-KaHyPar membership, runs its own fence GP and legalises that (`src/ioplace/drivers/run_placement_two_stage.py:174-266`). So the fallback is:
  ```bash
  "$IOPLACE_PYTHON" -m ioplace.drivers.run_placement --mode io \
      --config results/route_gp_20260914/mempool_tile_wrap.json --k 16 --rtype grid \
      --out runs/anchor_cmp/soft.json --node-anchor center \
      --snapshot-iters 600 --snapshot-dir runs/anchor_cmp/snaps
  "$IOPLACE_PYTHON" -m ioplace.drivers.run_placement --mode two_stage \
      --config results/route_gp_20260914/mempool_tile_wrap.json --k 16 --rtype grid \
      --out runs/anchor_cmp/fence.json
  "$IOPLACE_PYTHON" src/scripts/run_anchor_comparison.py \
      --config results/route_gp_20260914/mempool_tile_wrap.json --k 16 --rtype grid \
      --soft-npz runs/anchor_cmp/snaps/it0600.npz --tau-from-npz \
      --truth-result runs/anchor_cmp/fence.json \
      --out-dir runs/anchor_cmp --degraded
  ```
  `--degraded` is mandatory there and stamps `"degraded": true` plus a reason string into `anchor_comparison.json` and a bold warning into the markdown, because the two placements come from **different memberships**: the surrogate is read off run_io's GP snapshot while the truth comes from a partitioner-driven fence run that never saw it. That does **not** answer §7's question; it only exercises the script. Do not publish a fallback table as the §7 result.

- [ ] **Step 1: Write the failing test**

Create `tests/test_anchor_comparison.py`:

```python
import json
import os

import numpy as np
import pytest

from scripts.run_anchor_comparison import (ANCHOR_TABLE_COLUMNS, build_parser,
                                           compare, render_markdown)


def _rows():
    return [{"anchor": "lower_left", "l_io_soft": 900.0, "lambda_sum_soft": 5100.0,
             "n_active": 4200, "tau": 0.05},
            {"anchor": "center", "l_io_soft": 1010.0, "lambda_sum_soft": 5210.0,
             "n_active": 4200, "tau": 0.05},
            {"anchor": "pin", "l_io_soft": 1180.0, "lambda_sum_soft": 5380.0,
             "n_active": 4200, "tau": 0.05}]


def test_parser_requires_a_config_and_an_out_dir():
    parser = build_parser()
    args = parser.parse_args(["--config", "c.json", "--out-dir", "o"])
    assert args.k == 16 and args.rtype == "grid" and args.degraded is False
    assert args.anchors == ["lower_left", "center", "pin"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--out-dir", "o"])


def test_compare_scores_every_anchor_against_the_post_lg_lower_bound():
    truth = {"hard_lambda_sum": 1000, "io_count": 1640}
    rows = compare(_rows(), truth)
    assert [r["anchor"] for r in rows] == ["lower_left", "center", "pin"]
    assert [r["abs_err"] for r in rows] == [-100.0, 10.0, 180.0]
    assert rows[1]["rel_err"] == pytest.approx(0.01)
    assert all(r["io_lb_final"] == 1000 for r in rows)
    assert all(r["io_count_final"] == 1640 for r in rows)
    assert [r["closest"] for r in rows] == [False, True, False]


def test_compare_breaks_a_tie_toward_the_earlier_anchor_deterministically():
    truth = {"hard_lambda_sum": 1000, "io_count": 1640}
    rows = _rows()
    rows[0]["l_io_soft"] = 1010.0        # same |error| as center
    out = compare(rows, truth)
    assert [r["closest"] for r in out] == [True, False, False]


def test_compare_refuses_a_zero_lower_bound_rather_than_dividing_by_it():
    with pytest.raises(ValueError, match="hard_lambda_sum"):
        compare(_rows(), {"hard_lambda_sum": 0, "io_count": 12})


def test_markdown_uses_the_documented_column_order():
    assert ANCHOR_TABLE_COLUMNS == ("anchor", "l_io_soft", "lambda_sum_soft",
                                    "io_lb_final", "abs_err", "rel_err",
                                    "io_count_final", "closest")
    text = render_markdown(compare(_rows(), {"hard_lambda_sum": 1000,
                                             "io_count": 1640}), degraded=False)
    header = text.splitlines()[0]
    assert header == "| " + " | ".join(ANCHOR_TABLE_COLUMNS) + " |"
    assert "lower_left" in text and "center" in text and "pin" in text


def test_markdown_shouts_when_the_run_was_degraded():
    text = render_markdown(compare(_rows(), {"hard_lambda_sum": 1000,
                                             "io_count": 1640}), degraded=True,
                           reason="soft snapshot and fence run use different memberships")
    assert "**DEGRADED**" in text
    assert "different memberships" in text


@pytest.mark.slow
@pytest.mark.gpu
def test_anchor_comparison_runs_end_to_end_on_gcd(tmp_path):
    """Smoke test of the real path on the smallest case available. The physical
    claim is made on mempool_tile_wrap by the campaign in
    docs/results/2026-09-19-p-f-anchor-comparison.md, not here -- GCD is 508
    movable cells and its errors are not meaningful."""
    pytest.importorskip("torch")
    pytest.importorskip("ioplace.drivers.run_main_flow",
                        reason="P-B has not landed; the end-to-end path needs "
                               "run_main_flow --phase soft/fence")
    from scripts.run_anchor_comparison import main
    out_dir = str(tmp_path / "gcd")
    record = main(["--config", os.path.abspath("results/route_feedback_20260914/gcd.json"),
                   "--out-dir", out_dir, "--k", "4", "--rtype", "grid",
                   "--dp-seed", "1000", "--run-flow"])
    assert [r["anchor"] for r in record["rows"]] == ["lower_left", "center", "pin"]
    assert all(np.isfinite(r["l_io_soft"]) for r in record["rows"])
    assert record["degraded"] is False
    assert sum(r["closest"] for r in record["rows"]) == 1
    with open(os.path.join(out_dir, "anchor_comparison.json")) as handle:
        assert json.load(handle)["rows"] == record["rows"]
    assert os.path.exists(os.path.join(out_dir, "anchor_comparison.md"))
```

- [ ] **Step 2: Run the fast tests to verify they fail**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_anchor_comparison.py -m "not slow" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_anchor_comparison'`. (`pyproject.toml` sets `pythonpath = ["src"]`, so script modules import as `scripts.<name>` — the convention `tests/test_stage2_raw_companion.py:4` and `tests/test_routing_gp_driver.py:152` already use.)

- [ ] **Step 3: Write the script**

Create `src/scripts/run_anchor_comparison.py`:

```python
"""Design v2 sec 7's anchor-verification experiment.

From ONE soft solution, compute the GP's own IO surrogate
L_IO = sum_e max(lambda_e - 1, 0) under each of the three anchors, and compare
it against the post-fence-LG truth hard_lambda_sum = sum_e max(lambda_e^hard - 1, 0)
-- the same functional of the same per-net distinct-region counts, evaluated
where the soft assignment has become hard. Expect the cell centre to be closest.
If `pin` is closest, LG displacement exceeds half a cell and LG must be examined
(sec 7).

io_count (MST-geometry crossings) is carried as context only: the surrogate
contains no MST, so scoring against it would confound anchor bias with routing
geometry.
"""
import argparse
import json
import os

import numpy as np

ANCHOR_TABLE_COLUMNS = ("anchor", "l_io_soft", "lambda_sum_soft", "io_lb_final",
                        "abs_err", "rel_err", "io_count_final", "closest")
ANCHORS = ("lower_left", "center", "pin")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--rtype", default="grid", choices=["grid", "slicing"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchors", nargs="+", default=list(ANCHORS), choices=list(ANCHORS))
    parser.add_argument("--regions", default=None,
                        help="regions.json; default <out-dir>/regions.json")
    parser.add_argument("--soft-npz", default=None,
                        help="the soft solution; default <out-dir>/soft.npz")
    parser.add_argument("--truth-result", default=None,
                        help="the post-fence-LG result.json; default "
                             "<out-dir>/result.json")
    parser.add_argument("--tau", type=float, default=None,
                        help="tau for the surrogate; default freeze.json's tau")
    parser.add_argument("--tau-from-npz", action="store_true",
                        help="read tau from the --soft-npz snapshot's own 'tau' "
                             "entry (run_placement_io --snapshot-iters writes one)")
    parser.add_argument("--d-max", type=int, default=None, dest="ignore_net_degree")
    parser.add_argument("--run-flow", action="store_true",
                        help="run run_main_flow --phase soft then --phase fence "
                             "into --out-dir first (requires P-B)")
    parser.add_argument("--dp-seed", type=int, default=None)
    parser.add_argument("--degraded", action="store_true",
                        help="mandatory when the soft solution and the fence LG "
                             "do not come from the same membership")
    parser.add_argument("--degraded-reason", default=
                        "soft snapshot and fence run use different memberships")
    return parser


def compare(rows, truth):
    """Score every surrogate row against the post-fence-LG lower bound."""
    lb = int(truth["hard_lambda_sum"])
    if lb <= 0:
        raise ValueError("truth hard_lambda_sum must be positive to score a "
                         "relative error; got %r" % (lb,))
    out = []
    for row in rows:
        scored = dict(row)
        scored["io_lb_final"] = lb
        scored["io_count_final"] = int(truth["io_count"])
        scored["abs_err"] = float(row["l_io_soft"]) - lb
        scored["rel_err"] = scored["abs_err"] / lb
        scored["closest"] = False
        out.append(scored)
    # first-occurrence tie-break, so the verdict is deterministic
    best = min(range(len(out)), key=lambda i: abs(out[i]["abs_err"]))
    out[best]["closest"] = True
    return out


def render_markdown(rows, *, degraded, reason=None):
    lines = ["| " + " | ".join(ANCHOR_TABLE_COLUMNS) + " |",
             "|" + "---|" * len(ANCHOR_TABLE_COLUMNS)]
    for row in rows:
        cells = []
        for column in ANCHOR_TABLE_COLUMNS:
            value = row[column]
            if isinstance(value, bool):
                cells.append("yes" if value else "")
            elif isinstance(value, float):
                cells.append(("%.4f" if column == "rel_err" else "%.1f") % value)
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    if degraded:
        lines.append("")
        lines.append("**DEGRADED** — %s. This table does not answer design v2 "
                     "sec 7's question and must not be published as its result."
                     % (reason,))
    return "\n".join(lines)


def surrogate_io(nl, placedb, rs_scaled, node_x, node_y, anchor, tau, *,
                 ignore_net_degree, device="cuda"):
    """L_IO and sum_e lambda_e at one anchor, on already-scaled coordinates."""
    import torch
    from ioplace.ops.io_term import (IoTerm, IoTermRef, build_net_node_csr,
                                     build_net_pin_csr)
    from ioplace.ops.soft_assign import rect_table

    rects, r2k = rect_table(rs_scaled)
    csr = build_net_node_csr(nl, ignore_net_degree)
    num_nodes = int(placedb.num_nodes)
    kw = dict(csr=csr, rects=rects, rect2region=r2k, K=rs_scaled.k,
              num_movable=nl.num_movable, num_physical=nl.num_physical,
              num_nodes=num_nodes, device=device, w_mode="unit")
    if anchor == "pin":
        term = IoTermRef(node_anchor="pin", pin_csr=build_net_pin_csr(nl, csr), **kw)
    else:
        term = IoTerm(node_anchor=anchor,
                      node_size_x=(nl.node_size_x if anchor == "center" else None),
                      node_size_y=(nl.node_size_y if anchor == "center" else None),
                      **kw)
    pos = torch.zeros(2 * num_nodes, dtype=torch.float64, device=device)
    pos[:nl.num_physical] = torch.as_tensor(node_x[:nl.num_physical], device=device)
    pos[num_nodes:num_nodes + nl.num_physical] = torch.as_tensor(
        node_y[:nl.num_physical], device=device)
    if anchor == "pin":
        with torch.no_grad():
            # IoTermRef.diagnostics() is deliberately refused under the pin
            # anchor (Task 2); read both numbers off _forward_io instead.
            # lambda_io=1, lambda_margin=0, so L_io here IS the surrogate.
            l_io_t, lam, _ = term._forward_io(*term._split_xy(pos), tau)
            l_io, lambda_sum = float(l_io_t), float(lam.sum())
    else:
        # diagnostics() returns both numbers. It must NOT run inside
        # torch.no_grad(): its grad_share loop takes one backward pass per
        # non-empty degree bucket (io_term.py:399-410), which is a few seconds
        # at tile scale and free information about where the surrogate's
        # gradient lives.
        diag = term.diagnostics(pos, tau)
        l_io, lambda_sum = float(diag["l_io"]), float(diag["soft_lambda_sum"])
    return {"anchor": anchor, "l_io_soft": l_io, "lambda_sum_soft": lambda_sum,
            "n_active": int(term.n_active), "tau": float(tau)}


def _load_soft(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return data


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    regions_json = args.regions or os.path.join(args.out_dir, "regions.json")
    soft_npz = args.soft_npz or os.path.join(args.out_dir, "soft.npz")
    truth_json = args.truth_result or os.path.join(args.out_dir, "result.json")

    if args.run_flow:
        from ioplace.drivers.run_main_flow import run_main_flow
        common = dict(k=args.k, rtype=args.rtype, seed=args.seed,
                      dp_seed=args.dp_seed, node_anchor="center")
        run_main_flow(args.config, args.out_dir, phase="soft", **common)
        run_main_flow(args.config, args.out_dir, phase="fence", **common)

    from ioplace.netlist import load_netlist
    from ioplace.regions import RegionSet
    nl, placedb, params = load_netlist(args.config)
    rs_native = RegionSet.from_json(regions_json)
    from ioplace.artifacts import scaled_region_set
    rs_scaled = scaled_region_set(rs_native, params.shift_factor, params.scale_factor)

    soft = _load_soft(soft_npz)
    shift = np.asarray(soft.get("shift_factor", params.shift_factor), dtype=np.float64)
    scale = float(soft.get("scale_factor", params.scale_factor))
    if "kind" in soft or "shift_factor" in soft:      # an artifacts.save_positions file
        node_x = (np.asarray(soft["node_x"], dtype=np.float64) - shift[0]) * scale
        node_y = (np.asarray(soft["node_y"], dtype=np.float64) - shift[1]) * scale
    else:                                             # a run_placement_io snapshot
        node_x = np.asarray(soft["node_x"], dtype=np.float64)
        node_y = np.asarray(soft["node_y"], dtype=np.float64)

    tau = args.tau
    if tau is None and args.tau_from_npz:
        tau = float(soft["tau"])
    if tau is None:
        with open(os.path.join(args.out_dir, "freeze.json")) as handle:
            tau = float(json.load(handle)["tau"])

    ignore = args.ignore_net_degree
    if ignore is None:
        ignore = int(params.ignore_net_degree)

    rows = [surrogate_io(nl, placedb, rs_scaled, node_x, node_y, anchor, tau,
                         ignore_net_degree=ignore) for anchor in args.anchors]
    with open(truth_json) as handle:
        truth = json.load(handle)
    rows = compare(rows, truth)

    record = {"config": os.path.abspath(args.config), "k": args.k,
              "rtype": args.rtype, "seed": args.seed, "tau": tau,
              "regions_json": os.path.abspath(regions_json),
              "soft_npz": os.path.abspath(soft_npz),
              "truth_result": os.path.abspath(truth_json),
              "degraded": bool(args.degraded),
              "degraded_reason": args.degraded_reason if args.degraded else None,
              "truth": {key: truth.get(key) for key in
                        ("hard_lambda_sum", "io_count", "straddle_cells",
                         "straddle_area_fraction", "straddle_pin_split_nets",
                         "fence_compliance", "fence_compliance_center")},
              "columns": list(ANCHOR_TABLE_COLUMNS), "rows": rows}
    with open(os.path.join(args.out_dir, "anchor_comparison.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    markdown = render_markdown(rows, degraded=args.degraded,
                               reason=args.degraded_reason)
    with open(os.path.join(args.out_dir, "anchor_comparison.md"), "w") as handle:
        handle.write(markdown + "\n")
    print(markdown)
    return record


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the fast tests**

Run: `"$IOPLACE_PYTHON" -m pytest tests/test_anchor_comparison.py -m "not slow" -v`
Expected: PASS — 6 passed.

- [ ] **Step 5: Write the result-doc template**

Create `docs/results/2026-09-19-p-f-anchor-comparison.md`:

```markdown
# P-F — Soft-assign anchor comparison (design v2 §7)

**Status:** template — fill from `runs/anchor_cmp/<case>/anchor_comparison.json`.
**Spec:** `docs/superpowers/specs/2026-09-19-v2-io-aware-redesign-design.md` §7,
"Verification experiment".
**Plan:** `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` Task 7.

## Question

From one soft solution, does the cell-centre anchor predict post-fence-LG IO
better than the node lower-left or the pin anchor? Spec §7 expects centre
closest. **If `pin` is closest, LG displacement exceeds half a cell and the
legaliser must be examined** — that is a finding about LG, not about the anchor.

## Protocol

| item | value |
|---|---|
| case | `mempool_tile_wrap` |
| config | `benchmarks/ispd25/h100/mempool_tile_wrap.json` (fallback: `results/route_gp_20260914/mempool_tile_wrap.json`) |
| K / geometry | 16 / grid 4×4 |
| soft solution | `run_main_flow --phase soft --node-anchor center --dp-seed 1000` → `soft.npz`, `freeze.json` |
| fence LG | `run_main_flow --phase fence --dp-seed 1000` → `placement.npz`, `result.json` |
| τ | `freeze.json["tau"]` (the τ the soft solution actually stopped at) |
| surrogate | `L_IO = Σ_e max(λ_e − 1, 0)`, `w_mode=unit`, `λ_io=1`, no margin |
| truth | `result.json["hard_lambda_sum"]` on the post-fence-LG placement |
| script | `"$IOPLACE_PYTHON" src/scripts/run_anchor_comparison.py --config <cfg> --out-dir runs/anchor_cmp/mempool_tile_wrap --k 16` |
| host | H100 NVL, `CUDA_VISIBLE_DEVICES=3` |

## Result

| anchor | l_io_soft | lambda_sum_soft | io_lb_final | abs_err | rel_err | io_count_final | closest |
|---|---|---|---|---|---|---|---|
| lower_left | | | | | | | |
| center | | | | | | | |
| pin | | | | | | | |

## Straddle context (from the same `result.json`)

| metric | value |
|---|---|
| `straddle_cells` | |
| `straddle_area_fraction` | |
| `straddle_pin_split_nets` | |
| `straddle_wide_cells` | |
| `fence_compliance` (lower_left) | |
| `fence_compliance_center` | |
| `io_delta_at_freeze` | |
| `lg_loss` | |
| identity residual (`io_identity.verify_io_identity`) | |

## Reading

- **Centre closest (expected).** §7's fix is confirmed: the centre is the best
  available predictor of post-LG cell ownership and coincides with the freeze
  rule and fence ownership. Keep `--node-anchor center` as the default. Quote
  `abs_err(lower_left) − abs_err(center)` as the systematic half-cell bias the
  change removes.
- **Pin closest.** Per §7 this means LG moved cells by more than half a cell, so
  the pre-LG straddling state the pin anchor describes survived legalisation.
  Do **not** promote `pin` — it is `(P,K)` and double-counts multi-pin cells.
  Open an LG-displacement investigation instead, and report the median and p99
  `|Δx|`, `|Δy|` between `soft.npz` and `placement.npz`.
- **Lower-left closest.** Unexpected. Check `straddle_cells` first: if it is
  ~0, no anchor can differ materially and the case does not test anything;
  re-run on a case with real straddling.

## Artefacts

| file | sha256 |
|---|---|
| `runs/anchor_cmp/mempool_tile_wrap/soft.npz` | |
| `runs/anchor_cmp/mempool_tile_wrap/placement.npz` | |
| `runs/anchor_cmp/mempool_tile_wrap/result.json` | |
| `runs/anchor_cmp/mempool_tile_wrap/anchor_comparison.json` | |
```

- [ ] **Step 6: Run the GCD smoke test (only if GPU 3 is free)**

Run `nvidia-smi`; if GPU 3 is idle:
`CUDA_VISIBLE_DEVICES=3 "$IOPLACE_PYTHON" -m pytest tests/test_anchor_comparison.py -m slow -v`
Expected: PASS if P-B has landed, otherwise `SKIPPED`.

- [ ] **Step 7: Commit**

```bash
git add src/scripts/run_anchor_comparison.py tests/test_anchor_comparison.py \
        docs/results/2026-09-19-p-f-anchor-comparison.md
git commit -m "feat(experiment): sec 7 anchor comparison, surrogate vs post-fence-LG truth

One soft solution, three anchors, scored against hard_lambda_sum on the
post-fence-LG placement -- the same functional of the same per-net
distinct-region counts, so the comparison is apples to apples; io_count is
context only. --degraded is mandatory when the soft solution and the fence LG do
not share a membership, and stamps the JSON and the markdown so a fallback table
cannot be mistaken for the sec 7 result.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Documentation sync

**Files:**
- Modify: `docs/dev-env.md` (new subsection after "Normalisation module (P-H) flags", line 73)
- Modify: `docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md` (this file — tick the boxes as tasks land)
- Test: none new; `tests/test_m4_report_lint.py` is the only doc gate in the suite and it does not read `dev-env.md`

**Interfaces:**
- Consumes: the flag and field names Tasks 1, 3, 4 and 5 introduced. Nothing consumes this task.

- [x] **Step 1: Add the P-F section to `docs/dev-env.md`**

Insert immediately before `## Installed toolchain` (line 74):

```markdown
### Straddling / anchor (P-F) flag and diagnostics

`ioplace.drivers.run_placement --mode io` (and, once P-B lands,
`ioplace.drivers.run_main_flow`) evaluates the soft region assignment at the
anchor chosen by one flag (v2 design section 7):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--node-anchor` | `center` | `center` evaluates the region SDF at `x+0.5·w, y+0.5·h`, which is what the freeze rule and whole-cell fence ownership both use; `lower_left` is the legacy anchor; `pin` is **rejected by every driver** — it exists only in `ops/io_term.IoTermRef` as a small-scale bias probe, because it costs `(P,K)` instead of `(N,K)` and double-counts a cell carrying two pins of one net |

The anchor is a per-node constant offset applied where `x`/`y` leave `pos`
(`ops/soft_assign.anchor_offsets`), so no tensor shape and no gradient changes;
`FtTerm` inherits it from the `IoTerm` it wraps.

Both evaluators report three straddle diagnostics (`ioplace/straddle.py` fixes
the conventions: closed four-corner box, cell-centre owner, movable cells only,
quadrant area split with wide cells counted separately, pin re-attribution
scored by distinct-region count):

| Field | Meaning |
| --- | --- |
| `straddle_cells` | movable cells whose box `[x,x+w]×[y,y+h]` meets more than one region |
| `straddle_area_fraction` | out-of-owner area over total movable area |
| `straddle_pin_split_nets` | nets whose distinct-pin-region count drops when every straddling cell's pins are re-attributed to that cell's owner |
| `straddle_out_area` / `straddle_movable_area` | the fraction's numerator and denominator |
| `straddle_wide_cells` | cells spanning more than two lattice cells per axis, where the quadrant area split is approximate |

They cost two persistent `(num_physical,)` float64 tensors on the GPU context
and one extra `torch.unique` per `evaluate()`, so both `GpuEvalContext(...,
straddle=False)` and `ctx.evaluate(..., straddle=False)` exist; the in-loop
diagnostic callback in `run_placement_io` uses the per-call form.

`evaluation.npz` is at `SCHEMA_VERSION = 2`: it adds `per_node_straddle`
`(num_physical,)` uint8, `per_net_pin_split` `(num_nets,)` int32 (signed) and a
`metadata["straddle"]` block, and `load_evaluation` still accepts schema 1, so
the historical evidence under `results/` stays pairable. `metadata["straddle"]`
is `None` exactly when the evaluator ran with `straddle=False`.

Parity contract (spec section 9): the integer straddle fields are **bit-exact**
between `evaluator_ref` and `evaluator_gpu` and across `mst_chunk_budget` /
`seg_chunk_budget` / `edge_batch_size`; `straddle_area_fraction`,
`straddle_out_area` and `straddle_movable_area` carry `tree_wl`/`hpwl`'s
`rel <= 1e-12`.

`ioplace.io_identity.verify_io_identity` checks design section 7's identity
`io(final) = io(soft) + io_delta_at_freeze + lg_loss` against IO counts
**re-measured** from `soft.npz` and `placement.npz`, which is what
`main_flow_metrics.io_accounting`'s always-zero residual structurally cannot do.
`ioplace.io_identity.p_f_diagnostics` extracts P-F's five-diagnostic exit
criterion from a `result.json`.
```

- [x] **Step 2: Verify the docs still lint and the suite is green**

Run: `"$IOPLACE_PYTHON" -m pytest -m "not slow"`
Expected: PASS, matching the Task 5 baseline.

- [x] **Step 3: Tick this plan's completed checkboxes and record the P-B state**

If Task 5 Step 7 took the `PENDING` branch, make sure the cross-plan note is in
`docs/superpowers/plans/2026-09-19-v2-p-b-main-flow.md` Task 7's Interfaces
block and that Tasks 6 and 7's slow tests are recorded as `SKIPPED (P-B pending)`
in the commit message rather than as passes.

- [ ] **Step 4: Commit**

```bash
git add docs/dev-env.md docs/superpowers/plans/2026-09-19-v2-p-f-straddling.md
git commit -m "docs: --node-anchor, the straddle diagnostics and evaluation.npz schema 2

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-Review

Run against the spec with fresh eyes after the plan was complete.

**1. Spec coverage.**

| Spec requirement | Task |
|---|---|
| §0 "soft-assign anchor = cell centre (flag `--node-anchor {lower_left,center,pin}`, default center; pin only in IoTermRef)" | 1 (flag, `center`, driver rejection), 2 (the `IoTermRef`-only `pin` arm) |
| §0/§7 "evaluator stays pin-based" | 3 — `evaluate` is untouched on its crossing path; the diagnostics are additive and `test_evaluate_reports_the_diagnostics_and_leaves_the_legacy_fields_alone` pins that |
| §0/§7 "no overlap/straddle penalty" | Global Constraints; no task adds a term |
| §7 "the inconsistency" (`io_term.py:101-110`, `soft_assign.py:19-31`, `evaluator_ref.py:97,113`, `evaluator_gpu.py:303-306`, `fence_inject.py:28-34`) | 1 — the anchor is applied at exactly the `_split_xy` / `forward` sites §7 names, and `region_sdf_l1` is left alone |
| §7 "one-line change at `region_sdf_l1`'s caller; keeps the `(N,K)` shape and the no-double-counting property" | 1 Step 4/5; Task 2's `test_pin_arm_double_counts_a_node_carrying_two_pins_of_one_net` is the evidence for what node anchoring preserves |
| §7 "Why the cell centre" | 3's convention 2 and the plan's `_straddler` test, which shows the centre scoring the crossing the fence LG will realise |
| §7 verification experiment on `mempool_tile` | 7 |
| §7 diagnostic 1 `straddle_cells` (four corners via `region_of_points`) | 3, 4, 5 |
| §7 diagnostic 2 `straddle_area_fraction` | 3, 4, 5 |
| §7 diagnostic 3 `straddle_pin_split_nets` | 3, 4, 5 |
| §7 diagnostic 4 `io_delta_at_freeze` | **P-B Task 3/6** — not re-implemented here; consumed by Task 6's `verify_io_identity`/`p_f_diagnostics` and Task 5's guarded field wiring |
| §7 diagnostic 5 `fence_compliance` | **P-B Task 6** — same |
| §7 identity `io(final) = io(soft) + io_delta_at_freeze + lg_loss` | 6 |
| §9 parity: new evaluator fields bit-exact ref vs GPU across batch sizes and chunk budgets | 4 (`_assert_straddle_equal`, the extended `_assert_batch_invariant_fields`, `test_gpu_straddle_bit_exact_across_batch_sizes_and_chunk_budgets`), narrowed to integer fields per Global Constraints |
| §9 "done" for F: five diagnostics reported, identity closing within ±1 | 6 (`p_f_diagnostics`, `tol=1`), 5 (the fields reach `result.json`) |
| §1 `evaluation.npz` schema | 5 |

**Known gap, stated rather than hidden:** F's exit criterion cannot be *demonstrated* until P-B lands — `io_delta_at_freeze` and `fence_compliance` come from `run_main_flow`, and neither the end-to-end identity test (Task 6 Step 5) nor the end-to-end anchor comparison (Task 7 Step 6) can run without it. Both skip with an explicit reason rather than passing vacuously, and Task 5 Step 7's `PENDING` branch records the outstanding wiring in P-B's own plan. Everything P-F owns is fully testable today.

**Two recorded interpretations of the spec** (both argued in place, neither a silent choice):

1. The class-level default of `node_anchor` on `IoTermRef`/`IoTerm` is `"lower_left"`, not `"center"`: the spec sets the default of the *flag*, and `center` needs node sizes that fourteen existing construction sites do not supply. Task 1 pins both the flag default and the class default with tests.
2. `straddle_pin_split_nets` is scored by distinct-pin-region count (`per_net_lambda`), not by `per_net_crossings`. §7 says "whose crossing count would drop"; `per_net_crossings` is a function of coordinates, which re-attribution does not change, so λ is the only crossing measure the re-attribution can move. `straddle.py`'s convention 5 states the argument.

**2. Placeholder scan.** Searched for `TBD`, `TODO`, `implement later`, `fill in details`, `Similar to Task`, `add appropriate`, `handle edge cases`, `write tests for the above`. No hits. Every Step 1 carries runnable test code and every implementation step carries the exact block to write. The one document that is deliberately empty is `docs/results/2026-09-19-p-f-anchor-comparison.md`, which is a *result* template whose cells the Task 7 campaign fills — its protocol, columns and three decision branches are all fully specified. One drafting artefact was found and fixed during this review: `surrogate_io`'s `lambda_sum` read-back had dead lines and would have run `IoTerm.diagnostics` inside `torch.no_grad()`, where its `grad_share` backward passes fail; it is now a clean `if anchor == "pin"` branch with `diagnostics()` outside any no-grad context.

**3. Type consistency.**

- `node_anchor` is the parameter name in `anchor_offsets`, `IoTermRef`, `IoTerm`, `run_io`, `run_main_flow` and `surrogate_io`; the CLI spelling is `--node-anchor` everywhere (`run_placement.build_parser`, `run_main_flow.build_parser`).
- `anchor_dx`/`anchor_dy` are the buffer names in both term classes; `_anchor_xy` is the method name in both and is called by `FtTerm._evaluate_parts` and `FtTermRef._values`.
- `pin_node`/`pin_offset_x`/`pin_offset_y`/`net_pos` are `PinCsr`'s fields in Task 2's `build_net_pin_csr`, and the buffers `IoTermRef` registers from them are `pin_node`/`pin_dx`/`pin_dy`/`pin_net_idx` — distinct names, deliberately, because the buffers are torch tensors on the term's device while the dataclass holds numpy.
- The six scalars are spelled identically in `straddle.STRADDLE_SCALARS`, `EvalResult`, `_straddle_stats`'s returned dict (which is splatted straight into `EvalResult`, so a typo is a `TypeError` at the first GPU evaluation, not a silent zero), `_pack_straddle_metrics`, `RESULT_FIELDS`, `MAIN_FLOW_RESULT_FIELDS` and `metadata["straddle"]`.
- The two arrays are `per_node_straddle` and `per_net_pin_split` in `StraddleStats`, `EvalResult`, `_straddle_stats`, `save_evaluation`/`load_evaluation` and every test.
- `straddle` is the keyword on `evaluator_ref.evaluate`, `GpuEvalContext.__init__`, `GpuEvalContext.evaluate` and `evaluate_gpu` — same name, same meaning (compute the diagnostics), with the context-level form additionally controlling whether the size tensors are allocated.
- `verify_io_identity(recorded, measured, *, tol=1)` and `p_f_diagnostics(result)` are used with those exact signatures in `tests/test_io_identity.py` and in Task 7's result-doc protocol.
- `ANCHOR_TABLE_COLUMNS` is the single source of the eight column names, asserted in `test_markdown_uses_the_documented_column_order` and reproduced verbatim in the result-doc template's table header.
