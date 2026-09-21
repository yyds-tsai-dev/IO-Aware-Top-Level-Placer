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

*** Fix round 2 (adversarial review): the single-scalar `compare()` below is
DEGENERATE as a predictive metric. All three anchors' l_io_soft sit on the same
side of hard_lambda_sum (over 136k nets, over- and under-counting cancels
before scoring ever happens), so `argmin |l_io_soft - hard_lambda_sum|` is
identical to `argmin l_io_soft` -- a magnitude comparison, not a prediction
comparison. Worse: the truth (evaluator_gpu.py's per_net_lambda, a popcount
over actual PIN coordinates -- NOT the `node_region_convention:
lower_left_position` in export/evaluation.py, which governs a different,
unrelated array) is itself pin-anchored, so scoring `pin` against it with a
magnitude metric can penalise `pin` for a bias that has nothing to do with LG
displacement. `net_l1_report()` below is the metric fix: a per-net paired L1,
Σ_e |lambda_soft_e - lambda_hard_e| restricted to 2<=deg<100, computed
whenever `<out-dir>/evaluation.npz` is available. `main()` computes both and
`render_markdown()` prints an unconditional caveat next to the scalar table
so a reader cannot mistake `closest` there for the sec 7 answer on its own.
See docs/results/2026-09-19-p-f-anchor-comparison.md for the worked example
and the reasoning in full. ***
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


def render_markdown(rows, *, degraded, reason=None, l1=None):
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
    # Fix round 2: this caveat is unconditional (not just under --degraded) --
    # `closest` above is argmin over a single shared scalar (hard_lambda_sum),
    # and every anchor's l_io_soft sits on the same side of it, so this ranks
    # by raw magnitude (argmin l_io_soft), not by prediction accuracy. See
    # net_l1_report()/the section below for the metric that actually measures
    # prediction, and this module's docstring for why the scalar is degenerate.
    lines.append("")
    lines.append("**Note:** `closest` above is `argmin |l_io_soft - hard_lambda_sum|` "
                 "over a single shared scalar; since every anchor's l_io_soft sits on "
                 "the same side of hard_lambda_sum, this is equivalent to "
                 "`argmin l_io_soft` and does not by itself show which anchor PREDICTS "
                 "the post-fence-LG truth (design v2 sec 7 fix round 2 -- see the "
                 "per-net L1 section below, or docs/results/2026-09-19-p-f-anchor-"
                 "comparison.md, for the metric that does).")
    if degraded:
        lines.append("")
        lines.append("**DEGRADED** -- %s. This table does not answer design v2 "
                     "sec 7's question and must not be published as its result."
                     % (reason,))
    if l1 is not None:
        lines.append("")
        lines.append("### Per-net L1 (the predictive metric)")
        lines.append("")
        lines.append("Sum_e |lambda_soft_e - lambda_hard_e|, restricted to "
                     "2 <= net_degree < %d, lambda_hard read from "
                     "evaluation.npz's per_net_lambda (PIN-anchored truth)."
                     % l1["deg_hi"])
        lines.append("")
        lines.append("| anchor | l1_abs_err | band_nets |")
        lines.append("|---|---|---|")
        for anchor, value in l1["l1"].items():
            lines.append("| %s | %.1f | %d |" % (anchor, value, l1["band_nets"]))
        boot = l1.get("bootstrap_center_vs_lower_left")
        if boot is not None:
            lines.append("")
            lines.append(
                "Paired bootstrap (n_boot=%d, seed=%d) over per-net |error|, "
                "d_i = |err_center_i| - |err_lower_left_i| (negative => centre "
                "better): mean(d)=%.6g, sum(d)=%.6g, 95%% CI=[%.6g, %.6g] (%s)."
                % (boot["n_boot"], boot["seed"], boot["mean_d"], boot["sum_d"],
                   boot["ci95"][0], boot["ci95"][1],
                   "excludes zero" if boot["excludes_zero"] else "includes zero"))
    return "\n".join(lines)


def _io_term_lam(term, pos, tau):
    """Per-active-net soft lambda for an IoTerm instance (the 'lower_left'/
    'center' anchors), ordered exactly like `term.net_idx`/the CSR's
    `net_ids` -- i.e. the order surrogate_io's caller must scatter it back
    into full net-id space with.

    This replicates IoTerm's own FWD-1/FWD-2 (`_IoFn.forward`, io_term.py)
    under no_grad. `IoTerm.diagnostics()` computes exactly this internally
    but only returns the aggregate `soft_lambda_sum` -- it was never asked to
    return the per-net array before the sec 7 fix-round L1 rescoring needed
    one, so this is duplicated here rather than changing io_term.py's public
    contract for a script-only need."""
    import torch
    from ioplace.ops.soft_assign import _chunks, chunk_p_ell, region_sdf_l1, softmax_stats
    with torch.no_grad():
        x = pos[:term.num_physical]
        y = pos[term.num_nodes:term.num_nodes + term.num_physical]
        x, y = term._anchor_xy(x, y)
        rects = term.rects.to(dtype=x.dtype)
        m, t, am = softmax_stats(x, y, rects, term.rect2region, term.K, tau, chunk=term.k_chunk)
        lam = torch.zeros(term.n_active, dtype=torch.float64, device=x.device)
        for lo, hi in _chunks(term.K, term.k_chunk):
            sdf_c = region_sdf_l1(x, y, rects, term.rect2region, lo, hi)
            _, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
            S_c = torch.zeros((term.n_active, hi - lo), dtype=torch.float64,
                              device=x.device).index_add_(0, term.net_idx,
                                                          ell_c[term.node_idx].double())
            lam = lam + (-torch.expm1(S_c)).sum(dim=1)
    return lam


def surrogate_io(nl, placedb, rs_scaled, node_x, node_y, anchor, tau, *,
                 ignore_net_degree, device="cuda", return_per_net=False):
    """L_IO and sum_e lambda_e at one anchor, on already-scaled coordinates.

    return_per_net=True additionally returns `lam_active` (the per-active-net
    soft lambda, float64 numpy array) and `net_ids` (the netlist net ids each
    entry of `lam_active` belongs to, ascending, from the same NetCsr) -- fix
    round 2's input to net_l1_report(). It costs one extra chunked pass for
    the node anchors (diagnostics() alone doesn't expose the per-net array);
    the pin anchor gets it for free since _forward_io already returns it.
    """
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
    lam = None
    if anchor == "pin":
        with torch.no_grad():
            # IoTermRef.diagnostics() is deliberately refused under the pin
            # anchor (Task 2); read both numbers off _forward_io instead.
            # lambda_io=1, lambda_margin=0, so L_io here IS the surrogate.
            l_io_t, lam, _ = term._forward_io(*term._split_xy(pos), tau)
            l_io, lambda_sum = float(l_io_t), float(lam.sum())
    elif return_per_net:
        lam = _io_term_lam(term, pos, tau)
        l_io = float((term.w * (lam - 1.0).clamp(min=0)).sum())
        lambda_sum = float(lam.sum())
    else:
        # diagnostics() returns both numbers. It must NOT run inside
        # torch.no_grad(): its grad_share loop takes one backward pass per
        # non-empty degree bucket (io_term.py:399-410), which is a few seconds
        # at tile scale and free information about where the surrogate's
        # gradient lives.
        diag = term.diagnostics(pos, tau)
        l_io, lambda_sum = float(diag["l_io"]), float(diag["soft_lambda_sum"])
    out = {"anchor": anchor, "l_io_soft": l_io, "lambda_sum_soft": lambda_sum,
           "n_active": int(term.n_active), "tau": float(tau)}
    if return_per_net:
        out["lam_active"] = lam.detach().cpu().numpy()
        out["net_ids"] = np.asarray(csr.net_ids)
    return out


def net_l1_report(nl, placedb, out_dir, per_net, *, deg_lo=2, deg_hi=100,
                  n_boot=10000, boot_seed=1000):
    """Fix round 2's metric fix: per-net paired L1 against evaluation.npz's
    per-net hard truth, restricted to the same degree band build_net_node_csr
    uses. Unlike `compare()`'s single scalar, this cannot degenerate into
    "smallest number wins": each net is scored against its OWN hard truth, so
    over- and under-counting across nets no longer cancel before scoring.

    The truth (`per_net_lambda`, evaluator_gpu.py's popcount over PIN
    coordinates) is pin-anchored, NOT lower-left -- `node_region_convention:
    lower_left_position` in evaluation.npz's metadata governs only the
    separate `node_region` export array, not lambda.

    `per_net`: {anchor: (net_ids, lam_active)} from surrogate_io(...,
    return_per_net=True) -- same soft solution, same tau, every anchor.

    Returns None if <out_dir>/evaluation.npz does not exist (e.g. the
    --degraded fallback path, which never runs the real evaluator). Raises
    ValueError (via load_evaluation's own net_names check) if the net order
    this call's `nl`/`placedb` were loaded with does not match
    evaluation.npz's -- the per-net pairing is meaningless otherwise, and
    fix round 2's finding was exactly that this must be checked, not assumed.
    """
    path = os.path.join(out_dir, "evaluation.npz")
    if not os.path.exists(path):
        return None
    from ioplace.export.evaluation import load_evaluation
    data = load_evaluation(path, net_names=placedb.net_names)
    hard = np.asarray(data["per_net_lambda"], dtype=np.float64)
    degrees = np.asarray(data["net_degrees"], dtype=np.int64)
    band = (degrees >= deg_lo) & (degrees < deg_hi)

    full = {}
    for anchor, (net_ids, lam_active) in per_net.items():
        arr = np.ones(nl.num_nets, dtype=np.float64)  # single-node nets: trivially 1 region
        arr[np.asarray(net_ids)] = lam_active
        full[anchor] = arr
    abs_err = {anchor: np.abs(arr - hard)[band] for anchor, arr in full.items()}
    l1 = {anchor: float(err.sum()) for anchor, err in abs_err.items()}

    boot = None
    if "center" in abs_err and "lower_left" in abs_err:
        d = abs_err["center"] - abs_err["lower_left"]
        rng = np.random.default_rng(boot_seed)
        n = d.shape[0]
        means = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            means[b] = d[idx].mean()
        ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
        boot = {"mean_d": float(d.mean()), "sum_d": float(d.sum()),
                "ci95": [float(ci_lo), float(ci_hi)],
                "excludes_zero": bool(not (ci_lo <= 0.0 <= ci_hi)),
                "n_boot": int(n_boot), "seed": int(boot_seed)}
    return {"deg_lo": deg_lo, "deg_hi": deg_hi, "band_nets": int(band.sum()),
           "l1": l1, "bootstrap_center_vs_lower_left": boot}


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

    # Fix round 2: compute the per-net array whenever the real evaluator
    # ran (evaluation.npz present) so net_l1_report below has something to
    # score against -- the --degraded fallback path never writes one, so
    # return_per_net there just costs nothing extra and l1_report is None.
    want_per_net = os.path.exists(os.path.join(args.out_dir, "evaluation.npz"))
    rows_raw = [surrogate_io(nl, placedb, rs_scaled, node_x, node_y, anchor, tau,
                             ignore_net_degree=ignore, return_per_net=want_per_net)
               for anchor in args.anchors]
    per_net = {r["anchor"]: (r["net_ids"], r["lam_active"]) for r in rows_raw
              if want_per_net}
    rows = [{k: v for k, v in r.items() if k not in ("lam_active", "net_ids")}
           for r in rows_raw]
    with open(truth_json) as handle:
        truth = json.load(handle)
    rows = compare(rows, truth)

    l1_report = net_l1_report(nl, placedb, args.out_dir, per_net,
                              deg_hi=ignore) if want_per_net else None

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
              "columns": list(ANCHOR_TABLE_COLUMNS), "rows": rows,
              "l1": l1_report}
    with open(os.path.join(args.out_dir, "anchor_comparison.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    markdown = render_markdown(rows, degraded=args.degraded,
                               reason=args.degraded_reason, l1=l1_report)
    with open(os.path.join(args.out_dir, "anchor_comparison.md"), "w") as handle:
        handle.write(markdown + "\n")
    print(markdown)
    return record


if __name__ == "__main__":
    main()
