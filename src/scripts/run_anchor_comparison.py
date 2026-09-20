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
        lines.append("**DEGRADED** -- %s. This table does not answer design v2 "
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
