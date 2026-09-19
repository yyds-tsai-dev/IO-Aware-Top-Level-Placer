"""Route-derived position proposals with post-legalization cost-difference gates.

The scalable signal uses a fixed cohort of degree-two nets. It is a search
proposal, not a derivative of integer crossings or a prediction of router IO.
"""
import hashlib
import numpy as np
from ioplace.route_eval.budgeted import BudgetedRouter


class RouteFeedback:
    def __init__(self, nl, region_grid, node_x, node_y, *, wirelength_budget=.05,
                 max_displacement_cells=8., chunk_size=65536):
        if (not np.isfinite(wirelength_budget) or wirelength_budget < 0
                or not np.isfinite(max_displacement_cells) or max_displacement_cells <= 0
                or chunk_size < 1):
            raise ValueError("invalid route feedback settings")
        self.nl, self.rg = nl, region_grid
        self.router = BudgetedRouter(region_grid)
        self.budget, self.cap, self.chunk = wirelength_budget, max_displacement_cells, chunk_size
        ids = np.flatnonzero(nl.net_degrees == 2)
        offsets = nl.flat_net2pin_start[ids]
        pins = np.column_stack((nl.flat_net2pin[offsets], nl.flat_net2pin[offsets+1]))
        nodes = nl.pin2node[pins]
        points = np.stack((np.asarray(node_x)[nodes]+nl.pin_offset_x[pins],
                           np.asarray(node_y)[nodes]+nl.pin_offset_y[pins]), axis=2)
        low, high = np.array(region_grid.die[:2]), np.array(region_grid.die[2:])
        inside = np.isfinite(points).all((1, 2)) & (points >= low).all((1, 2)) & (points <= high).all((1, 2))
        self.net_ids, self.pins = ids[inside], pins[inside]
        self.cohort = dict(eligible_nets=int(inside.sum()), excluded_other_degree=int(nl.num_nets-len(ids)),
            excluded_outside_die=int((~inside).sum()),
            net_ids_sha256=hashlib.sha256(self.net_ids.tobytes()).hexdigest(),
            scope="fixed baseline in-die degree-two cohort; all other nets retained in full-net gate")

    def evaluate(self, x, y, *, signal=False):
        nl = self.nl
        delta = np.zeros((nl.num_movable, 2)) if signal else None
        weight = np.zeros(nl.num_movable) if signal else None
        baseline_io = opportunity_io = 0
        base_length = length = 0.
        improved = 0
        for start in range(0, len(self.pins), self.chunk):
            pins = self.pins[start:start+self.chunk]
            nodes = nl.pin2node[pins]
            points = np.stack((np.asarray(x)[nodes]+nl.pin_offset_x[pins],
                               np.asarray(y)[nodes]+nl.pin_offset_y[pins]), axis=2)
            route = self.router.route_edges(points[:, 0], points[:, 1], wirelength_budget=self.budget)
            baseline_io += int(route["baseline_crossings"].sum())
            opportunity_io += int(route["crossings"].sum())
            base_length += float(route["baseline_wirelength"].sum())
            length += float(route["wirelength"].sum())
            savings = route["baseline_crossings"]-route["crossings"]
            improved += int(np.count_nonzero(savings))
            if signal:
                # First nonzero bend from either terminal. Zero-length bend
                # slots occur for L paths and must not swallow the signal.
                for end, order in ((0, (1, 2)), (1, (2, 1))):
                    target = route["points"][:, order[0]].copy()
                    same = (target == points[:, end]).all(axis=1)
                    target[same] = route["points"][same, order[1]]
                    movable = (nodes[:, end] < nl.num_movable) & (savings > 0)
                    ids = nodes[movable, end]
                    np.add.at(delta, ids, (target[movable]-points[movable, end])*savings[movable, None])
                    np.add.at(weight, ids, savings[movable])
        metrics = dict(legacy_cohort_io=baseline_io, opportunity_io=opportunity_io,
            opportunity_wirelength=length, cohort_manhattan_wirelength=base_length,
            reroute_savings=baseline_io-opportunity_io, improved_route_nets=improved)
        if not signal:
            return metrics
        active = weight > 0
        delta[active] /= weight[active, None]
        # Cap displacement in lattice units, preserving direction.
        lattice_delta = delta / [self.rg.cell_w, self.rg.cell_h]
        norm = np.linalg.norm(lattice_delta, axis=1)
        delta *= np.minimum(1., self.cap / np.maximum(norm, 1e-300))[:, None]
        return metrics, delta


def accept_move(original, incumbent, proposal, control, *, hpwl_budget=.05):
    """Gate comparable full-net metrics and a fixed-cohort route evaluation.

    Route pair is lexicographic (IO, length); full legacy IO must also improve
    over both incumbent and extra-legalization control. Budgets are always
    relative to the ORIGINAL baseline, preventing round-by-round drift.
    """
    if not np.isfinite(hpwl_budget) or hpwl_budget < 0:
        raise ValueError("finite nonnegative HPWL budget required")
    reasons = []
    keys = ("hpwl", "io_count", "opportunity_io", "opportunity_wirelength")
    for name, metric in (("proposal", proposal), ("control", control)):
        if metric is None or not all(np.isfinite(metric[k]) and metric[k] >= 0 for k in keys):
            reasons.append(name+"_invalid")
    if reasons:
        return dict(accepted=False, reasons=reasons)
    if proposal["hpwl"] > original["hpwl"] * (1+hpwl_budget):
        reasons.append("original_full_hpwl_budget")
    if proposal["opportunity_wirelength"] > original["opportunity_wirelength"] * (1+hpwl_budget):
        reasons.append("original_route_length_budget")
    pair = lambda m: (m["opportunity_io"], m["opportunity_wirelength"])
    for name, reference in (("incumbent", incumbent), ("control", control)):
        if proposal["io_count"] >= reference["io_count"]:
            reasons.append("no_legacy_io_improvement_over_"+name)
        if pair(proposal) >= pair(reference):
            reasons.append("no_route_pair_improvement_over_"+name)
    return dict(accepted=not reasons, reasons=reasons)


def closed_loop(feedback, x, y, *, legalize, is_legal, full_metrics,
                rounds=2, fractions=(.25, .5, 1.), hpwl_budget=.05, checkpoint=None, proposal_mode="bend", max_active=512):
    """Legalizer and full evaluator adapters permit DREAMPlace or fixture use.

    Every candidate starts from the same incumbent; rebuild routes after every
    legalization and accepted round. Failed/illegal proposals never mutate it.
    checkpoint(label,x,y,metrics) is called for baseline, controls, and accepted
    rounds, so independent routing can assess representative checkpoints.
    """
    if max_active < 0:
        raise ValueError("nonnegative active limit required")
    if proposal_mode not in ("bend", "cost_delta_swap"):
        raise ValueError("unknown proposal mode")
    if rounds < 0 or not fractions or any(not np.isfinite(f) or not 0 < f <= 1 for f in fractions):
        raise ValueError("nonnegative rounds and fractions in (0,1] required")
    if not np.isfinite(hpwl_budget) or hpwl_budget < 0:
        raise ValueError("finite nonnegative HPWL budget required")
    x, y = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True)
    nm, n = feedback.nl.num_movable, feedback.nl.num_physical
    fixed = x[nm:].copy(), y[nm:].copy()
    def valid(a, b):
        return (a.shape == b.shape == (n,) and np.isfinite(a).all() and np.isfinite(b).all()
                and np.array_equal(a[nm:], fixed[0]) and np.array_equal(b[nm:], fixed[1])
                and is_legal(a, b))
    def evaluate(a, b):
        return {**full_metrics(a, b), **feedback.evaluate(a, b)}
    if not valid(x, y):
        raise ValueError("legal finite baseline with unchanged fixed nodes required")
    original = incumbent = evaluate(x, y)
    if not all(np.isfinite(original[k]) and original[k]>=0 for k in
               ("hpwl", "io_count", "opportunity_io", "opportunity_wirelength")):
        raise ValueError("finite nonnegative baseline metrics required")
    history = []
    if checkpoint:
        checkpoint("baseline", x, y, incumbent)
    for iteration in range(rounds):
        _, direction = feedback.evaluate(x, y, signal=True)
        control_x, control_y = legalize(x.copy(), y.copy())
        control = None
        if valid(control_x, control_y):
            try:
                control = evaluate(control_x, control_y)
            except ValueError:
                pass
        if checkpoint and control is not None:
            checkpoint(f"round{iteration}_control", control_x, control_y, control)
        record = dict(round=iteration, active_nodes=int(np.count_nonzero(np.any(direction != 0, axis=1))),
                      incumbent=incumbent, control=control, proposals=[])
        best = None
        for fraction in fractions:
            detail = {}
            if proposal_mode == "cost_delta_swap":
                xx, yy, detail = cost_delta_swaps(feedback,x,y,direction,fraction,max_active=max_active)
                # Equal-size swaps already preserve legal rectangles. Verify
                # explicitly, without introducing unrelated legalizer motion.
            else:
                xx, yy = x.copy(), y.copy()
                xx[:nm] += fraction*direction[:, 0]
                yy[:nm] += fraction*direction[:, 1]
                xx[:nm] = np.clip(xx[:nm], feedback.nl.xl, feedback.nl.xh-feedback.nl.node_size_x[:nm])
                yy[:nm] = np.clip(yy[:nm], feedback.nl.yl, feedback.nl.yh-feedback.nl.node_size_y[:nm])
                xx, yy = legalize(xx, yy)
            metric = None
            if valid(xx, yy):
                try:
                    metric = evaluate(xx, yy)
                except ValueError:  # fixed cohort endpoint left the routing die
                    pass
            verdict = accept_move(original, incumbent, metric, control, hpwl_budget=hpwl_budget)
            record["proposals"].append(dict(fraction=fraction, metrics=metric, detail=detail, **verdict))
            if checkpoint and metric is not None:
                checkpoint(f"round{iteration}_candidate{fraction:g}", xx, yy, metric)
            if verdict["accepted"]:
                rank = (metric["opportunity_io"], metric["opportunity_wirelength"], metric["io_count"])
                if best is None or rank < best[0]:
                    best = (rank, xx.copy(), yy.copy(), metric, fraction)
        record["accepted_fraction"] = best[-1] if best else None
        history.append(record)
        if best is None:
            break
        _, x, y, incumbent, _ = best
        if checkpoint:
            checkpoint(f"round{iteration}_accepted", x, y, incumbent)
    return x, y, dict(original=original, final=incumbent, rounds=history, cohort=feedback.cohort,
        accepted_rounds=sum(r["accepted_fraction"] is not None for r in history),
        routing_verified=False, actual_router_io=None, proposal_mode=proposal_mode)


def cost_delta_swaps(feedback, x, y, direction, fraction, *, max_active=512, neighbors=8):
    """Bounded legal-size swaps steered by bends, evaluated on all incident nets.

    Swapping equal-size movable rectangles preserves geometric legality before
    the external legality check. Incident topology/routes are rebuilt for every
    trial; pins of the same cell/net move together. Degree>256 incidents are
    excluded rather than mis-evaluated. This is a CPU refinement, not O(N) GP.
    """
    from scipy.spatial import cKDTree
    from ioplace.evaluator_ref import net_mst_edges
    nl, rg, router = feedback.nl, feedback.rg, feedback.router
    direction = np.asarray(direction, dtype=float)
    if (max_active < 0 or neighbors < 1 or not np.isfinite(fraction) or not 0 < fraction <= 1
            or direction.shape != (nl.num_movable, 2) or not np.isfinite(direction).all()):
        raise ValueError("invalid swap proposal settings or direction")
    nm = nl.num_movable
    xx, yy = np.array(x,copy=True), np.array(y,copy=True)
    active = np.flatnonzero(np.any(direction != 0,axis=1))
    active = sorted(active,key=lambda i:(-float(np.linalg.norm(direction[i])),int(i)))[:max_active]
    nodes = nl.pin2node
    order = np.argsort(nodes,kind='stable')
    start = np.r_[0,np.cumsum(np.bincount(nodes,minlength=nl.num_physical))]
    cohort = set(map(int,feedback.net_ids))
    tree = cKDTree(np.column_stack((x[:nm],y[:nm])))
    moved=set();accepted=[];attempts=0;unsupported=0
    def metrics(ids):
        legacy=opp=0;hpwl=length=0.
        for net in ids:
            pins=nl.flat_net2pin[nl.flat_net2pin_start[net]:nl.flat_net2pin_start[net+1]]
            px=xx[nodes[pins]]+nl.pin_offset_x[pins];py=yy[nodes[pins]]+nl.pin_offset_y[pins]
            if len(pins)<2:continue
            hpwl+=float(np.ptp(px)+np.ptp(py))
            edges=net_mst_edges(px,py);a=np.column_stack((px,py))[edges[:,0]];b=np.column_stack((px,py))[edges[:,1]]
            legacy+=int((router._h(a[:,0],b[:,0],a[:,1])+router._v(a[:,1],b[:,1],b[:,0])).sum())
            if int(net) in cohort:
                route=router.route_edges(a,b,wirelength_budget=feedback.budget)
                opp+=int(route['crossings'].sum());length+=float(route['wirelength'].sum())
        return legacy,opp,length,hpwl
    for i in active:
        if i in moved:continue
        target=np.array([xx[i],yy[i]])+direction[i]*fraction
        _,candidates=tree.query(target,k=min(neighbors,nm))
        best=None
        for j in np.atleast_1d(candidates):
            j=int(j)
            if i==j or j in moved or nl.node_size_x[i]!=nl.node_size_x[j] or nl.node_size_y[i]!=nl.node_size_y[j]:continue
            displacement=np.array([xx[j]-xx[i],yy[j]-yy[i]])
            if (np.dot(displacement,direction[i])<=0 or
                    np.linalg.norm(displacement/[rg.cell_w,rg.cell_h])>feedback.cap*fraction):continue
            pins=np.r_[order[start[i]:start[i+1]],order[start[j]:start[j+1]]]
            ids=np.unique(nl.pin2net[pins])
            if np.any(nl.net_degrees[ids]>256):unsupported+=1;continue
            before=metrics(ids)
            xx[i],xx[j]=xx[j],xx[i];yy[i],yy[j]=yy[j],yy[i]
            try:
                after=metrics(ids)
            except ValueError:
                after=None
            finally:
                xx[i],xx[j]=xx[j],xx[i];yy[i],yy[j]=yy[j],yy[i]
            attempts+=1
            if after is None:continue
            if after[0]<before[0] and after[1:3]<=before[1:3] and after[3]<=before[3]*1.05:
                rank=tuple(a-b for a,b in zip(after,before))
                if best is None or rank<best[0]:best=(rank,j,before,after)
        if best:
            _,j,before,after=best
            xx[i],xx[j]=xx[j],xx[i];yy[i],yy[j]=yy[j],yy[i]
            moved.update((int(i),j))
            accepted.append(dict(nodes=[int(i),j],before=list(before),after=list(after)))
    return xx,yy,dict(attempted=attempts,unsupported_incidents=unsupported,
                     active_limit=max_active,accepted_swaps=accepted)
