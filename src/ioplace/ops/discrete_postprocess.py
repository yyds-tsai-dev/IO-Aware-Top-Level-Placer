"""Legal acceptance for CE/refinement proposals, with an untouched incumbent."""
import hashlib
import time
import numpy as np


def acceptance(incumbent,proposal,*,legal,fixed_unchanged,control=None,control_valid=True,hpwl_ratio=1.01):
    """All metric arguments must be full-net evaluations of legalizer outputs."""
    io_scale=max(incumbent["io_count"],1);ft_scale=max(incumbent["ft_count"],1)
    score=lambda value:value["io_count"]/io_scale+value["ft_count"]/ft_scale
    baseline_score=score(incumbent)
    threshold=min(baseline_score,score(control)) if control is not None else baseline_score
    reason=[]
    if not control_valid:reason.append("legalizer_control_invalid")
    if not legal:reason.append("proposal_illegal")
    if not fixed_unchanged:reason.append("fixed_nodes_changed")
    if proposal is None:
        return dict(accepted=False,reasons=[*reason,"proposal_not_evaluable"],
            incumbent_score=baseline_score,proposal_score=None,acceptance_score_threshold=threshold,
            hpwl_ratio_guard=hpwl_ratio,metric_scope="full netlist after legalization")
    if not np.isfinite(proposal["hpwl"]) or proposal["hpwl"]>hpwl_ratio*incumbent["hpwl"]:
        reason.append("full_hpwl_guard")
    if not np.isfinite(score(proposal)) or score(proposal)>=threshold-1e-12:
        reason.append("no_full_io_ft_improvement_over_incumbent_and_lg_control")
    return dict(accepted=not reason,reasons=reason,incumbent_score=baseline_score,
                proposal_score=score(proposal),acceptance_score_threshold=threshold,
                hpwl_ratio_guard=hpwl_ratio,metric_scope="full netlist after legalization")


def run_discrete_postprocess(nl,regions,placer,placedb,params,legalize,context,*,mode,max_active=65536):
    import torch
    from ioplace.ops.discrete_placement import prepare_discrete,decode_placement,refine_placement
    if mode not in ("ce","refine","ce_refine"):
        raise ValueError("unknown discrete postprocess mode")
    started=time.perf_counter()
    n,nm,nall=nl.num_physical,nl.num_movable,placedb.num_nodes
    incumbent=placer.pos[0].detach().clone()
    x=incumbent[:n].cpu().numpy().astype(np.float64)
    y=incumbent[nall:nall+n].cpu().numpy().astype(np.float64)
    def metrics(position):
        xx=position[:n].detach().cpu().numpy().astype(np.float64)
        yy=position[nall:nall+n].detach().cpu().numpy().astype(np.float64)
        result=context.evaluate(xx,yy)
        return xx,yy,dict(io_count=result.io_count,ft_count=result.ft_count,hpwl=result.hpwl,
                         io_rg=result.io_rg,ft_rg=result.ft_rg)
    def fixed_ok(position):
        return bool(torch.equal(position[nm:n],incumbent[nm:n]) and
                    torch.equal(position[nall+nm:nall+n],incumbent[nall+nm:nall+n]))
    def legal(position):
        return bool(placer.op_collections.legality_check_op(position))
    def finite_position(position):
        return bool(torch.is_tensor(position) and position.shape==incumbent.shape
                    and torch.isfinite(position).all())
    with torch.no_grad():
        if not finite_position(incumbent) or not legal(incumbent):
            raise ValueError("M5 requires a legal GP+LG incumbent")
        _,_,baseline=metrics(incumbent)
        # Account for a possible improvement caused only by an extra legalizer call.
        control=legalize(incumbent.clone())
        control_valid=finite_position(control) and legal(control) and fixed_ok(control)
        _,_,control_metrics=metrics(control) if control_valid else (None,None,None)
    prepared=prepare_discrete(nl,regions,x,y,max_active=max_active)
    stages={}
    if mode in ("ce","ce_refine"):
        xx,yy,stages["ce"]=decode_placement(prepared,x,y,baseline["io_count"],baseline["ft_count"])
    else:xx,yy=x.copy(),y.copy()
    if mode in ("refine","ce_refine"):
        xx,yy,stages["refine"]=refine_placement(prepared,xx,yy,baseline["io_count"],baseline["ft_count"])
    with torch.no_grad():
        candidate=incumbent.clone()
        candidate[:nm]=torch.as_tensor(xx[:nm],dtype=candidate.dtype,device=candidate.device)
        candidate[nall:nall+nm]=torch.as_tensor(yy[:nm],dtype=candidate.dtype,device=candidate.device)
        proposal=legalize(candidate)
        proposal_finite=finite_position(proposal)
        proposal_legal=proposal_finite and legal(proposal)
        proposed_x,proposed_y,proposal_metrics=metrics(proposal) if proposal_legal else (None,None,None)
        verdict=acceptance(baseline,proposal_metrics,legal=proposal_legal,fixed_unchanged=proposal_finite and fixed_ok(proposal),
                           control=control_metrics,control_valid=control_valid)
        chosen=proposal if verdict["accepted"] else incumbent
        placer.pos[0].data.copy_(chosen)
        chosen_x=chosen[:nm].detach().cpu().numpy().copy()
        chosen_y=chosen[nall:nall+nm].detach().cpu().numpy().copy()
        placedb.apply(params,chosen_x,chosen_y)
    final_x,final_y,final_metrics=metrics(chosen)
    assigned=prepared["region_grid"].region_of_points(final_x,final_y)
    digest=hashlib.sha256(np.ascontiguousarray(assigned).tobytes()).hexdigest()
    def compact(stage):
        result={key:value for key,value in stage.items() if not isinstance(value,np.ndarray)
                and key not in ("metrics_after",)}
        trace=stage["objective_trace"]
        result.update(objective_initial=float(trace[0]),objective_final=float(trace[-1]),steps=len(trace)-1)
        return result
    report=dict(mode=mode,baseline=baseline,legalizer_control=control_metrics,
        legalizer_control_valid=control_valid,
        legalizer_control_positions_identical=bool(finite_position(control) and torch.equal(control,incumbent)),
        proposal=proposal_metrics,final=final_metrics,**verdict,
        preparation=prepared["diagnostics"],stages={key:compact(value) for key,value in stages.items()},
        final_region_assignment_sha256=digest,elapsed_s=time.perf_counter()-started,
        final_assignment_source="accepted post-legalization coordinates",
        unchanged_incumbent_retained=not verdict["accepted"])
    arrays=dict(active_nodes=prepared["active"],final_region_assignment=assigned)
    for name,stage in stages.items():
        for key,value in stage.items():
            if isinstance(value,np.ndarray):arrays[name+"_"+key]=value
    return report,arrays
