"""Iteration-boundary publication of the paper's frozen-Steiner GP objective."""
import hashlib
import torch
from ioplace.dp_hook import refresh_nesterov_secant


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class SteinerGPController:
    """Attach as an extra PlaceObj term and as the post-step callback.

    The first step is WA warmup. Subsequent steps are wrapped on this optimizer
    instance only. Rebuild before step/line search, never during forward/backward.
    Both Nesterov secant endpoints are recomputed with the published objective.
    """
    def __init__(self,term,placer,*,enabled=True,start=200,rebuild_every=1,match_refresh=True):
        if start<1 or rebuild_every<1:raise ValueError('positive start/rebuild interval required')
        self.term,self.placer=term,placer
        self.enabled,self.start,self.every=enabled,start,rebuild_every
        self.active=False;self.gamma=None;self.next_iteration=1
        self.cache_generation=0;self.trace=[];self.rebuilds=[];self.forward_generations=[]
        self.optimizer=None;self.original_step=None
        self.match_refresh=match_refresh;self.cache_refreshes=0
        self.gradient_audit=None

    def audit_gradient_sum(self,pos):
        """Check the actual PlaceObj gradient against WA+density plus our term."""
        leaf=pos.detach().clone().requires_grad_(True)
        extra=self.term(leaf,self.gamma)
        added,=torch.autograd.grad(extra,leaf)
        self.active=False
        try:
            base=self.placer.model.obj_fn(leaf)
            baseline,=torch.autograd.grad(base,leaf)
        finally:
            self.active=True
        total=self.placer.model.obj_fn(leaf)
        combined,=torch.autograd.grad(total,leaf)
        error=float((combined-baseline-added).abs().max())
        tolerance=1e-4*(1+float(added.abs().max()))
        if not error<=tolerance:raise RuntimeError('PlaceObj gradient did not include the paper term correctly')
        return dict(gradient_sum_max_error=error,tolerance=tolerance,
            added_l1=float(added.abs().sum()),base_l1=float(baseline.abs().sum()),
            total_l1=float(combined.abs().sum()))

    def __call__(self,pos):
        if not self.active:return pos[:0].sum()
        self.forward_generations.append(self.term.generation)
        return self.term(pos,self.gamma)

    def _step(self,*args,**kwargs):
        iteration=self.next_iteration
        self.forward_generations=[]
        if iteration>=self.start and (self.enabled or self.match_refresh):
            model=self.placer.model
            pos=self.placer.pos[0]
            if self.enabled and (not self.active or (iteration-self.start)%self.every==0):
                metadata=self.term.rebuild(pos)
                self.rebuilds.append(dict(iteration=iteration,**metadata))
            self.gamma=model.gamma.detach().clone()
            self.active=self.enabled
            if self.active and self.gradient_audit is None:
                self.gradient_audit=self.audit_gradient_sum(pos)
            # Also refresh on gamma/density changes between topology rebuilds.
            # Forward calls all see this one immutable topology/gamma pair.
            refresh_nesterov_secant(self.optimizer)
            self.cache_refreshes+=1
            self.cache_generation=self.term.generation
        generation=self.term.generation
        value=self.original_step(*args,**kwargs)
        if self.term.generation!=generation:raise RuntimeError('topology changed during optimizer step')
        return value

    def callback(self,iteration,pos):
        optimizer=self.placer.optimizer
        if self.optimizer is None:
            self.optimizer=optimizer;self.original_step=optimizer.step
            optimizer.step=self._step
        elif optimizer is not self.optimizer:
            raise ValueError('paper GP controller supports a single optimizer stage')
        self.next_iteration=iteration+1
        gradient_l1=0.;value=0.
        if self.active:
            leaf=pos.detach().clone().requires_grad_(True)
            interior=self.term(leaf,self.gamma)
            gradient,=torch.autograd.grad(interior,leaf)
            if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(interior)):
                raise FloatingPointError('nonfinite paper GP value/gradient')
            gradient_l1=float(gradient.abs().sum());value=float(interior.detach())
        generations=self.forward_generations or [self.term.generation]
        self.trace.append(dict(iteration=int(iteration),active=self.active,
            generation=self.term.generation,cache_generation=self.cache_generation,
            forward_generation_min=min(generations),forward_generation_max=max(generations),
            interior_value=value,interior_gradient_l1=gradient_l1,
            position_sha256=tensor_digest(pos),gamma=float(self.placer.model.gamma),
            hpwl=float(self.placer.op_collections.hpwl_op(pos).detach()),
            pre_step_overflow=float(self.placer.model.overflow.max()),
            density_weight=self.placer.model.density_weight.detach().cpu().tolist()))
