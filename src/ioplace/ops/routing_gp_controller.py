"""Publish frozen route objectives and router observations between GP steps."""
import math

import torch

from ioplace.dp_hook import refresh_nesterov_secant
from ioplace.gr_in_loop import require_gr_in_loop
from ioplace.norm import TermNormalizer
from ioplace.ops.steiner_gp import tensor_digest


class RoutingGPController:
    """Per-instance PlaceObj term and optimizer-boundary callback.

    Positive WA warmup precedes activation. A step uses one immutable route
    generation, observation version, gamma, and routing coefficient. Both
    Nesterov secant endpoints are refreshed before any step with these settings.
    """
    def __init__(self, term, placer, *, mode="joint", start=200, rebuild_every=20,
                 tau=1., route_strength=.1, router_every=0, router_callback=None):
        require_gr_in_loop()
        if mode not in ("wa_standard", "wa", "paper", "joint"):
            raise ValueError("unknown GP mode")
        if start < 1 or rebuild_every < 1 or router_every < 0:
            raise ValueError("positive start/rebuild and nonnegative router interval required")
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(route_strength) or route_strength < 0:
            raise ValueError("finite positive tau and nonnegative route strength required")
        self.term, self.placer, self.mode = term, placer, mode
        self.start, self.every, self.tau = start, rebuild_every, float(tau)
        self.route_strength, self.route_lambda = float(route_strength), 0.
        self.router_every, self.router_callback = router_every, router_callback
        self.active, self.gamma, self.next_iteration = False, None, 1
        self.optimizer, self.original_step = None, None
        self.cache_generation, self.cache_refreshes = 0, 0
        self.trace, self.rebuilds, self.forward_generations = [], [], []
        self.forward_observations = []
        self.gradient_audit = None

    def __call__(self, pos):
        if not self.active:
            return pos[:0].sum()
        self.forward_generations.append(self.term.generation)
        self.forward_observations.append(self.term.published_observation_version)
        value = self.term.paper(pos, self.gamma)
        if self.mode == "joint":
            value = value + self.route_lambda * self.term(pos, tau=self.tau, gamma=self.gamma)
        return value

    def _gradient_stats(self, pos):
        leaf = pos.detach().clone().requires_grad_(True)
        paper = self.term.paper(leaf, self.gamma)
        paper_gradient, = torch.autograd.grad(paper, leaf)
        result = dict(paper_value=float(paper.detach()), paper_gradient_l1=float(paper_gradient.abs().sum()))
        if not torch.isfinite(paper_gradient).all() or not torch.isfinite(paper):
            raise FloatingPointError("nonfinite paper objective/gradient")
        if self.mode == "joint":
            parts = self.term.components(leaf, tau=self.tau, gamma=self.gamma)
            for name in ("io", "wirelength", "congestion", "objective"):
                gradient, = torch.autograd.grad(parts[name], leaf, retain_graph=True)
                if not torch.isfinite(gradient).all() or not torch.isfinite(parts[name]):
                    raise FloatingPointError(f"nonfinite routing {name}/gradient")
                label = "route" if name == "objective" else name
                result[f"{label}_value"] = float(parts[name].detach())
                result[f"{label}_gradient_l1"] = float(gradient.abs().sum())
        return result

    def _calibrate(self, pos):
        stats = self._gradient_stats(pos)
        leaf = pos.detach().clone().requires_grad_(True)
        wa = self.placer.model.op_collections.wirelength_op(leaf)
        gradient, = torch.autograd.grad(wa, leaf)
        wa_l1 = float(gradient.abs().sum())
        if not math.isfinite(wa_l1):
            raise FloatingPointError("nonfinite WA gradient during routing calibration")
        norm = stats["route_gradient_l1"]
        # v2 P-H (design sec 4): the one-shot ratio normalisation now lives in
        # ioplace.norm; this is the third and last of the retired coefficient
        # paths, kept only as an adapter.
        self.route_lambda = TermNormalizer.oneshot_lambda(self.route_strength,
                                                           wa_l1, norm)
        if not math.isfinite(self.route_lambda):
            raise FloatingPointError("nonfinite routing coefficient")
        return dict(**stats, wa_gradient_l1=wa_l1, route_lambda=self.route_lambda,
                    requested_route_strength=self.route_strength,
                    effective_route_strength=self.route_lambda * norm / wa_l1 if wa_l1 > 0 else 0.)

    def audit_gradient_sum(self, pos):
        leaf = pos.detach().clone().requires_grad_(True)
        added, = torch.autograd.grad(self(leaf), leaf)
        self.active = False
        try:
            baseline, = torch.autograd.grad(self.placer.model.obj_fn(leaf), leaf)
        finally:
            self.active = True
        total, = torch.autograd.grad(self.placer.model.obj_fn(leaf), leaf)
        error = float((total - baseline - added).abs().max())
        tolerance = 1e-4 * (1 + float(added.abs().max()))
        if not math.isfinite(error) or error > tolerance:
            raise RuntimeError("PlaceObj gradient did not include the complete routing objective")
        return dict(gradient_sum_max_error=error, tolerance=tolerance,
                    added_l1=float(added.abs().sum()), base_l1=float(baseline.abs().sum()),
                    total_l1=float(total.abs().sum()))

    def _step(self, *args, **kwargs):
        iteration = self.next_iteration
        self.forward_generations, self.forward_observations = [], []
        if iteration >= self.start and self.mode != "wa_standard":
            pos = self.placer.pos[0]
            self.gamma = self.placer.model.gamma.detach().clone()
            enabled = self.mode in ("paper", "joint")
            pending = self.term.observation_version != self.term.published_observation_version
            if enabled and (not self.active or (iteration - self.start) % self.every == 0 or pending):
                metadata = self.term.rebuild(pos)
                if self.mode == "joint":
                    metadata["gradient_calibration"] = self._calibrate(pos)
                self.rebuilds.append(dict(iteration=iteration, **metadata))
            self.active = enabled
            if self.active and self.gradient_audit is None:
                self.gradient_audit = self.audit_gradient_sum(pos)
            refresh_nesterov_secant(self.optimizer)
            self.cache_refreshes += 1
            self.cache_generation = self.term.generation
        generation = self.term.generation
        observation = self.term.published_observation_version
        result = self.original_step(*args, **kwargs)
        if self.term.generation != generation or self.term.published_observation_version != observation:
            raise RuntimeError("routing objective changed during an optimizer step")
        return result

    def callback(self, iteration, pos):
        optimizer = self.placer.optimizer
        if self.optimizer is None:
            self.optimizer, self.original_step = optimizer, optimizer.step
            optimizer.step = self._step
        elif optimizer is not self.optimizer:
            raise ValueError("routing GP requires a single optimizer stage")
        self.next_iteration = iteration + 1
        stats = self._gradient_stats(pos) if self.active else {}
        generations = self.forward_generations or [self.term.generation]
        observations = self.forward_observations or [self.term.published_observation_version]
        self.trace.append(dict(iteration=int(iteration), active=self.active,
            generation=self.term.generation, cache_generation=self.cache_generation,
            observation_version=self.term.observation_version,
            published_observation_version=self.term.published_observation_version,
            forward_generation_min=min(generations), forward_generation_max=max(generations),
            forward_observation_min=min(observations), forward_observation_max=max(observations),
            route_lambda=self.route_lambda, tau=self.tau, **stats,
            position_sha256=tensor_digest(pos), gamma=float(self.placer.model.gamma),
            hpwl=float(self.placer.op_collections.hpwl_op(pos).detach()),
            pre_step_overflow=float(self.placer.model.overflow.max()),
            density_weight=self.placer.model.density_weight.detach().cpu().tolist()))
        if self.mode == "joint" and self.router_callback and self.router_every and (iteration + 1) % self.router_every == 0:
            self.router_callback(iteration, pos)
