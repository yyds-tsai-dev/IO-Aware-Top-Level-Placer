"""The single seam between the v2 main flow and term normalisation.

`run_main_flow.py` must never import `schedules.py` or `norm.py` directly:
subproject P-H replaced the three ad-hoc normalisation paths with
`src/ioplace/norm.py`'s `TermNormalizer` (design v2 sec 4), and this module is
what lets one driver drive either path with a one-word `--norm-policy` change.

Two adapters, one protocol (the plan's Task 5 table):

* `LegacyNormAdapter` -- `schedules.ScheduleState` plus
  `ops/ft_callback.publish_atomic`: the recorded pre-v2 coefficients,
  bit-for-bit. Writes `legacy_trace.jsonl`.
* `TermNormalizerAdapter` -- policies `grandplan` (design sec 4 policy A) and
  `adaptive` (policy B) over the live `norm.TermNormalizer`. A `ScheduleState`
  survives here too, but *only* as the tau / rho / activation schedule:
  `TermNormalizer` owns no temperature and no activation trigger, so something
  still has to map overflow onto tau. Every coefficient comes from the
  normalizer. Writes `norm_trace.jsonl` through `norm_trace.NormTraceWriter`.

`TermNormalizerAdapter` mirrors, call for call, the wiring P-H Task 7 landed in
`src/ioplace/drivers/run_placement_io.py` (`_register_norm_terms:83-116`, the
per-iteration `set_activation`/`update_activation` pair at `:570-577`, the
`should_probe`/`transaction` pair at `:657-659`, `term_fn`'s
`applied_lambda` reads at `:450-456`, and the
`refresh_nesterov_secant`/`mark_refreshed` gate at `:823-828`). Anything that
differs between the two drivers is a bug in this module.
"""
import json
import math
import os

from ioplace.norm import TermNormalizer, VersionPair, parse_target_shares
from ioplace.norm_trace import NormTraceWriter
from ioplace.ops.ft_callback import publish_atomic
from ioplace.ops.norm_terms import FtNormTerm, IoNormTerm
from ioplace.schedules import ScheduleState

#: Bound to P-H's tuple rather than re-spelled, so the two cannot drift.
NORM_POLICIES = TermNormalizer.POLICIES

LEGACY_TRACE_NAME = "legacy_trace.jsonl"
NORM_TRACE_NAME = "norm_trace.jsonl"

#: Policy B's IO share when `--norm-target-share` is silent. Same value and
#: same meaning as `run_placement_io.DEFAULT_IO_TARGET_SHARE` (:74-76) and
#: docs/dev-env.md's P-H flag table, so the two drivers' `adaptive` arms are
#: comparable.
DEFAULT_IO_TARGET_SHARE = 0.3

#: FT's own activation gate, the ledger ruling of 2026-09-19 that
#: `run_placement_io.FT_ACTIVATE_OVERFLOW` (:77-80) records: the legacy
#: tau_rel window 0.12 -> 0.05 maps through `tau_rel_from_overflow` to overflow
#: 0.57 -> 0.25, and 0.30 is the threshold the capacity term uses (design sec
#: 5). Duplicated rather than imported: importing the legacy driver would pull
#: DREAMPlace, numpy, scipy and the GPU evaluator into every consumer of this
#: module. `tests/test_norm_adapter.py` locks the two constants together.
FT_ACTIVATE_OVERFLOW = 0.30

#: Keys `_ScheduleBacked` -- and therefore both adapters -- understands.
#: `eps_rel` is still a `ScheduleState` field (schedules.py:128) and still
#: governs the legacy arm's `derive_kappa_ft`; the P-H fix wave removed it from
#: `TermNormalizer` entirely (controller ruling F1'), so it is deliberately not
#: forwarded to the normalizer below.
_SCHEDULE_KEYS = ("L_R", "rho_max", "tau_hi", "tau_lo", "of_on", "of_end",
                  "of_full", "f_ft_max", "ft_ramp_mode", "tau_start",
                  "tau_full", "ema", "c_lip", "n_ramp", "kappa_max", "eps_rel",
                  "out_dir")

#: Keys only `TermNormalizerAdapter` understands. The driver builds one kwarg
#: set for every policy, so under `legacy` these are accepted and dropped
#: rather than raising -- `--norm-policy` must stay a one-word change.
_NORMALIZER_KEYS = ("io_term", "ft_term", "ecc_max", "num_movable",
                    "num_nodes", "norm_p", "probe_every", "wt0", "wt_step",
                    "ramp_period", "wt_max", "momentum", "target_shares",
                    "track_cancellation")


class _ScheduleBacked(object):
    """Shared tau / rho / activation plumbing and trace-file placement.

    Both adapters keep a `ScheduleState`: it is the only thing in the tree that
    maps overflow onto the soft-assign temperature
    (`schedules.tau_rel_from_overflow`) and latches activation, and
    `TermNormalizer` deliberately owns neither.
    """

    TRACE_NAME = None

    def __init__(self, *, L_R, rho_max, tau_hi, tau_lo, of_on, of_end, of_full,
                 f_ft_max, ft_ramp_mode, tau_start, tau_full, ema=0.5,
                 c_lip=1.0, n_ramp=20, kappa_max=100.0, eps_rel=1e-3,
                 out_dir=None):
        if not L_R > 0:
            raise ValueError("L_R must be positive")
        self.L_R = float(L_R)
        self.state = ScheduleState(
            rho_max=rho_max, tau_hi=tau_hi, tau_lo=tau_lo, of_on=of_on,
            of_end=of_end, of_full=of_full, n_ramp=n_ramp, c_lip=c_lip,
            f_ft_max=f_ft_max, ft_ramp_mode=ft_ramp_mode, tau_start=tau_start,
            tau_full=tau_full, kappa_max=kappa_max, eps_rel=eps_rel, ema=ema)
        self.out_dir = out_dir
        if out_dir is None:
            self.trace_path = None
        else:
            os.makedirs(out_dir, exist_ok=True)
            self.trace_path = os.path.join(out_dir, self.TRACE_NAME)

    @property
    def active(self):
        return bool(self.state.active)

    @property
    def tau(self):
        return float(self.state.tau)

    @property
    def tau_rel(self):
        return float(self.state.tau) / self.L_R


class LegacyNormAdapter(_ScheduleBacked):
    """The pre-v2 path: `ScheduleState` plus the seven-step atomic FT
    transaction (schedules.py:178-237, ops/ft_callback.py:18-49). Always uses
    the atomic discipline -- with `ft_term=None` `publish_atomic` degenerates
    to the IO-only ratio update while still bumping `obj_version` exactly
    once, which is what the Nesterov invariant
    (dp_hook.install_version_invariant) requires."""

    policy = "legacy"
    TRACE_NAME = LEGACY_TRACE_NAME

    def __init__(self, **config):
        super().__init__(**config)
        self._trace = None if self.trace_path is None else open(self.trace_path, "w")

    @property
    def lambda_io(self):
        return float(self.state.lambda_io)

    @property
    def kappa_ft(self):
        return float(self.state.kappa_ft)

    @property
    def obj_version(self):
        return int(self.state.obj_version)

    @property
    def refreshed_version(self):
        # dp_hook.install_version_invariant reads obj_version and
        # refreshed_version off whatever object it is handed, so the adapter
        # exposes both and the driver never reaches through to `.state`.
        return int(self.state.refreshed_version)

    def begin_iteration(self, iteration, overflow, gamma):
        return bool(self.state.update_continuous(iteration, overflow, self.L_R, gamma))

    def probe(self, iteration, pos, *, io_term, ft_term, wirelength_op,
              ecc_max, gamma):
        row = publish_atomic(self.state, io_term, ft_term, wirelength_op, pos,
                             iteration, self.tau_rel, ecc_max, gamma)
        row.update(policy=self.policy, iteration=int(iteration), tau=self.tau,
                   tau_rel=self.tau_rel,
                   refreshed_version=int(self.state.refreshed_version))
        return row

    def write_trace_row(self, row, extra):
        """One `legacy_trace.jsonl` line per probe: `publish_atomic`'s keys plus
        whatever the driver measured in the same callback. This file is
        deliberately *not* `norm_trace.jsonl` -- that name belongs to design
        sec 4's schema, which `norm_trace.NormTraceWriter` validates and this
        row does not satisfy (pre-flight amendment A-9)."""
        record = dict(row)
        record.update(extra)
        if self._trace is not None:
            self._trace.write(json.dumps(record) + "\n")
            self._trace.flush()
        return record

    def close(self):
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def needs_refresh(self):
        return bool(self.state.needs_refresh())

    def mark_refreshed(self):
        self.state.mark_refreshed()


class TermNormalizerAdapter(_ScheduleBacked):
    """Policies `grandplan` (A) and `adaptive` (B) over `norm.TermNormalizer`.

    `ScheduleState` supplies tau, rho and the single activation instant for
    `io`; the normalizer supplies every coefficient, the Lipschitz cap, the
    cancellation ratio and `norm_trace.jsonl`. `VersionPair` presents both
    version counters as one, so a single
    `dp_hook.install_version_invariant(optimizer, adapter)` covers an
    activation bump from `update_continuous` *and* a coefficient bump from
    `transaction()` -- stacking two invariant wrappers would not work, because
    `refresh_nesterov_secant` unwraps exactly one `__wrapped__` level.

    Terms are registered in the constructor, not on the first probe: every
    registered term's activation gate must be evaluated on *every* GP
    iteration (`update_activation`), and a term registered at the first
    `every`-gated probe would latch up to `every` iterations late -- which is
    review I1a's defect, with `n_ramp=20` aliased away by `every=50`.
    """

    TRACE_NAME = NORM_TRACE_NAME

    def __init__(self, policy, *, io_term=None, ft_term=None, ecc_max=0.0,
                 num_movable=None, num_nodes=None, norm_p=1, probe_every=50,
                 wt0=0.05, wt_step=0.05, ramp_period=100, wt_max=1.0,
                 momentum=0.75, target_shares=None, track_cancellation=True,
                 **config):
        if policy not in ("grandplan", "adaptive"):
            raise ValueError(
                "TermNormalizerAdapter serves 'grandplan' and 'adaptive'; "
                "'legacy' is LegacyNormAdapter's (got %r)" % (policy,))
        if io_term is None:
            raise ValueError(
                "TermNormalizerAdapter needs io_term (and ft_term/ecc_max when "
                "FT is on) at construction: terms are registered before the GP "
                "loop so update_activation() sees every gate on every iteration")
        super().__init__(**config)
        self.policy = policy
        # Same validation the landed driver applies in `run_io`
        # (run_placement_io.py:257-265): with a non-legacy policy every
        # coefficient comes from the normalizer, so `--rho-max 0` -- the
        # reweight-only "IO off" setting under legacy -- would silently turn
        # the IO penalty on at full normalizer strength. The main flow has no
        # `--rho-margin` and no `--callback-order`, so the other two
        # combination rules the driver checks cannot arise here.
        if not float(self.state.rho_max) > 0.0:
            raise ValueError(
                "--rho-max 0 means 'IO off' only under --norm-policy legacy; "
                "policy %r derives lambda_io from gradient norms and would run "
                "the IO penalty at full strength" % (policy,))
        self._io_term, self._ft_term = io_term, ft_term
        if num_movable is None:
            num_movable = io_term.num_movable
        if num_nodes is None:
            num_nodes = io_term.num_nodes
        self._trace = (None if self.trace_path is None
                       else NormTraceWriter(self.trace_path))
        # No `eps_rel`: the P-H fix wave (commit 5e07cdf, controller ruling
        # F1') removed the relative deadness threshold from the non-legacy
        # coefficient path, and `TermNormalizer.__init__` no longer accepts the
        # keyword. `kappa_max` is forwarded so the recovered
        # kappa = lambda_ft/lambda_io is bounded by the same constant the
        # legacy arm's `derive_kappa_ft` uses (review M2).
        self.normalizer = TermNormalizer(
            policy=policy, norm_p=norm_p, ema=self.state.ema,
            probe_every=probe_every, wt0=wt0, wt_step=wt_step,
            ramp_period=ramp_period, wt_max=wt_max, momentum=momentum,
            c_lip=self.state.c_lip, kappa_max=self.state.kappa_max,
            num_movable=int(num_movable), num_nodes=int(num_nodes),
            track_cancellation=track_cancellation, trace=self._trace)
        self._register(ft_term, ecc_max, parse_target_shares(target_shares))
        self._versions = VersionPair(self.state, self.normalizer)
        self._iteration = -1
        self._overflow = float("nan")

    def _register(self, ft_term, ecc_max, overrides):
        """`run_placement_io._register_norm_terms` (:83-116), term for term.

        Curvatures are design sec 4's declared values: 1 for IO, `ecc_max` for
        FT, floored at 1 because `register` rejects a curvature below 1 (the
        curvature of a term with no eccentricity spread).

        Three load-bearing details, all from the landed driver:

        * `ft` declares `requires="io"`. `FtTerm` can only express the FT force
          as `lambda_io * kappa`, so a `lambda_ft > 0` with `lambda_io == 0` is
          unapplicable; the normalizer publishes 0 for `ft` instead of the
          driver aborting a multi-hour run on one transient probe (review C1).
        * `ft`'s activation gate is `FT_ACTIVATE_OVERFLOW`, not `of_on`: FT
          must not switch on with IO.
        * Under `grandplan`, `ft`'s weight ceiling is `f_ft_max * wt_max`, so
          the FT force share mirrors the legacy `f_ft_max` instead of
          converging to IO's ceiling (review I4). Under `adaptive` the target
          shares already say what each term's share is, so no override is
          installed.
        """
        self.normalizer.register(
            "io", IoNormTerm(self._io_term), 1.0,
            target_share=overrides.get("io", DEFAULT_IO_TARGET_SHARE),
            activate_overflow=self.state.of_on, n_ramp=self.state.n_ramp)
        if ft_term is not None:
            ceiling = float(self.state.f_ft_max) * float(self.normalizer.wt_max)
            self.normalizer.register(
                "ft", FtNormTerm(ft_term), max(float(ecc_max), 1.0),
                target_share=overrides.get("ft", float(self.state.f_ft_max)),
                activate_overflow=FT_ACTIVATE_OVERFLOW,
                n_ramp=self.state.n_ramp, requires="io",
                wt_max=(ceiling if self.policy == "grandplan" and ceiling > 0.0
                        else None))
        unknown = sorted(set(overrides) - set(self.normalizer.configs))
        if unknown:
            raise ValueError("target shares name unregistered terms %r "
                             "(registered: %r)"
                             % (unknown, sorted(self.normalizer.configs)))
        #: Effective share per *registered* term, for the record and the tests.
        self.target_shares = dict(
            (name, config.target_share)
            for name, config in self.normalizer.configs.items())

    @property
    def lambda_io(self):
        """The coefficient `term_fn` must apply *at this iteration*: the
        committed lambda times the activation ramp (`applied_lambda`, review
        I1b). `self.normalizer.lambdas` only moves on an `every`-gated
        transaction, so reading it directly would step the coefficient once
        instead of drifting it through the ramp window, exactly the aliasing
        the legacy arm never had."""
        return float(self.normalizer.applied_lambda("io", self._iteration))

    @property
    def kappa_ft(self):
        """`lambda_ft / lambda_io`, both *applied* -- the ratio
        `ops/ft_term.FtTerm.forward` wants, since it scales the FT part by
        `lambda_io * kappa_ft`, so the product is exactly FT's applied
        coefficient. 0.0 when the applied `lambda_io` is 0: there is nothing to
        divide by, and `term_fn` is gated on `lambda_io != 0` anyway. The ratio
        stays inside `kappa_max`: `_compute` bounds the committed ratio, and
        FT's gate (0.30) is below IO's (`of_on`), so FT's ramp is never ahead
        of IO's."""
        lambda_io = self.normalizer.applied_lambda("io", self._iteration)
        if lambda_io <= 0.0:
            return 0.0
        return float(self.normalizer.applied_lambda("ft", self._iteration)) / lambda_io

    @property
    def obj_version(self):
        return int(self._versions.obj_version)

    @property
    def refreshed_version(self):
        return int(self._versions.refreshed_version)

    def begin_iteration(self, iteration, overflow, gamma):
        """`run_placement_io.cb`'s first four statements (:565-577), in order:
        record the iteration `applied_lambda` is answered against, run the
        continuous schedule, hand `io` the schedule's own activation instant,
        and latch every other term's overflow gate."""
        self._iteration = int(iteration)
        self._overflow = float(overflow)
        discrete = bool(self.state.update_continuous(iteration, overflow,
                                                     self.L_R, gamma))
        # Review I1a: `set_activation` is monotone and idempotent and makes
        # `io`'s clock external, so `update_activation`'s own overflow gate
        # skips it; the remaining terms (`ft`) latch on their own threshold,
        # every iteration rather than every `probe_every`.
        self.normalizer.set_activation("io", self.state.it_activate)
        self.normalizer.update_activation(self._iteration, self._overflow)
        return discrete

    def probe(self, iteration, pos, *, io_term, ft_term, wirelength_op,
              ecc_max, gamma):
        """One WL backward plus one isolated backward per term on the probe
        cadence, then one atomic coefficient transaction -- the landed driver's
        `should_probe`/`transaction` pair (run_placement_io.py:657-659). A
        transaction on a callback that did not probe reuses the previous
        probe's norms, which is why `norm_trace.jsonl` has one row per
        transaction rather than one per probe.

        `io_term`/`ft_term`/`ecc_max` are the protocol's, i.e.
        `LegacyNormAdapter`'s, arguments; this adapter registered its terms at
        construction and only checks that the driver is still handing over the
        same objects. Returns the pending `norm_trace.jsonl` row;
        `mark_refreshed()` is what actually writes it.

        A non-finite coefficient raises `FloatingPointError` out of
        `transaction()` before anything is committed (review I6). The driver
        does not catch it: a `nan` lambda silently passes every downstream
        `lam <= 0` guard and poisons the whole objective."""
        if io_term is not self._io_term or ft_term is not self._ft_term:
            raise ValueError(
                "probe() was handed different term objects than the ones "
                "registered at construction")
        if not math.isfinite(self._overflow):
            raise RuntimeError("begin_iteration() must run before probe()")
        ctx = {"iteration": int(iteration), "overflow": self._overflow,
               "tau": self.tau, "gamma": float(gamma)}
        if self.normalizer.should_probe(iteration):
            self.normalizer.probe(iteration, pos, wirelength_op, ctx)
        transaction = self.normalizer.transaction(iteration, self._overflow,
                                                  self.tau, gamma)
        return transaction.row

    def write_trace_row(self, row, extra):
        """No-op. `norm_trace.jsonl` is written by the normalizer's own
        `NormTraceWriter` at `mark_refreshed()` time, and that writer rejects
        any row whose keys are not exactly `norm_trace.ROW_FIELDS` (pre-flight
        amendment A-9: one schema per file name). The driver's per-probe
        extras -- `io_count`, `ft_count`, `churn`, `lambda_io`, `kappa_ft` --
        reach `result.json` through `soft_summary["probe_samples"]` instead."""
        return None

    def close(self):
        if self._trace is not None:
            self._trace.close()
            self._trace = None
            self.normalizer.trace = None

    def needs_refresh(self):
        return self.obj_version != self.refreshed_version

    def mark_refreshed(self):
        self.state.mark_refreshed()
        self.normalizer.mark_refreshed()


def make_norm_adapter(policy, **config):
    if policy not in NORM_POLICIES:
        raise ValueError(f"unknown norm policy {policy!r}; expected one of {NORM_POLICIES}")
    unknown = sorted(set(config) - set(_SCHEDULE_KEYS) - set(_NORMALIZER_KEYS))
    if unknown:
        raise TypeError(f"make_norm_adapter got unexpected keyword(s): {unknown}")
    schedule = {key: value for key, value in config.items() if key in _SCHEDULE_KEYS}
    if policy == "legacy":
        # `publish_atomic` measures L1 gradients internally, so an L2 request
        # under legacy would be silently ignored -- the landed driver rejects
        # the same pair (run_placement_io.py:237-239).
        if int(config.get("norm_p", 1)) != 1:
            raise ValueError("norm policy 'legacy' measures L1 gradients inside "
                             "publish_atomic; norm_p must be 1, got %r"
                             % (config.get("norm_p"),))
        return LegacyNormAdapter(**schedule)
    extra = {key: value for key, value in config.items() if key in _NORMALIZER_KEYS}
    return TermNormalizerAdapter(policy, **schedule, **extra)
