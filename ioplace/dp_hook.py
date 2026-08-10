"""DREAMPlace integration helpers for the M2 differentiable IO term.

design v2 sec 6.2 (params-borne per-instance patch) and sec 6.4 (Nesterov secant
invalidation). No global mutation of DREAMPlace classes happens here.
"""

def attach_terms(params, terms):
    """Attach objective terms to a *run-specific* Params object. The patched
    PlaceObj.__init__ copies this list into every PlaceObj it builds."""
    params._extra_obj_terms = list(terms)


def detach_terms(params):
    if hasattr(params, "_extra_obj_terms"):
        delattr(params, "_extra_obj_terms")


def assert_optimizer_lock(params):
    """design v2 sec 3.2.4: the surrogate-only forward is only behaviourally
    equivalent under nesterov + use_bb=0 + Lsub_iteration=1. Must run after
    placedb.initialize(params): use_bb defaults to the string 'auto' and is
    only resolved to 0/1 there (PlaceDB.py:837)."""
    stage = params.global_place_stages[0]
    opt = str(stage.get("optimizer", "")).lower()
    assert opt == "nesterov", f"optimizer lock: expected nesterov, got {opt!r}"
    use_bb = getattr(params, "use_bb", 0)
    assert not isinstance(use_bb, str), (
        f"optimizer lock: use_bb still unresolved ({use_bb!r}); "
        "call assert_optimizer_lock after placedb.initialize(params)")
    assert int(use_bb) == 0, \
        f"optimizer lock: expected use_bb==0 (step_nobb), got {params.use_bb!r}"
    lsub = int(stage.get("Lsub_iteration", 1))
    assert lsub == 1, f"optimizer lock: expected Lsub_iteration==1, got {lsub}"


def refresh_nesterov_secant(optimizer):
    """在 objective 離散變更後,讓 Nesterov 的梯度快取在新 objective 下重算。
    前提(由 T5 assert):optimizer 為 NesterovAcceleratedGradientOptimizer、
    use_bb == 0(走 step_nobb)、單一 param group 且單一 param。"""
    g = optimizer.param_groups[0]
    if not g["g_k"]:
        return
    # The refresh IS the mechanism that resolves a version mismatch, so its own
    # evaluations must not be vetted by install_version_invariant's wrapper
    # (they run before mark_refreshed() by the mandated call order); unwrap to
    # the original obj_and_grad_fn if the invariant is installed.
    f = optimizer.obj_and_grad_fn
    f = getattr(f, "__wrapped__", f)
    obj_k, grad_k = f(g["v_k"][0])
    g["g_k"][0].copy_(grad_k.data)
    g["obj_k"][0].copy_(obj_k.data)
    if g["g_k_1"]:
        obj_k1, grad_k1 = f(g["v_k_1"][0])
        g["g_k_1"][0].copy_(grad_k1.data)
        g["obj_k_1"][0].copy_(obj_k1.data)
        dv = (g["v_k"][0].data - g["v_k_1"][0].data).norm(p=2)
        dg = (g["g_k"][0] - g["g_k_1"][0]).norm(p=2)
        if dv > 0 and dg > 0:
            g["alpha_k"][0].copy_(dv / dg)


def install_version_invariant(optimizer, state):
    """Assert that no *optimizer-step* gradient evaluation happens while the
    objective has changed but the Nesterov cache has not been refreshed
    (design v2 sec 6.4.2 (3)). The wrapper carries __wrapped__ so that
    refresh_nesterov_secant -- the sanctioned resolver of exactly that state --
    can evaluate through the original fn without asserting against itself."""
    orig = optimizer.obj_and_grad_fn

    def wrapped(p):
        assert state.obj_version == state.refreshed_version, (
            f"objective version {state.obj_version} evaluated while cache is at "
            f"{state.refreshed_version}; refresh_nesterov_secant() was not called")
        return orig(p)

    wrapped.__wrapped__ = orig
    optimizer.obj_and_grad_fn = wrapped
    return lambda: setattr(optimizer, "obj_and_grad_fn", orig)
