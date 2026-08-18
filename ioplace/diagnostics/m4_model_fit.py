"""M4 T0b model-discipline aggregator (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 7.1 T0b
row): reads `run_t0b_matrix.py`'s per-point JSONs (`results/m4/scaling/
t0b/*.json`) plus any additional case-level run JSONs the caller supplies
(e.g. `probe_gp_memory.py`'s `results/m4/probes/probe_gp_memory__*.json`
five case-level points -- design point 3 in the T0b task row), fits the two
models the design draft's Codex-#1/D4 findings called for, and writes
`results/m4/scaling/model_fit.json`.

Two models, both OLS with an intercept:

  1. **GP peak memory**: `gp_peak_bytes ~ b0 + a*N_total + b*N_pins +
     c*n_bins`. This is the v1 model (`87*N_total + 73*N_pins +
     160*n_bins`, sec 1.1) re-fit on a design where the three predictors
     can actually move independently -- `run_t0b_matrix.py`'s bins x
     target_density sweep moves n_bins and N_total (via the filler count
     target_density drives) at fixed N_pins, and its net-drop points move
     N_pins at fixed N_total/n_bins. v1 had no intercept and 3 coefficients
     from 5 collinear points (2 residual DoF, Codex finding #1); this
     module always includes an intercept and reports whether the resulting
     fit clears the identifiability gate below.
  2. **GP runtime**: `t_gp_s ~ T_fixed + n_iter*(a*N_total + b*N_pins +
     c*n_bins*log(n_bins))` (sec 1.1: "T_gp = n_iter x (a.N_total +
     b.N_pins + c.n_bins.log n_bins) + T_fixed"). Fit as one OLS of the
     *total* GP wall time against the three n_iter-scaled predictors plus
     an intercept -- the intercept is exactly `T_fixed` (a one-time,
     not-per-iteration cost) and `a/b/c` are the per-iteration model's own
     coefficients, both recovered from a single regression rather than two
     separate ad hoc steps.

**Hard, pre-registered rejection threshold (design draft sec 7.1 T0b
acceptance, verbatim)**: a model is `identifiable=True` only if BOTH (a)
the condition number of its *standardized* predictor matrix `kappa(X_std)
<= 30`, AND (b) every coefficient's (including the intercept's) 95%
confidence-interval relative half-width `<= 25%`. Either model failing
either leg sets that model's `identifiable=False`, and the top-level
`identifiable` field (the AND of both models) governs sec 2.1/sec 5.1's
"do not use this model to extrapolate anything" rule -- this module never
softens or works around a rejection; it only reports it.

CLI:
    PYTHONPATH=. $PY -m ioplace.diagnostics.m4_model_fit \\
        --t0b-dir results/m4/scaling/t0b \\
        --case-json results/m4/probes/probe_gp_memory__adaptec1.json \\
        --case-json results/m4/probes/probe_gp_memory__bigblue4.json \\
        --out results/m4/scaling/model_fit.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"

T0B_DIR_DEFAULT = os.path.join(REPO, "results", "m4", "scaling", "t0b")
OUT_PATH_DEFAULT = os.path.join(REPO, "results", "m4", "scaling", "model_fit.json")

# Design draft sec 7.1 T0b acceptance -- hard-coded, not a CLI knob (a
# rejection threshold that could be raised until a fit passes would defeat
# its own purpose).
KAPPA_THRESHOLD = 30.0
CI_REL_HALF_WIDTH_THRESHOLD = 0.25

UNITS = dict(
    n_total="count (movable + filler nodes)", n_pins="count",
    n_bins="count (num_bins_x * num_bins_y)",
    gp_peak_bytes="bytes", t_gp_s="seconds", n_iter="count (GP iterations run)",
)

# Files in a t0b-dir glob that are plans/scratch artifacts, not case-level
# run records -- excluded from aggregation. `_scratch_configs/` is a
# subdirectory so a flat "*.json" glob never descends into it anyway; listed
# here only for documentation.
_T0B_DIR_NON_RUN_FILES = {"dry_run_plan.json"}


def _normalize_record(rec, path):
    """Maps either of this repo's two GP-memory-bearing probe schemas onto
    the common (case, n_total, n_pins, n_bins, gp_peak_bytes, t_gp_s,
    n_iter) tuple this module fits against. Returns None for a run that
    isn't usable (ok=False, e.g. an OOM) rather than raising -- a failed
    run is legitimate information (design draft sec 2.2's three-state
    rule) but is not a fitting point.

    Schema A -- `ioplace/diagnostics/probes_m4/probe_gp_memory.py`
    (case-level points, design point 3): `num_physical`/`num_filler`/
    `num_pins`/`num_bins_x`/`num_bins_y`/`gp_peak_alloc_mb`/`gp_s`/
    `gp_iterations`.

    Schema B -- `ioplace/diagnostics/probes_m4/run_t0b_matrix.py` (design
    points 1/2, built on `ioplace/profile.py`'s schema): `n_physical`/
    `n_filler`/`n_total`/`n_pins`/`n_bins`/`gp_peak_alloc_gb`/`t_gp`/
    `gp_iterations_run`/`tag`.
    """
    if not rec.get("ok", True):
        return None
    if "num_physical" in rec:  # schema A
        n_total = rec["num_physical"] + rec.get("num_filler", 0)
        case = os.path.splitext(os.path.basename(rec.get("config") or path))[0]
        return dict(
            case=case, source_path=path,
            n_total=float(n_total), n_pins=float(rec["num_pins"]),
            n_bins=float(rec["num_bins_x"] * rec["num_bins_y"]),
            gp_peak_bytes=float(rec["gp_peak_alloc_mb"]) * 2**20,
            t_gp_s=float(rec["gp_s"]), n_iter=rec.get("gp_iterations"),
        )
    if "n_physical" in rec:  # schema B
        n_total = rec.get("n_total", rec["n_physical"] + (rec.get("n_filler") or 0))
        case = rec.get("tag") or os.path.splitext(os.path.basename(path))[0]
        return dict(
            case=case, source_path=path,
            n_total=float(n_total), n_pins=float(rec["n_pins"]),
            n_bins=float(rec["n_bins"]), gp_peak_bytes=float(rec["gp_peak_alloc_gb"]) * 2**30,
            t_gp_s=float(rec["t_gp"]), n_iter=rec.get("gp_iterations_run"),
        )
    raise ValueError(
        f"{path}: unrecognized case-level run schema -- neither probe_gp_memory's "
        "'num_physical' nor run_t0b_matrix's 'n_physical' key is present")


def load_points(t0b_dir=T0B_DIR_DEFAULT, case_json_paths=()):
    """Reads every `*.json` directly under `t0b_dir` (design points 1/2) and
    every path in `case_json_paths` (design point 3), normalizes them, and
    returns (points, skipped) where `skipped` lists (path, reason) for
    anything dropped (not ok=True, or unparseable)."""
    paths = sorted(
        p for p in glob.glob(os.path.join(t0b_dir, "*.json"))
        if os.path.basename(p) not in _T0B_DIR_NON_RUN_FILES
    )
    paths += list(case_json_paths)

    points, skipped = [], []
    for path in paths:
        with open(path) as f:
            rec = json.load(f)
        norm = _normalize_record(rec, path)
        if norm is None:
            skipped.append((path, f"ok={rec.get('ok')}"))
            continue
        points.append(norm)
    return points, skipped


def _standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe, mean, std_safe


def fit_ols_with_gate(X, y, predictor_names, cases, alpha=0.05):
    """OLS of `y` on `X` (n x p, no intercept column) with an intercept,
    95% coefficient CIs (Student-t, n-p-1 dof), the condition number of the
    *standardized* `X` (sec 7.1's `kappa(X_std)`), and leave-one-case-out
    residuals. Applies the hard rejection gate (module docstring) and
    returns `identifiable`/`rejection_reasons` alongside the fit.

    Raises ValueError if there are not strictly more points than
    parameters (p + 1 intercept) -- an OLS with dof <= 0 has no residual
    variance to report a CI from, so there is nothing this function could
    honestly call a fit."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    dof = n - (p + 1)
    if dof <= 0:
        raise ValueError(
            f"{n} points is not enough to fit {p + 1} parameters (intercept + "
            f"{p} predictors) with any residual degrees of freedom (dof={dof})")

    X_std, _, _ = _standardize(X)
    kappa = float(np.linalg.cond(X_std))

    X_design = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    resid = y - X_design @ beta
    sigma2 = float(np.sum(resid**2) / dof)
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0.0, None))
    tcrit = float(stats.t.ppf(1 - alpha / 2, dof))
    ci_half_width = tcrit * se

    names = ["intercept"] + list(predictor_names)
    coefficients = {}
    for name, b, hw in zip(names, beta, ci_half_width):
        rel_hw = float(hw / abs(b)) if b != 0 else float("inf")
        coefficients[name] = dict(
            value=float(b), ci95_half_width=float(hw), ci95_rel_half_width=rel_hw,
            ci95_lo=float(b - hw), ci95_hi=float(b + hw),
        )

    loo_residuals = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if mask.sum() <= p:  # refit itself would have <= 0 dof
            loo_residuals.append(dict(case=cases[i], residual=None,
                                       relative_residual=None,
                                       note="too few points to leave this one out"))
            continue
        beta_i, _, _, _ = np.linalg.lstsq(X_design[mask], y[mask], rcond=None)
        y_pred_i = float(X_design[i] @ beta_i)
        resid_i = float(y[i] - y_pred_i)
        loo_residuals.append(dict(
            case=cases[i], predicted=y_pred_i, actual=float(y[i]), residual=resid_i,
            relative_residual=(resid_i / float(y[i])) if y[i] else None,
        ))

    max_rel_hw = max(c["ci95_rel_half_width"] for c in coefficients.values())
    reasons = []
    if kappa > KAPPA_THRESHOLD:
        reasons.append(f"kappa(X_std)={kappa:.3f} > threshold {KAPPA_THRESHOLD}")
    if max_rel_hw > CI_REL_HALF_WIDTH_THRESHOLD:
        worst = max(coefficients, key=lambda k: coefficients[k]["ci95_rel_half_width"])
        reasons.append(
            f"coefficient {worst!r} 95% CI relative half-width "
            f"{coefficients[worst]['ci95_rel_half_width']:.3f} > threshold "
            f"{CI_REL_HALF_WIDTH_THRESHOLD}")
    identifiable = not reasons

    return dict(
        n_points=n, dof=dof, kappa_std=kappa,
        kappa_threshold=KAPPA_THRESHOLD, ci_rel_half_width_threshold=CI_REL_HALF_WIDTH_THRESHOLD,
        coefficients=coefficients, max_ci95_rel_half_width=max_rel_hw,
        identifiable=identifiable, rejection_reasons=reasons,
        loo_residuals=loo_residuals,
    )


def fit_gp_memory_model(points):
    """Design draft sec 1.1's GP peak memory model, predictors [N_total,
    N_pins, n_bins]."""
    cases = [pt["case"] for pt in points]
    X = np.array([[pt["n_total"], pt["n_pins"], pt["n_bins"]] for pt in points])
    y = np.array([pt["gp_peak_bytes"] for pt in points])
    fit = fit_ols_with_gate(X, y, ["N_total", "N_pins", "n_bins"], cases)
    fit["response"] = "gp_peak_bytes"
    fit["response_unit"] = UNITS["gp_peak_bytes"]
    fit["predictor_units"] = {k: UNITS[k] for k in ("n_total", "n_pins", "n_bins")}
    return fit


def fit_gp_runtime_model(points):
    """Design draft sec 1.1's GP runtime model: `t_gp_s ~ T_fixed +
    n_iter*(a*N_total + b*N_pins + c*n_bins*log(n_bins))`, fit as total GP
    wall time regressed on the three n_iter-scaled predictors -- the
    intercept of that regression *is* T_fixed (a one-time, not-per-
    iteration cost) recovered in the same OLS as a/b/c, rather than a
    separately estimated constant. Points lacking `n_iter` (e.g. a schema-A
    case-level run that never recorded `gp_iterations`) are excluded."""
    usable = [pt for pt in points if pt.get("n_iter")]
    cases = [pt["case"] for pt in usable]
    X = np.array([
        [pt["n_iter"] * pt["n_total"], pt["n_iter"] * pt["n_pins"],
         pt["n_iter"] * pt["n_bins"] * np.log(pt["n_bins"])]
        for pt in usable
    ])
    y = np.array([pt["t_gp_s"] for pt in usable])
    fit = fit_ols_with_gate(X, y, ["n_iter*N_total", "n_iter*N_pins", "n_iter*n_bins*log(n_bins)"],
                             cases)
    fit["response"] = "t_gp_s"
    fit["response_unit"] = UNITS["t_gp_s"]
    fit["predictor_units"] = {
        "n_iter*N_total": "count * count", "n_iter*N_pins": "count * count",
        "n_iter*n_bins*log(n_bins)": "count * count * dimensionless",
    }
    fit["excluded_for_missing_n_iter"] = [pt["case"] for pt in points if not pt.get("n_iter")]
    fit["intercept_is_T_fixed_seconds"] = fit["coefficients"]["intercept"]["value"]
    return fit


def build_model_fit(t0b_dir=T0B_DIR_DEFAULT, case_json_paths=()):
    points, skipped = load_points(t0b_dir, case_json_paths)
    design_matrix = [
        dict(case=pt["case"], n_total=pt["n_total"], n_pins=pt["n_pins"],
             n_bins=pt["n_bins"], gp_peak_bytes=pt["gp_peak_bytes"],
             t_gp_s=pt["t_gp_s"], n_iter=pt["n_iter"], source_path=pt["source_path"])
        for pt in points
    ]

    memory_model = fit_gp_memory_model(points) if points else None
    runtime_points = [pt for pt in points if pt.get("n_iter")]
    runtime_model = fit_gp_runtime_model(points) if runtime_points else None

    identifiable = bool(memory_model and memory_model["identifiable"]
                         and runtime_model and runtime_model["identifiable"])
    notes = []
    if not identifiable:
        notes.append("identifiable=false: this model must not be used for any "
                      "extrapolation (design draft sec 7.1 T0b acceptance)")

    return dict(
        t0b_dir=t0b_dir, case_json_paths=list(case_json_paths),
        design_matrix=design_matrix, design_matrix_units=UNITS,
        n_points_used=len(points), skipped=[dict(path=p, reason=r) for p, r in skipped],
        gp_memory_model=memory_model, gp_runtime_model=runtime_model,
        identifiable=identifiable, notes=notes,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t0b-dir", default=T0B_DIR_DEFAULT)
    ap.add_argument("--case-json", action="append", default=[], dest="case_json_paths",
                     help="additional case-level run JSON (repeatable), e.g. "
                          "results/m4/probes/probe_gp_memory__adaptec1.json")
    ap.add_argument("--out", default=OUT_PATH_DEFAULT)
    args = ap.parse_args()

    result = build_model_fit(t0b_dir=args.t0b_dir, case_json_paths=args.case_json_paths)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    verdict = "identifiable" if result["identifiable"] else "NOT identifiable"
    print(f"[m4_model_fit] {result['n_points_used']} point(s) -> {args.out}: {verdict}")


if __name__ == "__main__":
    main()
