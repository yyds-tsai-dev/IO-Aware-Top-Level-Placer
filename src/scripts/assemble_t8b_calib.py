#!/usr/bin/env python3
"""M4 T8b: assemble per-case rho* calibration records from the seed-A grid runs.

Selection rule (pre-registered, spec 2026-08-13-m4-scale-up-design-draft.md §7.1 T8b):
take the LARGEST rho on the fixed candidate grid whose Δhpwl vs the case's own
flat run (same seed A) is <= +5%. Selection is constraint-based only — Δio is
recorded but never consulted (winner-bias control). Report-grade runs must use
seed B and their run_ids must not appear in this calibration set.

Usage: python3 scripts/assemble_t8b_calib.py <case> [<case> ...]
  <case> is the t8b filename prefix, e.g. group / sb12 / bb4 / cluster.
Reads  results/m4/t8b/<case>__k16__grid__{flat,rho*}.json
Writes results/m4/calib/rho_<case>.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REPO = REPO.parent if REPO.name == "src" else REPO
T8B = REPO / "results" / "m4" / "t8b"
CALIB = REPO / "results" / "m4" / "calib"

RHO_GRID = ["0.05", "0.10", "0.15", "0.20"]  # M3 T5 grid, reused per spec
DHPWL_LIMIT_PCT = 5.0
SEED_A = 1000
SEED_B_REGISTERED = 2000


def load(case: str, name: str) -> dict:
    path = T8B / f"{case}__k16__grid__{name}.json"
    with open(path) as f:
        d = json.load(f)
    if d.get("status") not in (None, "ok"):
        raise SystemExit(f"{path}: status={d.get('status')!r}, not usable for calibration")
    return d


def row(d: dict, flat_hpwl: float) -> dict:
    hpwl = d["hpwl"]
    return {
        "rho_max": d["rho_max"],
        "hpwl": hpwl,
        "dhpwl_pct": (hpwl - flat_hpwl) / flat_hpwl * 100.0,
        "io_count": d["io_count"],
        "ft_count": d["ft_count"],
        "final_overflow": d["final_overflow"],
        "legalization_status": d["legalization_status"],
        "num_unplaced_cells": d["num_unplaced_cells"],
        "gp_iterations_run": d.get("gp_iterations_run"),
        "seed": d["seed"],
        "run_id": d["run_id"],
        "repo_commit": d.get("repo_commit"),
    }


def assemble(case: str) -> Path:
    flat = load(case, "flat")
    assert flat["seed"] == SEED_A, f"{case} flat seed {flat['seed']} != seed A {SEED_A}"
    flat_hpwl = flat["hpwl"]
    grid_rows = []
    for rho in RHO_GRID:
        d = load(case, f"rho{rho}")
        assert d["seed"] == SEED_A, f"{case} rho{rho} seed {d['seed']} != seed A {SEED_A}"
        grid_rows.append(row(d, flat_hpwl))

    eligible = [r for r in grid_rows
                if r["dhpwl_pct"] <= DHPWL_LIMIT_PCT
                and r["legalization_status"] == "success"
                and r["num_unplaced_cells"] == 0]
    rho_star = max((r["rho_max"] for r in eligible), default=0.0)

    out = {
        "task": "m4-t8b",
        "case": case,
        "k": 16,
        "rtype": "grid",
        "seed_a": SEED_A,
        "seed_b_registered": SEED_B_REGISTERED,
        "selection_rule": (
            f"largest rho on fixed grid {RHO_GRID} with dhpwl <= +{DHPWL_LIMIT_PCT}% vs own flat "
            "(seed A), legalization success and 0 unplaced; constraint-based, Δio never consulted "
            "(winner-bias control). Grid may not be extended post hoc (M4-G2)."
        ),
        "flat_ref": row(flat, flat_hpwl),
        "grid": grid_rows,
        "rho_star": rho_star,
        "rho_star_at_grid_boundary": rho_star == float(RHO_GRID[-1]),
        "calibration_run_ids": [flat["run_id"]] + [r["run_id"] for r in grid_rows],
    }
    CALIB.mkdir(parents=True, exist_ok=True)
    path = CALIB / f"rho_{case}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{case}: rho*={rho_star}  (boundary={out['rho_star_at_grid_boundary']})  -> {path}")
    for r in grid_rows:
        print(f"  rho={r['rho_max']:<5} dhpwl={r['dhpwl_pct']:+.2f}%  io={r['io_count']}  ft={r['ft_count']}")
    return path


if __name__ == "__main__":
    for case in sys.argv[1:] or ["group", "sb12", "bb4"]:
        assemble(case)
