#!/usr/bin/env python3
"""M4 T10 E5 "honest forecast" generator (design draft
`docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md` sec 6.3 E5
/ sec 6.4 H100 交接契約 / sec 7.1 T10 row).

E5 is explicitly **not** a precision prediction: "刪除第二層容忍帶", "誠實
forecast = 寬區間 + 明示假設,不是精準預測". This script reads three
already-measured/frozen L4 artifacts and turns them into a phase-level
Amdahl wall-time band, an analytic GPU-memory bound, and a (deliberately
unfit-labeled) host-RSS point estimate for the 27.7M single-arm H100 run
-- every multiplier used is an explicit, auditable field in the output
JSON, never a hardcoded conclusion:

  1. `results/m4/t8b/cluster__k16__grid__flat.json` -- the L4-measured
     `phases`/`t_read`/`t_gp`/`t_lg`/`t_eval` reference point (11.31M
     `mempool_cluster`, K=16, rho_max=0.0 "flat" arm -- io driver mode
     with the IO term weighted to zero, the established T8b convention).
  2. `results/m4/scaling/count_freeze_30m.json` -- the T7b-frozen actual
     27.7M (3x3) node/net/pin counts (`base_n_nodes`/`total_n_nets`/
     `total_n_pins`, glue included).
  3. `results/m4/probes/probe_gp_memory__mempool_cluster_probe2500.json`
     -- a clean, GP-only (no evaluator) memory probe on the *same*
     11.31M netlist, used both as the L4 GP-memory reference and as the
     node/net/pin count denominator for every scale ratio below (its
     `num_physical`/`num_nets`/`num_pins` describe the identical
     `mempool_cluster` netlist file #1's run measured, just counted by a
     different code path).

Two-stage extrapolation, both stages explicit:

  stage A (scale-up, same GPU): multiply each L4-measured quantity by
    27.7M/11.3M count ratio (`compute_scale_ratios`) -- `node_ratio` for
    node-count-dominated phases (gp/lg/read), `pin_ratio` for the pin-
    one-hot-dominated evaluator phase (design draft sec 4.1: `(P,K)` is
    the single largest evaluator tensor). This stage carries no L4->H100
    hardware assumption at all; it is a same-GPU size extrapolation.
  stage B (L4 -> H100 hardware): divide by a per-phase `s_p` speedup
    assumption. Per this task's explicit instructions (not sec 6.3's own
    3-scenario S-BW/S-SM/S-FP64 table, which this script does not
    implement): read/lg get a fixed `s_p=1.0` (CPU-bound, GPU generation
    does not move them); gp/eval get the band `s_p in [2.5, 5.0]`
    ("A100/H100 vs L4 常見帶", labeled 未驗證假設 throughout). The
    resulting [fast, slow] total is a hardware-assumption sensitivity
    band, not a statistical prediction interval (sec 6.3(ii): "首跑不產
    生 prediction interval").

GPU memory follows sec 6.3 E5(i)'s literal formula:
    upper_bound = lower_bound + max_p transient_p + allocator_reserve + cuda_context
`lower_bound` = the larger of the two (gp, eval) scaled `peak_alloc_gb`
components (the smaller one becomes `max_phase_transient_gb`, standing in
for "the other component staying resident if co-resident with the
dominant one" -- sec 2.2's conservative-upper-bound caveat, since no T9
full-lifetime spike has run at 27.7M yet). `allocator_reserve_gb` is the
larger observed L4 reserved-minus-alloc gap, scaled the same way.
`cuda_context_gb` uses the L4-measured `device_baseline_gb` as a
placeholder for the (unmeasured) H100 constant -- sec 6.3: "cuda_context
取 H100 的實測常數(T14 首跑補)".

Host RSS gets **no interval** (sec 6.3 E5(iii): "僅在 §1.1 的模型通過辨識
門檻時給 95% PI;否則只給解析估計與其假設") -- `count_freeze_small.json`'s
own `host_rss_three_coefficient_model.status == "pending_t0b"` confirms
that model has not been fit yet, so this script emits a labeled point
estimate only.

Freeze semantics: the canonical output path (`results/m4/forecast/
h100_prediction.json`) is never overwritten silently -- an existing file
at `--out` requires `--force`. This script does not itself decide *when*
the official freeze happens (that is a scheduler-level call, gated on the
T8 seed-B 3.1M results per the task instructions that commissioned this
script); run it with an explicit `--out .../h100_prediction.draft.json`
for a non-committal demo run.

CLI:
    m4_forecast.py [--cluster-flat PATH] [--count-freeze PATH]
                    [--probe-gp-memory PATH] [--out PATH] [--force]
                    [--freeze]
"""
import argparse
import datetime
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_CLUSTER_FLAT = REPO / "results/m4/t8b/cluster__k16__grid__flat.json"
DEFAULT_COUNT_FREEZE = REPO / "results/m4/scaling/count_freeze_30m.json"
DEFAULT_PROBE_GP_MEMORY = REPO / "results/m4/probes/probe_gp_memory__mempool_cluster_probe2500.json"
DEFAULT_OUT = REPO / "results/m4/forecast/h100_prediction.json"

DESIGN_DRAFT = "docs/superpowers/specs/2026-08-13-m4-scale-up-design-draft.md"

# design draft sec 6.3 E5 "登錄內容" table, verbatim -- if the actual H100
# machine differs, the design draft's own rule is "預測無效,須重登錄",
# not an adjustment to this constant.
H100_SKU = {
    "name": "H100 SXM5 80GB HBM3",
    "peak_memory_bandwidth_GBps": 3350.0,
    "sm_arch": "sm_90",
    "cuda_version": "12.8",
    "persistence_mode": "on",
    "mig": False,
    "clocks": "default (not locked/boosted)",
    "note": "sec 6.3 E5 登錄內容: 若實際機器不同(SKU/時脈/CUDA build)"
            "⇒預測無效,須重登錄而非事後放寬",
}

# design draft sec 6.3(ii)'s phase table, transcribed per this task's
# explicit instructions (a simplification of sec 6.3's own 3-scenario
# S-BW/S-SM/S-FP64 table into one [lo, hi] speedup band per phase, framed
# as memory-bandwidth-ratio / FP32-throughput-ratio evidence, not a
# statistical interval): (s_p_lo, s_p_hi), and which count ratio drives
# stage-A's same-GPU size extrapolation for that phase.
PHASE_S_BAND = {
    "read": (1.0, 1.0),
    "gp": (2.5, 5.0),
    "lg": (1.0, 1.0),
    "eval": (2.5, 5.0),
}
PHASE_S_BASIS = {
    "read": "s_p=1.0: CPU-only single-threaded parser, unaffected by GPU generation",
    "gp": "s_p in [2.5, 5.0]: memory-bandwidth-ratio / FP32-throughput-ratio band "
          "(A100/H100 vs L4, common range) -- 未驗證假設, task instructions",
    "lg": "s_p=1.0: legalization is mostly CPU + small kernels",
    "eval": "s_p in [2.5, 5.0]: same band as gp (lattice walk + scatter, memory-"
            "bandwidth/FP32-throughput-bound) -- 未驗證假設, task instructions",
}
PHASE_COUNT_RATIO_BASIS = {
    "read": "node_ratio",
    "gp": "node_ratio",
    "lg": "node_ratio",
    "eval": "pin_ratio",   # sec 4.1: (P,K) pin one-hot is the dominant evaluator tensor
}
PHASE_ORDER = ("read", "gp", "lg", "eval")


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _load_json(path):
    with open(path) as f:
        return json.load(f)


_SKU_REQUIRED_FIELDS = ("name", "sm_arch", "cuda_version", "mig",
                        "persistence_mode", "clocks")


def _validate_sku(sku):
    if not isinstance(sku, dict):
        raise ValueError("SKU JSON must contain an object")
    missing = [key for key in _SKU_REQUIRED_FIELDS if key not in sku]
    if missing:
        raise ValueError("SKU JSON missing required fields: " + ", ".join(missing))
    if not isinstance(sku["name"], str) or not sku["name"].strip():
        raise ValueError("SKU name must be a non-empty string")
    for key in ("sm_arch", "cuda_version", "persistence_mode", "clocks"):
        if not isinstance(sku[key], str) or not sku[key].strip():
            raise ValueError(f"SKU {key} must be a non-empty string")
    if type(sku["mig"]) is not bool:
        raise ValueError("SKU mig must be a boolean")
    return dict(sku)


def _load_sku_json(path):
    sku = _validate_sku(_load_json(path))
    sku["source_path"] = str(path)
    sku["source_sha256"] = sha256_file(path)
    return sku


# ---------------------------------------------------------------------------
# stage A: scale ratios (27.7M / 11.3M, same GPU)
# ---------------------------------------------------------------------------

def compute_scale_ratios(count_freeze, probe_gp):
    """27.7M (count_freeze_30m.json, T7b-frozen actual counts, glue
    included) divided by 11.3M (probe_gp_memory's own counts for the same
    `mempool_cluster` netlist) -- a same-GPU size extrapolation, no L4/
    H100 hardware assumption involved."""
    node_ratio = count_freeze["base_n_nodes"] / probe_gp["num_physical"]
    net_ratio = count_freeze["total_n_nets"] / probe_gp["num_nets"]
    pin_ratio = count_freeze["total_n_pins"] / probe_gp["num_pins"]
    return {
        "node_ratio": node_ratio,
        "net_ratio": net_ratio,
        "pin_ratio": pin_ratio,
        "basis": "count_freeze_30m.json's {base_n_nodes, total_n_nets, total_n_pins} "
                 "(27.7M, T7b-frozen actual counts incl. glue) divided by "
                 "probe_gp_memory's {num_physical, num_nets, num_pins} (11.31M "
                 "mempool_cluster) -- same netlist family, same-GPU size "
                 "extrapolation only (no H100 assumption here, see PHASE_S_BAND "
                 "for that)",
        "unvalidated": True,
    }


# ---------------------------------------------------------------------------
# wall-time forecast (stage A count scale-up, then stage B s_p speedup)
# ---------------------------------------------------------------------------

def build_wall_time_forecast(cluster_flat, ratios):
    phases_l4 = cluster_flat["phases"]
    out_phases = {}
    total_fast_s = 0.0
    total_slow_s = 0.0
    for name in PHASE_ORDER:
        t_l4_s = phases_l4[name]["t_s"]
        ratio_basis = PHASE_COUNT_RATIO_BASIS[name]
        ratio = ratios[ratio_basis]
        t_extrap_27m_s = t_l4_s * ratio
        s_lo, s_hi = PHASE_S_BAND[name]
        # larger s_p (faster H100) -> smaller time; s_hi is always the
        # optimistic/fast end, s_lo the pessimistic/slow end.
        t_h100_fast_s = t_extrap_27m_s / s_hi
        t_h100_slow_s = t_extrap_27m_s / s_lo
        total_fast_s += t_h100_fast_s
        total_slow_s += t_h100_slow_s
        out_phases[name] = {
            "t_l4_measured_s": t_l4_s,
            "count_scale_ratio": ratio,
            "count_scale_basis": ratio_basis,
            "t_l4_extrapolated_27m_s": t_extrap_27m_s,
            "s_p_lo": s_lo,
            "s_p_hi": s_hi,
            "s_p_basis": PHASE_S_BASIS[name],
            "t_h100_fast_s": t_h100_fast_s,
            "t_h100_slow_s": t_h100_slow_s,
        }
    return {
        "unit": "seconds",
        "phases": out_phases,
        "total_fast_s": total_fast_s,
        "total_slow_s": total_slow_s,
        "total_fast_h": total_fast_s / 3600.0,
        "total_slow_h": total_slow_s / 3600.0,
        "interval_type": "hardware-assumption sensitivity band (NOT a statistical "
                          "prediction interval -- design draft sec 6.3 E5(ii): "
                          "\"首跑不產生 prediction interval\")",
        "unvalidated": True,
    }


# ---------------------------------------------------------------------------
# GPU memory forecast (sec 6.3 E5(i)'s literal lower/upper bound formula)
# ---------------------------------------------------------------------------

def build_gpu_memory_forecast(cluster_flat, probe_gp, ratios):
    node_ratio = ratios["node_ratio"]
    pin_ratio = ratios["pin_ratio"]

    gp_l4_alloc_gb = probe_gp["gp_peak_alloc_mb"] / 1024.0
    gp_l4_reserved_gb = probe_gp["gp_peak_reserved_mb"] / 1024.0
    eval_phase = cluster_flat["phases"]["eval"]
    eval_l4_alloc_gb = eval_phase["peak_alloc_gb"]
    eval_l4_reserved_gb = eval_phase["peak_reserved_gb"]

    components = {
        "gp": {
            "l4_peak_alloc_gb": gp_l4_alloc_gb,
            "l4_peak_reserved_gb": gp_l4_reserved_gb,
            "count_scale_ratio": node_ratio,
            "count_scale_basis": "node_ratio",
            "scaled_alloc_gb": gp_l4_alloc_gb * node_ratio,
            "scaled_reserved_gb": gp_l4_reserved_gb * node_ratio,
        },
        "eval": {
            "l4_peak_alloc_gb": eval_l4_alloc_gb,
            "l4_peak_reserved_gb": eval_l4_reserved_gb,
            "count_scale_ratio": pin_ratio,
            "count_scale_basis": "pin_ratio",
            "scaled_alloc_gb": eval_l4_alloc_gb * pin_ratio,
            "scaled_reserved_gb": eval_l4_reserved_gb * pin_ratio,
        },
    }

    scaled_allocs = {n: c["scaled_alloc_gb"] for n, c in components.items()}
    dominant = max(scaled_allocs, key=scaled_allocs.get)
    other = min(scaled_allocs, key=scaled_allocs.get)

    lower_bound_gb = scaled_allocs[dominant]
    max_phase_transient_gb = scaled_allocs[other]
    allocator_reserve_gb = max(
        c["scaled_reserved_gb"] - c["scaled_alloc_gb"] for c in components.values()
    )
    allocator_reserve_gb = max(allocator_reserve_gb, 0.0)
    # sec 6.3: "cuda_context 取 H100 的實測常數(T14 首跑補)" -- not measured
    # yet, so the L4-measured device_baseline_gb (a real number, not a
    # fabricated guess) stands in as the placeholder.
    cuda_context_gb = cluster_flat.get("device_baseline_gb", 0.5)

    upper_bound_gb = lower_bound_gb + max_phase_transient_gb + allocator_reserve_gb + cuda_context_gb

    return {
        "unit": "GB",
        "components": components,
        "dominant_component": dominant,
        "lower_bound_gb": lower_bound_gb,
        "lower_bound_basis": f"scaled peak_alloc_gb of the dominant component ({dominant}) "
                              "-- an analytic resident-tensor account substitute (design draft "
                              "sec 6.3 E5(i)'s resident_tensors term), not a full per-tensor "
                              "dtype x shape rebuild",
        "max_phase_transient_gb": max_phase_transient_gb,
        "allocator_reserve_gb": allocator_reserve_gb,
        "cuda_context_gb": cuda_context_gb,
        "cuda_context_basis": "L4-measured device_baseline_gb used as a placeholder for the "
                               "(unmeasured) H100 constant -- T14's first run should replace this",
        "upper_bound_gb": upper_bound_gb,
        "formula": "design draft sec 6.3 E5(i): upper_bound = lower_bound + "
                   "max_p transient_p + allocator_reserve + cuda_context "
                   "(解析上下界,不是統計區間)",
        "unvalidated": True,
    }


# ---------------------------------------------------------------------------
# host RSS forecast (no interval -- model unfit, sec 6.3 E5(iii))
# ---------------------------------------------------------------------------

def build_host_rss_forecast(cluster_flat, ratios):
    node_ratio = ratios["node_ratio"]
    l4_host_peak_rss_gb = cluster_flat["host_peak_rss_gb"]
    point_estimate_gb = l4_host_peak_rss_gb * node_ratio
    return {
        "unit": "GB",
        "model_status": "unfit: design draft sec 1.1's 3-coefficient PlaceDB.read host RSS "
                         "model (N_nodes/N_raw_pins fitting set) has not been fit yet -- "
                         "results/m4/scaling/count_freeze_small.json's "
                         "host_rss_three_coefficient_model.status == \"pending_t0b\"",
        "l4_host_peak_rss_gb": l4_host_peak_rss_gb,
        "count_scale_ratio": node_ratio,
        "count_scale_basis": "node_ratio (proxy only -- the eventual model also depends on "
                              "N_raw_pins, not represented here)",
        "point_estimate_gb": point_estimate_gb,
        "interval": None,
        "h100_min_host_ram_gb_contract": 192.0,
        "note": "design draft sec 6.3 E5(iii): \"僅在 §1.1 的模型通過辨識門檻時給 95% PI;"
                "否則只給解析估計與其假設\" -- this is the \"否則\" branch: point estimate "
                "only, no prediction interval.",
        "unvalidated": True,
    }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_prediction(cluster_flat, count_freeze, probe_gp, *,
                      cluster_flat_path, count_freeze_path, probe_gp_path,
                      status="draft", sku=None):
    ratios = compute_scale_ratios(count_freeze, probe_gp)
    l4_gpu_name = cluster_flat.get("env", {}).get("gpu_name", "NVIDIA L4")

    return {
        "schema_version": 1,
        "task": "T10",
        "status": status,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "scripts/m4_forecast.py",
        "generator_commit": _git_head(),
        "sku": dict(H100_SKU if sku is None else _validate_sku(sku)),
        "l4_reference": {
            "name": l4_gpu_name,
            "note": "the L4 device the three input measurements below were taken on",
        },
        "inputs": {
            "cluster_flat_l4": {"path": str(cluster_flat_path), "sha256": sha256_file(cluster_flat_path)},
            "count_freeze_30m": {"path": str(count_freeze_path), "sha256": sha256_file(count_freeze_path)},
            "probe_gp_memory": {"path": str(probe_gp_path), "sha256": sha256_file(probe_gp_path)},
        },
        "scale_ratios": ratios,
        "wall_time_forecast": build_wall_time_forecast(cluster_flat, ratios),
        "gpu_memory_forecast": build_gpu_memory_forecast(cluster_flat, probe_gp, ratios),
        "host_rss_forecast": build_host_rss_forecast(cluster_flat, ratios),
        "t14_check_protocol": (
            "design draft sec 6.3 E5 (T14 首跑的判定): 檢定假設,不是檢定區間 -- 逐 phase "
            "比較實測 T_H100,p 與 T_L4,p/s_p,報出每個 phase 的實際 s_p^obs;主線假設被判為"
            "成立僅當每個 phase 的 s_p^obs 與宣告值同序(相對誤差 <= 50%)且總時間落在本檔"
            "wall_time_forecast 的 [total_fast_s, total_slow_s] 之間。記憶體則直接比對是否"
            "落在 gpu_memory_forecast 的 [lower_bound_gb, upper_bound_gb] 解析界內。首跑不"
            "產生新的 prediction interval。"
        ),
        "refit_on_miss_note": "design draft T14 row: 任一落外 ⇒ 在報告發表重擬合模型與歸因,"
                               "不得事後放寬區間。(see scripts/m4_check_prediction.py)",
        "provenance": {
            "repo_commit": _git_head(),
            "hostname": socket.gethostname(),
            "spec_doc": DESIGN_DRAFT,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cluster-flat", default=str(DEFAULT_CLUSTER_FLAT))
    ap.add_argument("--count-freeze", default=str(DEFAULT_COUNT_FREEZE))
    ap.add_argument("--probe-gp-memory", default=str(DEFAULT_PROBE_GP_MEMORY))
    ap.add_argument("--sku-json", help="JSON object registering the actual target GPU SKU")
    ap.add_argument("--protocol-json", help="prospective workload and measurement protocol")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true",
                     help="overwrite --out even if it already exists")
    ap.add_argument("--freeze", action="store_true",
                     help="mark status=frozen in the output metadata (does not by "
                          "itself decide when the official freeze happens)")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"m4_forecast.py: REFUSING to overwrite existing forecast at {out_path} "
              "(pass --force to override)", file=sys.stderr)
        return 2

    cluster_flat = _load_json(args.cluster_flat)
    count_freeze = _load_json(args.count_freeze)
    probe_gp = _load_json(args.probe_gp_memory)
    sku = _load_sku_json(args.sku_json) if args.sku_json else None

    prediction = build_prediction(
        cluster_flat, count_freeze, probe_gp,
        cluster_flat_path=args.cluster_flat,
        count_freeze_path=args.count_freeze,
        probe_gp_path=args.probe_gp_memory,
        status="frozen" if args.freeze else "draft",
        sku=sku,
    )
    if args.protocol_json:
        protocol = _load_json(args.protocol_json)
        if not isinstance(protocol, dict) or not protocol.get("config_sha256"):
            raise ValueError("protocol must register a config_sha256")
        prediction["execution_protocol"] = protocol
        prediction["execution_protocol_sha256"] = sha256_file(args.protocol_json)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(prediction, f, indent=1)
        f.write("\n")
    print(f"m4_forecast.py: wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
