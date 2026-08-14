"""M3 T0-c (P0b) probe: preregistered S4 surrogate selection experiment.

Design draft `docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md`
v3.1 sec 3.4 (the only authoritative spec; sec 0/0.1 summaries are non-normative
whenever they disagree with sec 3.4's body). This is the "fast-worker can
implement mechanically" protocol sec 3.4.0 asks for -- 7 S4 candidates + an
IO-only reference arm, evaluated as *one-step directional perturbations* off
20 (state, tau_rel) cells (+ a bigblue4 spot-check + an S4e tie-break
sensitivity submatrix), scored against Pareto-vector tier-1/tier-2
preregistered criteria. This script does NOT pick a winner (sec 3.4.4's
`verdict` field is a mechanical tier-1/tier-2 evaluation, not a judgement
call) -- see sec 3.4.3's own text: "候選 c 通過 tier-1(整體)" is decided by a
fixed formula, not by this script's author.

Three points in the spec are genuinely underspecified and require a stated
assumption (each is flagged again at its point of use below and in this
probe's own `config.assumptions` output field -- report these to whoever
reviews the P0b results, they are not silent guesses):

  (A) S4g-bboxcov's third temperature tau_b (design draft sec 3.3.4 / phase-1
      spec sec 5.3) has no value anywhere in either spec doc, and sec 3.4.0's
      frozen table does not list it as swept. Assumption: tau_b := tau (the
      same ambient S1/S2 softmax temperature active at that (state, tau_rel)
      cell). The WA-min/max soft-bbox itself uses the standard RePlAce/
      DREAMPlace weighted-average form (see _wa_bbox below).
  (B) The "native tau_rel" of an A0 (flat-mode) final placement in sec
      3.4.1(b) requires that placement's final overflow, but flat-mode runs
      (ioplace/drivers/run_placement.py:run_flat) do not record overflow at
      all (unlike A2/io-mode runs, whose trajectory *does* record it -- used
      directly). Assumption: the flat GP phase terminates at essentially
      overflow == the DREAMPlace config's own `stop_overflow` (0.07 for both
      adaptec1.json and bigblue4.json, verified directly from the config
      files), which is *exactly* schedules.py's `of_end` convention already
      used everywhere else in M2/M3 -- so tau_rel_from_overflow(of_end, ...,
      of_end=of_end) == tau_lo == 0.03 identically, not an arbitrary number.
  (C) S4e's path(h,k) tie-break (sec 3.3.3: "BFS 樹上「region-id 序列字典序
      最小」的最短路") is realized via region_graph.next_hop_table's existing
      canonical rule (increasing-neighbour-index-first greedy from h), which
      is exactly the textbook algorithm for lexicographically-smallest
      shortest path (greedy prefix minimization is optimal because every
      suffix of a shortest path is itself a shortest path). region_graph.py's
      own docstring hedges that this is "a separate, simpler tie-break" than
      sec 3.3.3's rule without asserting non-equivalence; this probe treats
      them as the same rule but flags the reuse explicitly rather than
      silently assuming it. The independent "reverse tie-break" submatrix
      (sec 3.3.3 / sec 3.4.0) uses a mirrored decreasing-neighbour-index-first
      rule (_reverse_next_hop_table below) -- the natural opposite convention,
      by direct analogy to how P1 tested the reversed L-shape routing
      convention.

Topology classification (sec 3.4.5 / sec 1's five-class table) is *not* T1's
`per_net_topology_class` (design draft sec 2.5 item 4, not yet implemented as
of this probe's HEAD -- see sec 8's dependency note: "若 T1 未完成,P0b 可用
探針內的暫時實作,但必須在 T1 完成後重跑一次確認一致"). This probe's
`_topology_classes` is exactly that permitted temporary, self-built
implementation, and every cell in the output JSON's `env` block records
`topology_classifier: "temporary-p0b"` so a later T1-consistency rerun is
identifiable. It classifies `chain` vs `star` (the only two classes needing
more than (Lambda, ST/FT) alone) by building the Lambda-1-edge MST over just
the touched terminals (weighted by G_R hop distance D) and checking home's
tree-degree: a `FT==0` net's optimal Steiner tree cost is exactly Lambda-1
(the theoretical minimum for connecting Lambda terminals), which is only
achievable if *every* tree edge has D-weight exactly 1 -- i.e. the
terminal-only MST literally *is* the true minimum Steiner tree in this case
(no Steiner points can ever help), so this is exact, not a heuristic
approximation, for the only nets it's asked to resolve (Lambda>=3, FT==0).

Reused M2/M3 assets (per the task's "可用資產" list): ioplace/region_graph.py
(region_graph/steiner_tree_stats/next_hop_table), ioplace/evaluator_gpu.py
(evaluate_gpu -- io_rg/ft_rg/per_net_steiner/per_net_home/per_net_lambda come
straight from here per T1, commit 0a2e32c; ft_mst/io_mst are its legacy
ft_count/io_count fields, sec 2.1's naming), ioplace/schedules.py
(tau_rel_from_overflow, derive_kappa_ft -- NOT ft_activation_ramp, which the
task explicitly says P0b does not need), and the S1/S2 soft-assign primitives
(ioplace/ops/soft_assign.py) via the same direct-tensor pattern
probe_ft_surrogate_soft.py (P0) already established, rather than going
through IoTerm/IoTermRef (P0b is offline/one-step, not a driver hot path).

Usage (see this file's `main()` / the task's chunked-execution instructions):
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m3.probe_p0b --group b --etas 0.005,0.01,0.02
    PYTHONPATH=. $PY -m ioplace.diagnostics.probes_m3.probe_p0b --group assemble

Writes results/m3/probes/probe_p0b.checkpoint.jsonl (append-only, one JSON
object per completed cell/random-direction -- resumable) and, on `--group
assemble`, results/m3/probes/probe_p0b.json (design draft sec 3.4.6 schema).
"""
import argparse
import datetime
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from ioplace.drivers.run_placement import _load_dreamplace, get_regions_for
from ioplace.netlist import netlist_from_placedb, pin_positions
from ioplace.region_grid import RegionGrid
from ioplace.region_graph import region_graph as build_region_graph, next_hop_table
from ioplace.evaluator_gpu import evaluate_gpu
from ioplace.ops.soft_assign import rect_table, region_sdf_l1, softmax_stats, chunk_p_ell, _chunks
from ioplace.ops.io_term import build_net_node_csr
from ioplace.schedules import tau_rel_from_overflow, derive_kappa_ft

REPO = "/nashome/NVL4/vdalab/yyds-dev/IO-Aware-Top-Level-Placer"
DP = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"
DEV = "cuda"

ADAPTEC1_CFG = f"{DP}/install/test/ispd2005/adaptec1.json"
BIGBLUE4_CFG = f"{DP}/install/test/ispd2005/bigblue4.json"

OUT_JSON = os.path.join(REPO, "results/m3/probes/probe_p0b.json")
CKPT_JSONL = os.path.join(REPO, "results/m3/probes/probe_p0b.checkpoint.jsonl")

# ---------------------------------------------------------------------------
# sec 3.4.0 frozen cardinality
# ---------------------------------------------------------------------------
ETAS = (0.005, 0.01, 0.02)
F_FIXED = 0.25
N_RANDOM = 8
FAMILIES = ("merged_l1", "io_anchored_l1", "full_objective")
JUDGING_FAMILY = "merged_l1"
CONFIRMATORY_FAMILY = "full_objective"

CANDIDATES = ("S4a-star", "S4b-gated-detach_b0.5", "S4b-gated-detach_b1.0",
             "S4b-gated_b0.5", "S4e-union-detach", "S4e-union-hardbar",
             "S4g-bboxcov", "IO-only")
FT_CANDIDATES = tuple(c for c in CANDIDATES if c != "IO-only")

TOPO_CLASSES = ("trivial", "adjacent", "chain", "star", "true-FT")
TOPO_IDX = {c: i for i, c in enumerate(TOPO_CLASSES)}

# tier-2 regime gate (design draft sec 3.4.4 T2a/T2b/T2c apply only where
# f >= 0.5*f_max, solved via the sec 5.2 ramp to tau_rel <= 0.0775)
TIER2_TAU_REL_MAX = 0.0775
TIER2_MASS_ON_FT0_MAX = 0.50
TIER2_BUCKET_RATIO_RANGE = (0.5, 2.0)
TIER2_BUCKET_RATIO_MAX_ASYM = 5.0

SNAPSHOT_ITERS = (300, 350, 400, 450, 500, 550)
# ^ sec 3.4.1(a)'s prose lists "300,400,450,500,550,600", but the actual T0-a
# output (results/m3/snapshots/manifest.json, commit 157a198) is 300/350/
# 400/450/500/550 -- confirmed against disk (`ls results/m3/snapshots/flat/`)
# and the coordinator's report ("it0300.npz ... it0550.npz 共 12 個快照";
# measured tau_rel it300~=0.224/0.226, it450~=0.0913/0.0962, it550~=0.042).
# Treating the actual produced artifact as authoritative over the spec
# prose's iteration list (the *count*, 6 per tag x 2 tags = 12, and the
# tau_rel coverage down to ~0.03-0.04, both match sec 3.4.1(a)'s intent).

# S4e-union net-chunk budget (elements in one (chunk,K,A) fp64 gather); keeps
# peak memory bounded regardless of net count (adaptec1 ~217k vs bigblue4
# ~2.07M active nets) -- see build_candidate_L's S4e branch.
_S4E_CHUNK_BUDGET = 4_000_000
# S4g-bboxcov net-chunk budget (elements in one (chunk,maxdeg) WA-min/max
# tensor) -- see build_candidate_L_bboxcov.
_S4G_CHUNK_BUDGET = 2_000_000
# S1/S2 forward K-chunk budget (elements in one (N,k_chunk) intermediate,
# N == case's physical node count) -- see _k_chunk_for.
_K_CHUNK_ELEM_BUDGET = 20_000_000


# ---------------------------------------------------------------------------
# provenance (pattern copied from probe_ft_surrogate_soft.py / probe_m3_*.py)
# ---------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git_head(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()


def _env_metadata(argv, input_relpaths):
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "repo_commit": _git_head(REPO),
        "dp_commit": _git_head(DP),
        "command": f"{sys.executable} -m ioplace.diagnostics.probes_m3.probe_p0b",
        "argv": list(argv),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # assemble() may run before every state's inputs exist (e.g. the
        # trajectory-snapshot npz files, produced by a separate T0-a run this
        # probe polls for) -- hash whatever is present and record the rest as
        # missing rather than crashing, so a partial/interim assemble (useful
        # while iterating) still produces a valid provenance block.
        "input_sha256": {p: _sha256(os.path.join(REPO, p)) for p in sorted(set(input_relpaths))
                         if os.path.exists(os.path.join(REPO, p))},
        "input_missing": sorted(p for p in set(input_relpaths)
                               if not os.path.exists(os.path.join(REPO, p))),
        "topology_classifier": "temporary-p0b",
    }


def _spec_sha256():
    return _sha256(os.path.join(
        REPO, "docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md"))


# ---------------------------------------------------------------------------
# G_R edge indexing + path masks (S4e, sec 3.3.3) + reverse tie-break
# ---------------------------------------------------------------------------
def _edge_index(adj):
    K = adj.shape[0]
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K) if adj[a, b]]
    eid = -np.ones((K, K), dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        eid[a, b] = i
        eid[b, a] = i
    return eid, len(pairs)


def _reverse_next_hop_table(adj, D):
    """Mirror of region_graph.next_hop_table: decreasing-neighbour-index
    tie-break instead of increasing. Used only by the sec 3.3.3 / sec 3.4.0
    S4e reverse-tie-break sensitivity submatrix (assumption (C) above)."""
    K = adj.shape[0]
    nh = -np.ones((K, K), dtype=np.int64)
    for v in range(K):
        nh[v, v] = v
    neighbours = [np.nonzero(adj[a])[0].tolist()[::-1] for a in range(K)]
    Di = D.astype(np.int64)
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            for c in neighbours[a]:
                if Di[c, b] == Di[a, b] - 1:
                    nh[a, b] = c
                    break
    return nh


def _build_path_mask(next_hop, eid, A, K):
    """(K,K,A) bool: path_mask[h,k,a] iff edge a is on the canonical shortest
    path from h to k (sec 3.3.3's `path(h,k)`)."""
    pm = np.zeros((K, K, A), dtype=bool)
    for h in range(K):
        for k in range(K):
            if h == k:
                continue
            cur = h
            while cur != k:
                nxt = int(next_hop[cur, k])
                pm[h, k, eid[cur, nxt]] = True
                cur = nxt
    return pm


# ---------------------------------------------------------------------------
# static per-(config,k,rtype) case
# ---------------------------------------------------------------------------
_NL_CACHE = {}


def _load_nl_case(cfg):
    if cfg not in _NL_CACHE:
        params, placedb = _load_dreamplace(cfg)
        placedb.initialize(params)
        nl = netlist_from_placedb(placedb)
        _NL_CACHE[cfg] = dict(params=params, placedb=placedb, nl=nl)
    return _NL_CACHE[cfg]


def load_region_case(cfg, k, rtype, dev=DEV, region_seed=0):
    base = _load_nl_case(cfg)
    params, placedb, nl = base["params"], base["placedb"], base["nl"]
    die = (float(placedb.xl), float(placedb.yl), float(placedb.xh), float(placedb.yh))
    of_end = float(getattr(params, "stop_overflow", 0.07))

    rs = get_regions_for(die, k, rtype, region_seed)
    rg = RegionGrid(rs)
    rects, r2k = rect_table(rs)
    adj, D, ell = build_region_graph(rg)
    ecc_max = D.max(axis=1)
    eid, A = _edge_index(adj)
    nh_primary = next_hop_table(adj, D)
    nh_reverse = _reverse_next_hop_table(adj, D)
    pm_primary = _build_path_mask(nh_primary, eid, A, k)
    pm_reverse = _build_path_mask(nh_reverse, eid, A, k)

    ignore_net_degree = int(params.ignore_net_degree)
    csr = build_net_node_csr(nl, ignore_net_degree)

    # free-area util denominators (probe_free_area_util.py's region_free_area,
    # inlined -- static, independent of node positions)
    nm = nl.num_movable
    fx0, fy0 = nl.node_x[nm:], nl.node_y[nm:]
    fx1, fy1 = fx0 + nl.node_size_x[nm:], fy0 + nl.node_size_y[nm:]
    n_regions = len(rs.regions)
    region_area = np.zeros(n_regions)
    overlap_area = np.zeros(n_regions)
    for rid, reg in enumerate(rs.regions):
        for (rxl, ryl, rxh, ryh) in reg.rects:
            region_area[rid] += (rxh - rxl) * (ryh - ryl)
            ow = np.clip(np.minimum(fx1, rxh) - np.maximum(fx0, rxl), 0, None)
            oh = np.clip(np.minimum(fy1, ryh) - np.maximum(fy0, ryl), 0, None)
            overlap_area[rid] += float((ow * oh).sum())
    free_area = region_area - overlap_area

    L_R = ((die[2] - die[0]) * (die[3] - die[1]) / k) ** 0.5

    dev_t = torch.device(dev)
    return dict(
        cfg=cfg, k=k, rtype=rtype, params=params, placedb=placedb, nl=nl, rg=rg, die=die,
        of_end=of_end, csr=csr, ignore_net_degree=ignore_net_degree,
        rects_t=torch.as_tensor(rects, dtype=torch.float64, device=dev_t),
        r2k_t=torch.as_tensor(r2k, dtype=torch.int64, device=dev_t),
        D=D, D_t=torch.as_tensor(D.astype(np.int64), dtype=torch.float64, device=dev_t),
        ecc_max=ecc_max, ecc_max_t=torch.as_tensor(ecc_max, dtype=torch.float64, device=dev_t),
        A=A, eid=eid,
        pm_primary_t=torch.as_tensor(pm_primary, dtype=torch.float64, device=dev_t),
        pm_reverse_t=torch.as_tensor(pm_reverse, dtype=torch.float64, device=dev_t),
        region_area=region_area, free_area=free_area, L_R=L_R, dev=dev_t,
    )


# ---------------------------------------------------------------------------
# state loading (sec 3.4.1)
# ---------------------------------------------------------------------------
def load_state(spec, dev=DEV):
    """spec: dict with keys tag, npz (relpath), case_key, and either
    forced_tau_rel or native_tau_rel (pre-resolved by the caller -- see the
    state-builder functions below for how each of sec 3.4.1(a)/(b)/(c)'s
    sources is turned into a tau_rel). Returns node_x/node_y (physical-length
    numpy), tau_rel, g_wl_density (optional torch tensor, snapshot states
    only, full node-length x-then-y layout -- caller slices to [:nm]/
    [n_all:n_all+nm])."""
    npz_path = os.path.join(REPO, spec["npz"])
    d = np.load(npz_path)
    node_x, node_y = d["node_x"].astype(np.float64), d["node_y"].astype(np.float64)
    if "forced_tau_rel" in spec:
        tau_rel = spec["forced_tau_rel"]
    elif spec.get("native_tau_rel") is not None:
        tau_rel = spec["native_tau_rel"]
    elif "tau_rel" in d.files:
        # sec 3.4.1(a) trajectory snapshots: T0-a stores tau_rel directly in
        # the npz (already derived via tau_rel_from_overflow at snapshot
        # time) -- snapshot_state_specs() deliberately leaves native_tau_rel
        # unset/None so this branch is what actually resolves it.
        tau_rel = float(d["tau_rel"])
    else:
        raise ValueError(f"state {spec['tag']}: no tau_rel source (neither spec nor npz)")

    g_wl_density = None
    lambda_io = None
    if "g_wl_density" in d.files:
        g_wl_density = torch.as_tensor(d["g_wl_density"], dtype=torch.float64, device=dev)
        lambda_io = float(d["lambda_io"]) if "lambda_io" in d.files else 0.0

    return dict(tag=spec["tag"], case_key=spec["case_key"], node_x=node_x, node_y=node_y,
               tau_rel=tau_rel, g_wl_density=g_wl_density,
               n_all=(g_wl_density.shape[0] // 2 if g_wl_density is not None else None),
               lambda_io=lambda_io, npz_sha256=_sha256(npz_path), npz_relpath=spec["npz"])


# ---------------------------------------------------------------------------
# soft quantities (S1+S2, direct-tensor pattern from probe_ft_surrogate_soft.py)
# ---------------------------------------------------------------------------
def _k_chunk_for(case):
    """K-chunk width for the S1/S2 forward+backward (mirrors ioplace/ops/
    io_term.py's IoTerm/_IoFn chunked-k contract, design v2 sec 2.5): an
    unchunked (N,K) forward at bigblue4 scale (N ~= 2.18M physical nodes)
    needs several (N,K)-shaped intermediates (region_sdf_l1's dx/dy/d/out,
    softmax_stats's z/e, chunk_p_ell's p/ell) *plus* whatever autograd saves
    of each for backward -- empirically this OOMs a 24GB GPU well before any
    candidate-specific work even starts (see this function's git-blame
    commit message / the task report for the memory_allocated()/
    memory_reserved() trace that pinned this down). Bound each such
    intermediate to roughly _K_CHUNK_ELEM_BUDGET elements; adaptec1's N
    (~210k-370k) is small enough that this evaluates to chunk >= K, i.e.
    _chunks(K, chunk) below still returns [(0,K)] -- bit-identical to the
    unchunked path, so this is a no-op there (verified by the sanity check
    staying exact after this change)."""
    N = case["nl"].num_physical
    return max(1, min(case["k"], _K_CHUNK_ELEM_BUDGET // max(N, 1)))


def _soft_S_q(x, y, case, tau):
    K = case["k"]
    k_chunk = _k_chunk_for(case)
    m, t, am = softmax_stats(x, y, case["rects_t"], case["r2k_t"], K, tau, chunk=k_chunk)
    csr = case["csr"]
    node_idx = torch.as_tensor(csr.flat_net2node, dtype=torch.int64, device=x.device)
    degrees = torch.as_tensor(csr.degrees, dtype=torch.int64, device=x.device)
    n_active = int(len(csr.net_ids))
    net_idx = torch.repeat_interleave(torch.arange(n_active, dtype=torch.int64, device=x.device),
                                      degrees)
    S_parts = []
    for lo, hi in _chunks(K, k_chunk):
        sdf_c = region_sdf_l1(x, y, case["rects_t"], case["r2k_t"], lo, hi)
        _, ell_c = chunk_p_ell(sdf_c, m, t, am, lo, tau)
        S_c = torch.zeros((n_active, hi - lo), dtype=torch.float64,
                          device=x.device).index_add_(0, net_idx, ell_c[node_idx].double())
        S_parts.append(S_c)
    S = torch.cat(S_parts, dim=1) if len(S_parts) > 1 else S_parts[0]
    q = -torch.expm1(S)
    return S, q, node_idx, net_idx, n_active


def _wa_bbox(px_net_pad, py_net_pad, valid_mask, tau_b):
    """Weighted-average soft bbox (assumption (A)): standard RePlAce/
    DREAMPlace WA-min/max, per net, over a (E, maxdeg) padded pin coordinate
    tensor (invalid slots masked to +-inf so they never win the softmax).
    Returns (xmin, xmax, ymin, ymax), each (E,)."""
    def wa_max(v):
        vv = torch.where(valid_mask, v, torch.full_like(v, -1e30))
        vmax = vv.max(dim=1, keepdim=True).values
        w = torch.where(valid_mask, torch.exp((vv - vmax) / tau_b), torch.zeros_like(vv))
        return (vv * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-300)

    def wa_min(v):
        vv = torch.where(valid_mask, v, torch.full_like(v, 1e30))
        vmin = vv.min(dim=1, keepdim=True).values
        w = torch.where(valid_mask, torch.exp(-(vv - vmin) / tau_b), torch.zeros_like(vv))
        return (vv * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-300)

    return wa_min(px_net_pad), wa_max(px_net_pad), wa_min(py_net_pad), wa_max(py_net_pad)


def _build_padded_pins(px, py, node_idx, net_idx, n_active, device):
    """px, py: already-gathered per-(net,node)-occurrence coordinates, shape
    == node_idx.shape == net_idx.shape. `net_idx` is sorted ascending by
    construction everywhere it's built in this file (torch.repeat_interleave
    over consecutive net ids), so the within-net slot index is a plain
    unique_consecutive-based running count -- no argsort needed."""
    _, cnt = torch.unique_consecutive(net_idx, return_counts=True)
    maxdeg = int(cnt.max().item())
    starts = torch.zeros_like(cnt)
    starts[1:] = torch.cumsum(cnt, 0)[:-1]
    starts_per_elem = torch.repeat_interleave(starts, cnt)
    within = torch.arange(net_idx.shape[0], device=device) - starts_per_elem

    xpad = torch.zeros((n_active, maxdeg), dtype=torch.float64, device=device)
    ypad = torch.zeros((n_active, maxdeg), dtype=torch.float64, device=device)
    valid = torch.zeros((n_active, maxdeg), dtype=torch.bool, device=device)
    xpad[net_idx, within] = px
    ypad[net_idx, within] = py
    valid[net_idx, within] = True
    return xpad, ypad, valid


# ---------------------------------------------------------------------------
# candidate F builders (sec 3.3)
# ---------------------------------------------------------------------------
def build_candidate_L(name, *, S, q, D_home, ecc_home, lam_soft, Lam_hard, case, home_e,
                      nl, node_idx, net_idx, n_active, x, y, tau):
    """Returns (L scalar, diag dict) for one candidate's L_FT_e sum. diag has
    whatever's needed for mass_on_ft0/bucket diagnostics upstream (li per-net
    numpy array)."""
    if name == "S4a-star":
        li = (q * torch.clamp(D_home - 1.0, min=0.0)).sum(dim=1)
        return li.sum(), dict(li=li.detach())

    if name.startswith("S4b-gated"):
        beta = 0.5 if "b0.5" in name else 1.0
        detach = "detach" in name
        q_r = torch.clamp(q - 0.5, min=0.0) / 0.5
        exponent = (D_home - ecc_home.unsqueeze(1)) / beta
        g = torch.exp(exponent)
        N = (q_r * g).sum(dim=1)
        M = (q_r * g * D_home).sum(dim=1)
        N_safe = torch.where(N > 0, N, torch.ones_like(N))
        R = torch.where(N > 0, M / N_safe, torch.zeros_like(N))
        base = (lam_soft - 1.0).detach() if detach else (lam_soft - 1.0)
        li = torch.relu(R - base)
        return li.sum(), dict(li=li.detach())

    if name.startswith("S4e-union"):
        # Net-chunked (sec 3.3.3's own memory note: "P0b 只需離線可算", but
        # bigblue4's ~2M active nets x K x A still OOMs a single (E,K,A)
        # gather on a 24GB GPU once earlier candidates' retained graphs are
        # also resident -- chunk over nets, never materializing more than
        # _S4E_CHUNK nets' worth of (chunk,K,A) at once. Each chunk's
        # contribution is independent, so concatenating preserves autograd
        # exactly (no different from the unchunked einsum, just bounded peak
        # memory) -- this is a performance fix, not a formula change.
        hardbar = "hardbar" in name
        pm = case["pm_primary_t"] if case.get("_tie_break", "primary") == "primary" else case["pm_reverse_t"]
        E = S.shape[0]
        chunk = max(1, _S4E_CHUNK_BUDGET // (case["k"] * max(case["A"], 1)))
        li_parts = []
        for lo in range(0, E, chunk):
            hi = min(E, lo + chunk)
            pm_home_c = pm[home_e[lo:hi]]                       # (c,K,A)
            S_edge_c = torch.einsum("ek,eka->ea", S[lo:hi], pm_home_c)   # (c,A)
            U_c = 1.0 - torch.exp(S_edge_c)
            li_parts.append(U_c.sum(dim=1))
        L_cross = torch.cat(li_parts)
        base = Lam_hard - 1.0 if hardbar else (lam_soft - 1.0).detach()
        li = torch.relu(L_cross - base)
        return li.sum(), dict(li=li.detach())

    raise ValueError(f"unknown candidate {name!r}")


def build_candidate_L_bboxcov(*, x, y, case, S, node_idx, net_idx, n_active, tau):
    """S4g-bboxcov (sec 3.3.4 / phase-1 spec sec 5.3), assumption (A) tau_b := tau.
    cov_k(e) = area(softbbox(e) inter R_k) / area(softbbox(e)); L = sum_e sum_k
    cov_k(e)*(1-q_{e,k}) = sum_e sum_k cov_k(e)*exp(S_{e,k}).

    "net e's skeleton" (phase-1 spec sec 5.3) is built from the same
    deduplicated per-(net,node) occurrences `node_idx`/`net_idx` that S1/S2
    already use for q_{e,k} (ioplace/ops/io_term.py's NetCsr convention is
    node-granularity, not per-pin -- q_{e,k} = 1 - prod_i(1-p_{i,k}) is a
    product over distinct *nodes* i in net e), so the soft bbox is built from
    plain node positions (no per-pin offset within a cell) for consistency
    with every other candidate in this file, not because pin offsets would be
    wrong in principle -- this is folded into assumption (A).

    Net-chunked for the same reason as S4e (build_candidate_L's S4e branch):
    the padded (E,maxdeg) WA-min/max tensors alone are ~1.6GB each at
    bigblue4's ~2.07M-active-net scale (maxdeg up to ignore_net_degree-1==99),
    with several such temporaries live per _wa_bbox call -- easily >8GB
    unchunked, well past what a shared 24GB GPU can spare here. `net_idx` is
    sorted ascending by construction (see _build_padded_pins), so each net-id
    chunk's occurrence-row span is a contiguous slice, found via cumulative
    per-net counts."""
    dev = x.device
    K = case["k"]
    rects = case["rects_t"]        # (R,4)
    r2k = case["r2k_t"]            # (R,)

    _, cnt = torch.unique_consecutive(net_idx, return_counts=True)
    maxdeg = int(cnt.max().item())
    starts = torch.zeros_like(cnt)
    starts[1:] = torch.cumsum(cnt, 0)[:-1]
    ends = starts + cnt

    net_chunk = max(1, _S4G_CHUNK_BUDGET // max(maxdeg, 1))
    li_parts = []
    for elo in range(0, n_active, net_chunk):
        ehi = min(n_active, elo + net_chunk)
        rlo, rhi = int(starts[elo].item()), int(ends[ehi - 1].item())
        node_idx_c = node_idx[rlo:rhi]
        net_idx_c = net_idx[rlo:rhi] - elo
        n_c = ehi - elo
        px_c = x[node_idx_c]
        py_c = y[node_idx_c]
        xpad, ypad, valid = _build_padded_pins(px_c, py_c, node_idx_c, net_idx_c, n_c, dev)
        xmin, xmax, ymin, ymax = _wa_bbox(xpad, ypad, valid, tau)
        bbox_area = ((xmax - xmin) * (ymax - ymin)).clamp_min(1e-6)
        ow = (torch.minimum(xmax.unsqueeze(1), rects[:, 2].unsqueeze(0))
              - torch.maximum(xmin.unsqueeze(1), rects[:, 0].unsqueeze(0))).clamp_min(0.0)
        oh = (torch.minimum(ymax.unsqueeze(1), rects[:, 3].unsqueeze(0))
              - torch.maximum(ymin.unsqueeze(1), rects[:, 1].unsqueeze(0))).clamp_min(0.0)
        overlap_er_c = ow * oh                                  # (chunk,R)
        overlap_ek_c = torch.zeros((n_c, K), dtype=torch.float64, device=dev)
        overlap_ek_c.index_add_(1, r2k, overlap_er_c)
        cov_c = overlap_ek_c / bbox_area.unsqueeze(1)
        li_parts.append((cov_c * torch.exp(S[elo:ehi])).sum(dim=1))
    li = torch.cat(li_parts)
    return li.sum(), dict(li=li.detach())


# ---------------------------------------------------------------------------
# per-net topology classification (sec 3.4.5) -- temporary self-built (see
# module docstring)
# ---------------------------------------------------------------------------
def _touched_lists(rg, nl, node_x, node_y, net_ids, K):
    px, py = pin_positions(nl, node_x, node_y)
    pin_rid = rg.region_of_points(px, py).astype(np.int64)
    pin2net = nl.pin2net.astype(np.int64)
    bm = np.zeros(nl.num_nets, dtype=np.uint64)
    np.bitwise_or.at(bm, pin2net, np.left_shift(np.uint64(1), pin_rid.astype(np.uint64)))
    bm_active = bm[net_ids]
    out = []
    for v in bm_active.tolist():
        out.append([b for b in range(K) if (v >> b) & 1])
    return out


def _home_branch_count(touched, home, D):
    """MST over `touched` (weighted by D) restricted to graph-adjacent (D==1)
    pairs cannot fail per the module docstring's exactness argument (only
    called for Lambda>=3, FT==0 nets) -- Prim's algorithm, terminal-only."""
    terms = list(touched)
    n = len(terms)
    if n < 2:
        return 0
    in_tree = [False] * n
    home_i = terms.index(home)
    in_tree[home_i] = True
    best_cost = [int(D[home, terms[j]]) for j in range(n)]
    best_from = [home_i] * n
    degree = {t: 0 for t in terms}
    for _ in range(n - 1):
        j = min((j for j in range(n) if not in_tree[j]), key=lambda j: (best_cost[j], j))
        u = terms[best_from[j]]
        v = terms[j]
        degree[u] += 1
        degree[v] += 1
        in_tree[j] = True
        for k in range(n):
            if not in_tree[k]:
                c = int(D[terms[j], terms[k]])
                if c < best_cost[k]:
                    best_cost[k] = c
                    best_from[k] = j
    return degree[home]


def classify_topology(Lam, FT, touched_lists, home_arr, D):
    n = len(Lam)
    cls = [None] * n
    for i in range(n):
        lam, ft = int(Lam[i]), int(FT[i])
        if lam <= 1:
            cls[i] = "trivial"
        elif ft >= 1:
            cls[i] = "true-FT"
        elif lam == 2:
            cls[i] = "adjacent"
        else:
            bc = _home_branch_count(touched_lists[i], int(home_arr[i]), D)
            cls[i] = "chain" if bc == 1 else "star"
    return cls


def transition_matrix(before_cls, after_cls):
    m = [[0] * 5 for _ in range(5)]
    for b, a in zip(before_cls, after_cls):
        m[TOPO_IDX[b]][TOPO_IDX[a]] += 1
    return m


# ---------------------------------------------------------------------------
# free-area util (sec 3.4.3's Delta_util, P6 convention)
# ---------------------------------------------------------------------------
def free_area_util_max_over_mean(nl, node_x, node_y, rg, region_area, free_area, k):
    nm = nl.num_movable
    cx = node_x[:nm] + nl.node_size_x[:nm] / 2.0
    cy = node_y[:nm] + nl.node_size_y[:nm] / 2.0
    rid = rg.region_of_points(cx, cy)
    area = nl.node_size_x[:nm] * nl.node_size_y[:nm]
    acc = np.zeros(k)
    np.add.at(acc, rid, area)
    u = acc / np.maximum(free_area, 1e-9)
    return float(u.max() / u.mean())


# ---------------------------------------------------------------------------
# per-state "before" bundle
# ---------------------------------------------------------------------------
def compute_before(case, st):
    nl, rg = case["nl"], case["rg"]
    node_x, node_y = st["node_x"], st["node_y"]
    ev = evaluate_gpu(nl, node_x, node_y, rg, device=case["dev"])
    util = free_area_util_max_over_mean(nl, node_x, node_y, rg, case["region_area"],
                                        case["free_area"], case["k"])

    net_ids = case["csr"].net_ids
    Lam = ev.per_net_lambda[net_ids]
    ST = ev.per_net_steiner[net_ids]
    FT_rg = ST - np.maximum(Lam - 1, 0)
    home = ev.per_net_home[net_ids]
    touched = _touched_lists(rg, nl, node_x, node_y, net_ids, case["k"])
    cls = classify_topology(Lam, FT_rg, touched, home, case["D"])

    return dict(ev=ev, util=util, Lam=Lam, ST=ST, FT_rg=FT_rg, home=home,
               touched=touched, cls=cls, net_ids=net_ids)


# ---------------------------------------------------------------------------
# gradients (I, F_candidate) at a state
# ---------------------------------------------------------------------------
def _grad_via_fresh_forward(case, st, tau, nm, build_L):
    """Build a brand-new x,y leaf + a brand-new S1/S2 forward (_soft_S_q),
    call `build_L(x,y,S,q,lam_soft,node_idx,net_idx,n_active)` -> (L, diag),
    then backward with retain_graph=False (the default) so the *entire*
    graph for this candidate -- shared S1/S2 portion included -- is freed
    the moment this function returns.

    Why not share one S1/S2 forward across all 8 candidates (I + 7 F's) with
    retain_graph=True instead, the way probe_ft_surrogate_soft.py (P0) does
    for its 13 candidates: that works fine at adaptec1's ~217k active-net
    scale, but at bigblue4's ~2.07M active nets it OOMs a 24GB GPU --
    retain_graph=True doesn't retain only the *shared* upstream graph, it's
    a graph-global flag that keeps every candidate's own private forward
    intermediates (S4b's (E,K) N/M/R tensors, S4e's chunked (chunk,K,A)
    tensors, S4g's (E,maxdeg) padded-pin tensors) alive for the rest of the
    loop too, since nothing forces Python to promptly collect the resulting
    reference cycles in the autograd graph. Recomputing the cheap S1/S2
    forward once per candidate (it is not the dominant per-state cost --
    each cell's own evaluate_gpu() call is) trades a small amount of
    redundant compute for a hard memory-safety guarantee that scales to
    bigblue4 without a manual gc.collect()/empty_cache() dance."""
    x = torch.as_tensor(st["node_x"], dtype=torch.float64, device=case["dev"]).clone().requires_grad_(True)
    y = torch.as_tensor(st["node_y"], dtype=torch.float64, device=case["dev"]).clone().requires_grad_(True)
    S, q, node_idx, net_idx, n_active = _soft_S_q(x, y, case, tau)
    lam_soft = q.sum(dim=1)
    L, diag = build_L(x, y, S, q, lam_soft, node_idx, net_idx, n_active)
    gx, gy = torch.autograd.grad(L, [x, y])
    return torch.cat([gx[:nm], gy[:nm]]), diag


def compute_gradients(case, st, before, tau):
    nl = case["nl"]
    nm = nl.num_movable
    dev = case["dev"]

    home_e = torch.as_tensor(before["home"], dtype=torch.int64, device=dev)
    Lam_hard = torch.as_tensor(before["Lam"].astype(np.float64), dtype=torch.float64, device=dev)
    D_home = case["D_t"][home_e]
    ecc_home = case["ecc_max_t"][home_e]

    def io_build(x, y, S, q, lam_soft, node_idx, net_idx, n_active):
        return torch.relu(lam_soft - 1.0).sum(), None

    I, _ = _grad_via_fresh_forward(case, st, tau, nm, io_build)
    grad_l1_io = float(I.abs().sum())
    if os.environ.get("P0B_DEBUG_MEM"):
        gc.collect(); torch.cuda.empty_cache()
        print(f"[mem] after I: allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
             f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    grads = {}
    diags = {}
    failed_candidates = {}
    for name in FT_CANDIDATES:
        def cand_build(x, y, S, q, lam_soft, node_idx, net_idx, n_active, name=name):
            if name == "S4g-bboxcov":
                return build_candidate_L_bboxcov(x=x, y=y, case=case, S=S, node_idx=node_idx,
                                                 net_idx=net_idx, n_active=n_active, tau=tau)
            return build_candidate_L(name, S=S, q=q, D_home=D_home, ecc_home=ecc_home,
                                     lam_soft=lam_soft, Lam_hard=Lam_hard, case=case,
                                     home_e=home_e, nl=nl, node_idx=node_idx, net_idx=net_idx,
                                     n_active=n_active, x=x, y=y, tau=tau)
        try:
            F, diag = _grad_via_fresh_forward(case, st, tau, nm, cand_build)
            grads[name] = F
            diags[name] = diag
        except Exception as exc:  # noqa: BLE001 -- per-candidate OOM (bigblue4
            # scale, see run_state's matching state-level catch's comment for
            # why this can't just be fixed by chunking harder): record which
            # candidate failed and let its siblings that DO fit (empirically
            # S4a/S4b/IO-only) still produce real cells, rather than losing
            # the whole state's worth of data to one memory-hungry candidate.
            failed_candidates[name] = repr(exc)
            print(f"[probe_p0b] candidate {name} FAILED during gradient computation "
                 f"(state setup): {exc!r}")
        if os.environ.get("P0B_DEBUG_MEM"):
            print(f"[mem] after {name}: allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                 f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
        # retain_graph=False frees each candidate's graph via normal autograd
        # bookkeeping, but the grad_fn nodes involved form reference cycles
        # (a well-known PyTorch gotcha) that plain refcounting can't collect
        # promptly -- at bigblue4's ~2.07M-active-net scale (S4g's own
        # (E,maxdeg) padded-pin tensors alone are ~4-5GB), leaving that to
        # Python's periodic cyclic GC OOM'd a 24GB GPU well before the 8th
        # candidate. Forcing collection once per candidate is the standard
        # fix for this pattern.
        gc.collect()
        torch.cuda.empty_cache()

    return dict(I=I, grads=grads, diags=diags, nm=nm, grad_l1_io=grad_l1_io,
               failed_candidates=failed_candidates)


# ---------------------------------------------------------------------------
# direction construction (sec 3.4.2) + one cell's after-metrics
# ---------------------------------------------------------------------------
def _mass_diag(li_tensor, FT_rg_active):
    li = li_tensor.detach().cpu().numpy()
    total = float(li.sum())
    ft_true = FT_rg_active
    mass_on_ft0 = float(li[ft_true == 0].sum() / total) if total > 0 else 0.0
    buckets = {}
    lam = None
    return total, mass_on_ft0, li


def process_cell(case, st, before, gradbundle, candidate, eta, family, tau,
                 tie_break="primary"):
    dev = case["dev"]
    nl, rg = case["nl"], case["rg"]
    nm = gradbundle["nm"]
    L_R = case["L_R"]
    I = gradbundle["I"]

    is_io_only = candidate == "IO-only"
    F = None if is_io_only else gradbundle["grads"][candidate]
    grad_l1_ft = None if is_io_only else float(F.abs().sum())

    if is_io_only:
        kappa = 0.0
        kappa_clamp_active = False
        C = I
    else:
        kappa = derive_kappa_ft(F_FIXED, gradbundle["grad_l1_io"], grad_l1_ft)
        kappa_clamp_active = bool(kappa >= 100.0 - 1e-9)
        C = I + kappa * F

    if family == "merged_l1":
        denom = C.abs().sum().clamp_min(1e-300)
        delta = -eta * L_R * (2 * nm) * C / denom
    elif family == "io_anchored_l1":
        alpha = eta * L_R * (2 * nm) / I.abs().sum().clamp_min(1e-300)
        delta = -alpha * C
    elif family == "full_objective":
        g_wld = st["g_wl_density"]
        n_all = st["n_all"]
        g_wld_mv = torch.cat([g_wld[:nm], g_wld[n_all:n_all + nm]])
        g_full = g_wld_mv + case_state_lambda_io(st) * C
        denom = g_full.abs().sum().clamp_min(1e-300)
        delta = -eta * L_R * (2 * nm) * g_full / denom
    else:
        raise ValueError(family)

    dx = delta[:nm].detach().cpu().numpy()
    dy = delta[nm:].detach().cpu().numpy()

    node_x2 = st["node_x"].copy()
    node_y2 = st["node_y"].copy()
    xl, yl, xh, yh = case["die"]
    node_x2[:nm] = np.clip(node_x2[:nm] + dx, xl, xh)
    node_y2[:nm] = np.clip(node_y2[:nm] + dy, yl, yh)

    ev2 = evaluate_gpu(nl, node_x2, node_y2, rg, device=dev)
    util2 = free_area_util_max_over_mean(nl, node_x2, node_y2, rg, case["region_area"],
                                         case["free_area"], case["k"])

    net_ids = before["net_ids"]
    Lam2 = ev2.per_net_lambda[net_ids]
    ST2 = ev2.per_net_steiner[net_ids]
    FT2_rg = ST2 - np.maximum(Lam2 - 1, 0)
    home2 = ev2.per_net_home[net_ids]
    touched2 = _touched_lists(rg, nl, node_x2, node_y2, net_ids, case["k"])
    cls2 = classify_topology(Lam2, FT2_rg, touched2, home2, case["D"])
    trans = transition_matrix(before["cls"], cls2)

    ev1 = before["ev"]
    d_ft_mst = ev2.ft_count - ev1.ft_count
    d_io_mst = ev2.io_count - ev1.io_count
    d_ft_rg = ev2.ft_rg - ev1.ft_rg
    d_io_rg = ev2.io_rg - ev1.io_rg
    d_hpwl = ev2.hpwl - ev1.hpwl
    d_util = util2 - before["util"]

    def pct(d, base):
        return None if base == 0 else 100.0 * d / base

    row = dict(
        state=st["tag"], tau_rel=st["tau_rel"], candidate=candidate, eta=eta, family=family,
        tie_break=tie_break,
        d_ft_mst=int(d_ft_mst), d_io_mst=int(d_io_mst), d_ft_rg=int(d_ft_rg),
        d_io_rg=int(d_io_rg), d_hpwl=float(d_hpwl), d_util=float(d_util),
        d_ft_mst_pct=pct(d_ft_mst, ev1.ft_count), d_io_mst_pct=pct(d_io_mst, ev1.io_count),
        d_ft_rg_pct=pct(d_ft_rg, ev1.ft_rg), d_io_rg_pct=pct(d_io_rg, ev1.io_rg),
        d_hpwl_pct=pct(d_hpwl, ev1.hpwl), d_util_pct=pct(d_util, before["util"]),
        grad_l1_io=gradbundle["grad_l1_io"], grad_l1_ft=grad_l1_ft, kappa=kappa,
        kappa_clamp_active=kappa_clamp_active,
        cos_io_ft=(None if is_io_only else
                  float(torch.dot(I, F) / (I.norm() * F.norm()).clamp_min(1e-300))),
        mass_on_ft0=None, bucket_share_ratio=None,
        topology_transitions=trans,
    )

    if not is_io_only:
        li = gradbundle["diags"][candidate]["li"].detach().cpu().numpy()
        total = float(li.sum())
        row["mass_on_ft0"] = (float(li[before["FT_rg"] == 0].sum() / total) if total > 0 else 0.0)
        bucket_ratio = {}
        for label, mask in (("1", before["Lam"] == 1), ("2", before["Lam"] == 2),
                            ("3", before["Lam"] == 3), ("4+", before["Lam"] >= 4)):
            true_sum = float(before["FT_rg"][mask].clip(min=0).sum())
            surr_sum = float(li[mask].sum()) if total > 0 else 0.0
            if true_sum > 0:
                bucket_ratio[label] = surr_sum / true_sum
            else:
                bucket_ratio[label] = None
        row["bucket_share_ratio"] = bucket_ratio

    return row


def case_state_lambda_io(st):
    """lambda_io for the full_objective family (sec 3.4.2 M3 row): recomputed
    from the snapshot's own recorded of/tau/gamma/ratio_ema via schedules.py,
    holding this cell's own kappa/Cmax fixed at the snapshot's recorded value
    (sec 3.4.2: "lambda_io 依 5.2 由 snapshot 記錄的 of/tau/gamma/ratio_ema
    與本次 kappa 的 Cmax 重算") -- simplified here to reuse the snapshot's own
    recorded lambda_io directly, since re-deriving rho_io(of)*ramp(iteration)
    from scratch needs the iteration/it_activate state this offline probe does
    not reconstruct; the snapshot's lambda_io already reflects the schedule at
    that exact point in the real trajectory it was pulled from."""
    return float(st.get("lambda_io", 0.0))


# ---------------------------------------------------------------------------
# RANDOM x8 (sec 3.3.5): family-independent, magnitude == the merged_l1
# IO-only reference arm's magnitude at this (state, eta) -- one set of 8
# directions shared by all three families' judging/reporting.
# ---------------------------------------------------------------------------
def process_random_direction(case, st, before, nm, eta, seed):
    dev = case["dev"]
    nl, rg = case["nl"], case["rg"]
    L_R = case["L_R"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(2 * nm, generator=gen, dtype=torch.float64)
    delta = -eta * L_R * (2 * nm) * r / r.abs().sum().clamp_min(1e-300)
    dx = delta[:nm].numpy()
    dy = delta[nm:].numpy()

    node_x2 = st["node_x"].copy()
    node_y2 = st["node_y"].copy()
    xl, yl, xh, yh = case["die"]
    node_x2[:nm] = np.clip(node_x2[:nm] + dx, xl, xh)
    node_y2[:nm] = np.clip(node_y2[:nm] + dy, yl, yh)

    ev2 = evaluate_gpu(nl, node_x2, node_y2, rg, device=dev)
    util2 = free_area_util_max_over_mean(nl, node_x2, node_y2, rg, case["region_area"],
                                         case["free_area"], case["k"])
    ev1 = before["ev"]
    return dict(
        state=st["tag"], tau_rel=st["tau_rel"], seed=seed, eta=eta,
        d_ft_mst=int(ev2.ft_count - ev1.ft_count), d_io_mst=int(ev2.io_count - ev1.io_count),
        d_ft_rg=int(ev2.ft_rg - ev1.ft_rg), d_io_rg=int(ev2.io_rg - ev1.io_rg),
        d_hpwl=float(ev2.hpwl - ev1.hpwl), d_util=float(util2 - before["util"]),
        kind="random",
    )


# ---------------------------------------------------------------------------
# case registry + state-list builders (sec 3.4.1)
# ---------------------------------------------------------------------------
CASE_KEYS = {
    "a1_k16_grid": (ADAPTEC1_CFG, 16, "grid"),
    "a1_k32_grid": (ADAPTEC1_CFG, 32, "grid"),
    "a1_k16_slicing": (ADAPTEC1_CFG, 16, "slicing"),
    "bb4_k16_grid": (BIGBLUE4_CFG, 16, "grid"),
}
_CASE_CACHE = {}


def get_case(case_key, dev=DEV):
    if case_key not in _CASE_CACHE:
        cfg, k, rtype = CASE_KEYS[case_key]
        _CASE_CACHE[case_key] = load_region_case(cfg, k, rtype, dev=dev)
    return _CASE_CACHE[case_key]


def snapshot_state_specs():
    """sec 3.4.1(a): 12 trajectory snapshots (flat/m2best x 6 iterations),
    adaptec1 k16 grid. tau_rel is read directly from the npz (T0-a stores it,
    already derived via tau_rel_from_overflow at snapshot time)."""
    specs = []
    for tag in ("flat", "m2best"):
        for it in SNAPSHOT_ITERS:
            npz = f"results/m3/snapshots/{tag}/it{it:04d}.npz"
            specs.append(dict(tag=f"{tag}_it{it:04d}", npz=npz, case_key="a1_k16_grid",
                              native_tau_rel=None))  # resolved from npz by load_state
    return specs


def _native_tau_rel_from_json(json_relpath, of_end_fallback=0.07):
    d = json.load(open(os.path.join(REPO, json_relpath)))
    traj = d.get("trajectory")
    if not traj:
        raise ValueError(f"{json_relpath}: no trajectory (flat-mode run?) -- "
                         "use the assumption-(B) of_end-based fallback instead")
    of = traj[-1]["overflow"]
    return tau_rel_from_overflow(of, d.get("tau_hi", 0.30), d.get("tau_lo", 0.03),
                                 d.get("of_on", 0.90), d.get("of_end", of_end_fallback))


FINAL_PLACEMENTS = (
    dict(tag="adaptec1_A0_k32_grid", npz="results/m2/ablation/adaptec1_A0_k32_grid.json.npz",
        json=None, case_key="a1_k32_grid", flat=True),
    dict(tag="adaptec1_A2_k32_grid", npz="results/m2/ablation/adaptec1_A2_k32_grid.json.npz",
        json="results/m2/ablation/adaptec1_A2_k32_grid.json", case_key="a1_k32_grid", flat=False),
    dict(tag="adaptec1_A0_k16_slicing", npz="results/m2/ablation/adaptec1_A0_k16_slicing.json.npz",
        json=None, case_key="a1_k16_slicing", flat=True),
    dict(tag="adaptec1_A2_k16_slicing", npz="results/m2/ablation/adaptec1_A2_k16_slicing.json.npz",
        json="results/m2/ablation/adaptec1_A2_k16_slicing.json", case_key="a1_k16_slicing", flat=False),
)

BIGBLUE4_PLACEMENTS = (
    dict(tag="bigblue4_A0_k16_grid", npz="results/m2/ablation/bigblue4_A0_k16_grid.json.npz",
        json=None, case_key="bb4_k16_grid", flat=True),
    dict(tag="bigblue4_A2_k16_grid", npz="results/m2/ablation/bigblue4_A2_k16_grid.json.npz",
        json="results/m2/ablation/bigblue4_A2_k16_grid.json", case_key="bb4_k16_grid", flat=False),
)


def final_8_state_specs():
    """sec 3.4.1(b): 4 final placements x {native tau_rel, forced 0.30}."""
    specs = []
    for p in FINAL_PLACEMENTS:
        if p["flat"]:
            # assumption (B): A0 (flat-mode) runs record no overflow; the GP
            # phase terminates at of == of_end (DREAMPlace's own
            # stop_overflow, verified == schedules.py's of_end convention),
            # so tau_rel_from_overflow(of_end,...,of_end=of_end) == tau_lo
            # == 0.03 exactly (frac == 0 identically), not an arbitrary guess.
            native = 0.03
        else:
            native = _native_tau_rel_from_json(p["json"])
        specs.append(dict(tag=p["tag"] + "_native", npz=p["npz"], case_key=p["case_key"],
                          native_tau_rel=native))
        specs.append(dict(tag=p["tag"] + "_tau030", npz=p["npz"], case_key=p["case_key"],
                          forced_tau_rel=0.30))
    return specs


def bigblue4_state_specs():
    """sec 3.4.1(c): 2 final placements, native tau_rel only."""
    specs = []
    for p in BIGBLUE4_PLACEMENTS:
        native = 0.03 if p["flat"] else _native_tau_rel_from_json(p["json"])
        specs.append(dict(tag=p["tag"] + "_native", npz=p["npz"], case_key=p["case_key"],
                          native_tau_rel=native))
    return specs


def all_20_state_specs():
    return snapshot_state_specs() + final_8_state_specs()


# ---------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------
def _ckpt_append(row):
    os.makedirs(os.path.dirname(CKPT_JSONL), exist_ok=True)
    with open(CKPT_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _ckpt_read_all():
    if not os.path.exists(CKPT_JSONL):
        return []
    rows = []
    with open(CKPT_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _done_keys(existing_rows):
    """Set of (kind, state, candidate/seed, eta, family, tie_break) already
    checkpointed, for resume/skip."""
    keys = set()
    for r in existing_rows:
        if r.get("kind") == "random":
            keys.add(("random", r["state"], r["seed"], r["eta"]))
        else:
            keys.add(("cell", r["state"], r["candidate"], r["eta"], r["family"], r["tie_break"]))
    return keys


# ---------------------------------------------------------------------------
# run one state fully (all candidates x its applicable etas/families), plus
# its 8 random directions at each eta -- shares one gradient computation.
# ---------------------------------------------------------------------------
def run_state(spec, *, families, etas, tie_break="primary", only_candidates=None,
             skip_keys=None, dev=DEV, log=print):
    skip_keys = skip_keys or set()
    case = get_case(spec["case_key"], dev=dev)
    st = load_state(spec, dev=dev)
    tau = st["tau_rel"] * case["L_R"]
    candidates = only_candidates or CANDIDATES

    try:
        before = compute_before(case, st)
        if os.environ.get("P0B_DEBUG_MEM"):
            print(f"[mem] after compute_before: allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                 f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
        if tie_break == "primary":
            case["_tie_break"] = "primary"
        else:
            case["_tie_break"] = "reverse"
        gradbundle = compute_gradients(case, st, before, tau)
        case["_tie_break"] = "primary"  # reset (tie_break only affects S4e; recomputed below)
        if torch.cuda.is_available():
            # defensive: return whatever the caching allocator can free now
            # that every candidate's own (retain_graph=False) forward/
            # backward graph is done, before the per-cell evaluate_gpu()
            # loop below runs.
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        # sec 3.4.0: "任何一格的失敗(OOM/NaN/evaluator 例外)必須寫進 JSON 的
        # failed_cells...不得靜默略過" -- this is exactly that case for a
        # whole state at once: the state-level setup (S1/S2 forward+backward
        # for I plus all 7 F candidates, sec 3.4.2) OOM'd before any
        # per-(candidate,eta,family) cell could be attempted (observed on
        # bigblue4's ~2.07M active nets: each candidate's *own* transient
        # forward-pass peak -- before its own backward frees it -- is
        # ~15-18GB on a 24GB GPU even after K-chunking the S1/S2 forward and
        # net-chunking S4e/S4g's own big tensors; a real fix needs a custom
        # chunked-backward torch.autograd.Function per candidate, matching
        # ioplace/ops/io_term.py's _IoFn -- that is T2's job, not P0b's).
        # Every (candidate, eta, family, tie_break) cell and every random
        # direction this state would have produced is recorded as failed
        # rather than silently dropped or crashing the whole run.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_failed = 0
        for eta in etas:
            for seed in range(N_RANDOM):
                key = ("random", spec["tag"], seed, eta)
                if key not in skip_keys:
                    _ckpt_append(dict(kind="failed", state=spec["tag"], candidate=None,
                                      seed=seed, eta=eta, family=None, tie_break=tie_break,
                                      reason=f"state-level setup failed: {exc!r}"))
                    n_failed += 1
            for family in families:
                # sec 3.4.0: full_objective (M3, confirmatory) is frozen at
                # eta=0.01 only -- 12 snapshot x 8 arms x 1 eta = 96 cells,
                # not x3 etas. Same restriction applied in the success path
                # below.
                if family == "full_objective" and eta != 0.01:
                    continue
                for cand in candidates:
                    if tie_break != "primary" and cand != "S4e-union-detach":
                        continue
                    key = ("cell", spec["tag"], cand, eta, family, tie_break)
                    if key not in skip_keys:
                        _ckpt_append(dict(kind="failed", state=spec["tag"], candidate=cand,
                                          eta=eta, family=family, tie_break=tie_break,
                                          reason=f"state-level setup failed: {exc!r}"))
                        n_failed += 1
        log(f"[probe_p0b] state={spec['tag']} SETUP FAILED ({exc!r}); recorded {n_failed} "
           "failed_cells, skipping this state")
        return n_failed

    nm = gradbundle["nm"]
    n_written = 0
    for eta in etas:
        for seed in range(N_RANDOM):
            key = ("random", spec["tag"], seed, eta)
            if key in skip_keys:
                continue
            row = process_random_direction(case, st, before, nm, eta, seed)
            _ckpt_append(row)
            n_written += 1
        for family in families:
            if family == "full_objective" and st["g_wl_density"] is None:
                continue
            # sec 3.4.0: full_objective (M3, confirmatory) is frozen at
            # eta=0.01 only -- 12 snapshot x 8 arms x 1 eta = 96 cells, not
            # x3 etas (the random-band contribution for M3 -- 12 x 1 x 8 = 96
            # -- is likewise eta=0.01-only, sec 3.4.7's table; the random
            # directions themselves are family-independent so this doesn't
            # skip anything in the `for seed in range(N_RANDOM)` block above,
            # only gates which etas get an M3 *candidate* cell).
            if family == "full_objective" and eta != 0.01:
                continue
            for cand in candidates:
                if tie_break != "primary" and cand != "S4e-union-detach":
                    continue
                key = ("cell", spec["tag"], cand, eta, family, tie_break)
                if key in skip_keys:
                    continue
                case["_tie_break"] = tie_break
                try:
                    if cand in gradbundle.get("failed_candidates", {}):
                        raise RuntimeError(
                            f"gradient unavailable: {gradbundle['failed_candidates'][cand]}")
                    row = process_cell(case, st, before, gradbundle, cand, eta, family, tau,
                                       tie_break=tie_break)
                    row["kind"] = "cell"
                    if not all(math.isfinite(v) for v in
                              (row["d_hpwl"], row["d_util"]) if v is not None):
                        raise FloatingPointError("non-finite delta")
                    _ckpt_append(row)
                except Exception as exc:  # noqa: BLE001 -- sec 3.4.0: must not be silently skipped
                    _ckpt_append(dict(kind="failed", state=spec["tag"], candidate=cand, eta=eta,
                                      family=family, tie_break=tie_break, reason=repr(exc)))
                    log(f"[probe_p0b] FAILED {spec['tag']} {cand} eta={eta} family={family} "
                       f"tie={tie_break}: {exc!r}")
                finally:
                    case["_tie_break"] = "primary"
                n_written += 1
    log(f"[probe_p0b] state={spec['tag']} tau_rel={st['tau_rel']:.4f} wrote {n_written} rows")
    return n_written


# ---------------------------------------------------------------------------
# sanity check (task instruction): S4a-star @ adaptec1 A2 k16 tau_rel=0.03
# forward total must equal probe_ft_surrogate_soft.json's S4a-soft same cell.
# ---------------------------------------------------------------------------
def run_sanity_check(dev=DEV):
    ref_path = os.path.join(REPO, "results/m3/probes/probe_ft_surrogate_soft.json")
    ref = json.load(open(ref_path))
    target = next(c for c in ref["cells"]
                  if c["placement"] == "adaptec1_A2_k16_grid" and c["tau_rel"] == 0.03
                  and c["candidate"] == "S4a-soft")
    ref_total = target["total"]

    case = get_case("a1_k16_grid", dev=dev)
    spec = dict(tag="sanity_A2_k16", npz="results/m2/ablation/adaptec1_A2_k16_grid.json.npz",
               case_key="a1_k16_grid", forced_tau_rel=0.03)
    st = load_state(spec, dev=dev)
    tau = 0.03 * case["L_R"]
    before = compute_before(case, st)
    x = torch.as_tensor(st["node_x"], dtype=torch.float64, device=dev).clone().requires_grad_(True)
    y = torch.as_tensor(st["node_y"], dtype=torch.float64, device=dev).clone().requires_grad_(True)
    S, q, node_idx, net_idx, n_active = _soft_S_q(x, y, case, tau)
    home_e = torch.as_tensor(before["home"], dtype=torch.int64, device=dev)
    D_home = case["D_t"][home_e]
    L, _ = build_candidate_L("S4a-star", S=S, q=q, D_home=D_home, ecc_home=None, lam_soft=None,
                             Lam_hard=None, case=case, home_e=home_e, nl=case["nl"],
                             node_idx=node_idx, net_idx=net_idx, n_active=n_active, x=x, y=y,
                             tau=tau)
    got_total = float(L.detach())
    rel_err = abs(got_total - ref_total) / max(abs(ref_total), 1e-9)
    ok = rel_err < 1e-6
    print(f"[probe_p0b sanity] S4a-star total={got_total!r} ref(S4a-soft)={ref_total!r} "
         f"rel_err={rel_err!r} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)
    return got_total, ref_total


# ---------------------------------------------------------------------------
# assemble: raw checkpoint rows -> sec 3.4.6 schema JSON (tier-1/tier-2)
# ---------------------------------------------------------------------------
METRICS = ("d_ft_mst", "d_ft_rg", "d_io_mst", "d_io_rg", "d_hpwl", "d_util")


def _sd_rand_table(random_rows):
    """(state,eta) -> {metric: (mean, sd, n, sd_floor_applied)}."""
    by_key = {}
    for r in random_rows:
        by_key.setdefault((r["state"], r["eta"]), []).append(r)
    table = {}
    for key, rows in by_key.items():
        entry = {}
        for m in METRICS:
            vals = np.array([r[m] for r in rows], dtype=np.float64)
            mean, sd = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            floor_applied = False
            if sd == 0.0:
                sd = 1.0
                floor_applied = True
            entry[m] = dict(mean=mean, sd=sd, n=len(vals), sd_floor_applied=floor_applied)
        table[key] = entry
    return table


def _tier1_detail(cell, io_only_cell, sd_table_entry):
    if io_only_cell is None or sd_table_entry is None:
        return None
    def sd(m):
        return sd_table_entry[m]["sd"]
    t1a = (cell["d_ft_mst"] < 0) and (cell["d_ft_mst"] <= io_only_cell["d_ft_mst"] - 3 * sd("d_ft_mst"))
    t1b = cell["d_io_mst"] <= io_only_cell["d_io_mst"] + 3 * sd("d_io_mst")
    t1c = cell["d_hpwl"] <= io_only_cell["d_hpwl"] + 3 * sd("d_hpwl")
    t1d = cell["d_util"] <= io_only_cell["d_util"] + 3 * sd("d_util")
    def sign(v):
        return (v > 0) - (v < 0)
    t1e = (sign(cell["d_ft_rg"]) == sign(cell["d_ft_mst"])) and \
         (sign(cell["d_io_rg"]) == sign(cell["d_io_mst"]))
    return dict(t1a=bool(t1a), t1b=bool(t1b), t1c=bool(t1c), t1d=bool(t1d), t1e=bool(t1e))


def _tier2_detail(cell):
    if cell["tau_rel"] > TIER2_TAU_REL_MAX:
        return None
    detail = {}
    if cell["mass_on_ft0"] is not None:
        detail["t2a"] = cell["mass_on_ft0"] <= TIER2_MASS_ON_FT0_MAX
    bsr = cell.get("bucket_share_ratio")
    if bsr:
        lo, hi = TIER2_BUCKET_RATIO_RANGE
        v2 = bsr.get("2")
        detail["t2b"] = (v2 is None) or (lo <= v2 <= hi)
        t2c = True
        for label, v in bsr.items():
            if v is not None and v > TIER2_BUCKET_RATIO_MAX_ASYM:
                t2c = False
        detail["t2c"] = t2c
    return detail


def assemble():
    rows = _ckpt_read_all()
    cells = [r for r in rows if r.get("kind") == "cell"]
    randoms = [r for r in rows if r.get("kind") == "random"]
    failed = [r for r in rows if r.get("kind") == "failed"]

    sd_table = _sd_rand_table(randoms)
    io_only_by_key = {}
    for c in cells:
        if c["candidate"] == "IO-only":
            io_only_by_key[(c["state"], c["eta"], c["family"], c["tie_break"])] = c

    for c in cells:
        sd_entry = sd_table.get((c["state"], c["eta"]))
        io_only = io_only_by_key.get((c["state"], c["eta"], c["family"], c["tie_break"]))
        c["tier1_detail"] = _tier1_detail(c, io_only, sd_entry) if c["candidate"] != "IO-only" else None
        c["tier1_pass"] = bool(c["tier1_detail"] and all(c["tier1_detail"].values())) \
            if c["tier1_detail"] else None
        c["tier2_detail"] = _tier2_detail(c) if c["candidate"] != "IO-only" else None
        c["tier2_pass"] = bool(c["tier2_detail"] and all(c["tier2_detail"].values())) \
            if c["tier2_detail"] else None

    random_bands = []
    for (state, eta), entry in sd_table.items():
        for m, stats in entry.items():
            random_bands.append(dict(state=state, eta=eta, metric=m, **stats))

    per_candidate = []
    for cand in FT_CANDIDATES:
        m1 = [c for c in cells if c["candidate"] == cand and c["family"] == JUDGING_FAMILY
             and c["tie_break"] == "primary"]
        m1_low = [c for c in m1 if c["tau_rel"] <= 0.10 and c["tier1_pass"] is not None]
        m1_high = [c for c in m1 if c["tau_rel"] >= 0.20]
        m3 = [c for c in cells if c["candidate"] == cand and c["family"] == CONFIRMATORY_FAMILY]

        p1_rate = (sum(1 for c in m1_low if c["tier1_pass"]) / len(m1_low)) if m1_low else None
        p1_pass = (p1_rate is not None and p1_rate >= 0.80)
        p2_violations = sum(1 for c in m1_high if c["tier1_detail"]
                            and not (c["tier1_detail"]["t1b"] and c["tier1_detail"]["t1c"]
                                    and c["tier1_detail"]["t1d"]))
        p2_pass = (p2_violations == 0)
        p3_a = [c for c in m3 if c["tier1_detail"]]
        p3_rate = (sum(1 for c in p3_a if c["tier1_detail"]["t1a"]) / len(p3_a)) if p3_a else None
        p3_b_violations = sum(1 for c in p3_a if not (c["tier1_detail"]["t1b"] and c["tier1_detail"]["t1c"]))
        p3_pass = (p3_rate is not None and p3_rate >= 0.60 and p3_b_violations == 0)

        tier1_overall = bool(p1_pass and p2_pass)
        tier2_cells = [c for c in m1 if c["tier2_detail"] is not None]
        tier2_pass = all(c["tier2_pass"] for c in tier2_cells) if tier2_cells else True

        if tier1_overall and tier2_pass:
            verdict = "合格" if p3_pass or not p3_a else "合格"
        elif tier1_overall and not tier2_pass:
            verdict = "受限合格"
        else:
            verdict = "淘汰"
        if tier1_overall and p3_a and not p3_pass:
            verdict = "受限合格" if verdict == "合格" else verdict

        worst = min(m1, key=lambda c: c["d_ft_mst"]) if m1 else None
        per_candidate.append(dict(
            candidate=cand,
            t1_p1_pass_rate_lowtau=p1_rate, t1_p1_pass=p1_pass,
            t1_p2_violations_hightau=p2_violations, t1_p2_pass=p2_pass,
            t1_p3_pass_rate_fullobj=p3_rate, t1_p3_pass=p3_pass,
            tier2_pass=tier2_pass, verdict=verdict,
            worst_cell=(dict(state=worst["state"], eta=worst["eta"], d_ft_mst=worst["d_ft_mst"])
                       if worst else None),
        ))

    ranking = sorted(
        [pc for pc in per_candidate if pc["verdict"] in ("合格", "受限合格")],
        key=lambda pc: (
            np.median([c["d_ft_mst"] for c in cells
                      if c["candidate"] == pc["candidate"] and c["family"] == JUDGING_FAMILY
                      and c["tau_rel"] <= 0.10]) if any(
                c["candidate"] == pc["candidate"] and c["family"] == JUDGING_FAMILY
                and c["tau_rel"] <= 0.10 for c in cells) else 0.0,
            0 if pc["tier2_pass"] else 1,
        ))

    input_relpaths = ["results/m3/probes/probe_ft_surrogate_soft.json"]
    for spec in all_20_state_specs():
        input_relpaths.append(spec["npz"])
    for spec in bigblue4_state_specs():
        input_relpaths.append(spec["npz"])

    out = {
        "probe": "probe_p0b",
        "env": _env_metadata(sys.argv, input_relpaths),
        "preregistration": {
            "doc": "docs/superpowers/specs/2026-08-13-m3-differentiable-ft-design-draft.md",
            "section": "3.4", "doc_sha256": _spec_sha256(),
        },
        "config": {
            "states": [dict(tag=s["tag"]) for s in all_20_state_specs()],
            "candidates": list(CANDIDATES), "etas": list(ETAS), "families": list(FAMILIES),
            "judging_family": JUDGING_FAMILY, "confirmatory_family": CONFIRMATORY_FAMILY,
            "f": F_FIXED, "n_random": N_RANDOM, "random_family_independent": True,
            "scope": {"steps": 1, "nesterov": False, "wl_density": "full_objective family only",
                     "line_search": False},
            "tier1_thresholds": {"t1_p1_min_pass_rate": 0.80, "t1_p1_tau_rel_max": 0.10,
                                 "t1_p2_tau_rel_min": 0.20, "t1_p3_min_pass_rate": 0.60},
            "tier2_thresholds": {"tau_rel_max": TIER2_TAU_REL_MAX,
                                 "mass_on_ft0_max": TIER2_MASS_ON_FT0_MAX,
                                 "bucket_ratio_range": list(TIER2_BUCKET_RATIO_RANGE),
                                 "bucket_ratio_max_asym": TIER2_BUCKET_RATIO_MAX_ASYM},
            "assumptions": {
                "A_tau_b_bboxcov": "tau_b := tau (ambient S1/S2 temperature); not specified in "
                                   "either spec doc; see module docstring assumption (A)",
                "B_A0_native_tau_rel": "flat-mode (A0) final placements have no recorded "
                                      "overflow; assumed of==of_end (DREAMPlace stop_overflow) "
                                      "=> tau_rel==tau_lo==0.03 exactly; see assumption (B)",
                "C_s4e_path_tie_break": "primary tie-break reuses region_graph.next_hop_table's "
                                       "canonical smallest-neighbour-index rule; reverse "
                                       "submatrix uses the mirrored largest-neighbour-index "
                                       "rule; see assumption (C)",
            },
        },
        "failed_cells": failed,
        "cells": cells,
        "random_bands": random_bands,
        "summary": {"per_candidate": per_candidate, "ranking": [r["candidate"] for r in ranking]},
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT_JSON)
    print(f"[probe_p0b] wrote {OUT_JSON} ({len(cells)} cells, {len(random_bands)} random_bands, "
         f"{len(failed)} failed_cells)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True,
                    choices=["sanity", "b", "d", "a", "c", "assemble"],
                    help="b=8 final-placement states (M1/M2), d=bigblue4 spot-check, "
                        "a=12 trajectory-snapshot states (M1/M2/M3), "
                        "c=S4e reverse-tie-break submatrix (20 states), assemble=final JSON")
    ap.add_argument("--states", default=None, help="comma-separated state tags to restrict to")
    ap.add_argument("--etas", default=None, help="comma-separated etas to restrict to")
    ap.add_argument("--candidates", default=None, help="comma-separated candidates to restrict to")
    args = ap.parse_args()

    if args.group == "sanity":
        run_sanity_check()
        return
    if args.group == "assemble":
        assemble()
        return

    etas = [float(e) for e in args.etas.split(",")] if args.etas else list(ETAS)
    cands = args.candidates.split(",") if args.candidates else None
    existing = _ckpt_read_all()
    skip_keys = _done_keys(existing)

    if args.group == "b":
        specs = final_8_state_specs()
        families = ["merged_l1", "io_anchored_l1"]
        tie_break = "primary"
    elif args.group == "d":
        specs = bigblue4_state_specs()
        families = ["merged_l1"]
        etas = [0.01]
        tie_break = "primary"
    elif args.group == "a":
        specs = snapshot_state_specs()
        families = ["merged_l1", "io_anchored_l1", "full_objective"]
        tie_break = "primary"
    elif args.group == "c":
        specs = all_20_state_specs()
        families = ["merged_l1"]
        etas = [0.01]
        cands = ["S4e-union-detach"]
        tie_break = "reverse"
    else:
        raise ValueError(args.group)

    if args.states:
        want = set(args.states.split(","))
        specs = [s for s in specs if s["tag"] in want]

    t0 = time.time()
    for spec in specs:
        run_state(spec, families=families, etas=etas, tie_break=tie_break,
                 only_candidates=cands, skip_keys=skip_keys, dev=DEV)
    print(f"[probe_p0b] group={args.group} states={[s['tag'] for s in specs]} "
         f"etas={etas} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
