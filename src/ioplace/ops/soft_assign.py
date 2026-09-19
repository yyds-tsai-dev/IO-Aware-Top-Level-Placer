"""S1 soft region assignment (design v2 sec 2.1/2.2/2.5).

All functions are chunk-aware: nothing here ever allocates a full (N,K) tensor
unless the caller explicitly asks for k_hi-k_lo == K.
"""
import numpy as np
import torch


def rect_table(rs):
    rects, r2k = [], []
    for rid, r in enumerate(rs.regions):
        arr = np.asarray(r.rects, dtype=np.float64).reshape(-1, 4)
        rects.append(arr)
        r2k.append(np.full(len(arr), rid, dtype=np.int64))
    return np.concatenate(rects, axis=0), np.concatenate(r2k, axis=0)


def region_sdf_l1(x, y, rects, rect2region, k_lo, k_hi):
    """(N,), (N,), (R,4), (R,) -> (N, k_hi-k_lo) L1 signed distance."""
    c = k_hi - k_lo
    sel = (rect2region >= k_lo) & (rect2region < k_hi)
    sub = rects[sel]                                  # (r,4)
    col = (rect2region[sel] - k_lo)                   # (r,) in [0,c)
    dx = torch.maximum(sub[:, 0][None] - x[:, None], x[:, None] - sub[:, 2][None])
    dy = torch.maximum(sub[:, 1][None] - y[:, None], y[:, None] - sub[:, 3][None])
    d = dx.clamp(min=0) + dy.clamp(min=0) + torch.maximum(dx, dy).clamp(max=0)   # (N,r)
    out = torch.full((x.shape[0], c), float("inf"), dtype=d.dtype, device=d.device)
    out = out.scatter_reduce(1, col[None].expand(x.shape[0], -1), d,
                             reduce="amin", include_self=True)
    return out


def _chunks(K, chunk):
    if chunk is None or chunk >= K:
        return [(0, K)]
    return [(lo, min(lo + chunk, K)) for lo in range(0, K, chunk)]


def softmax_stats(x, y, rects, rect2region, K, tau, chunk=None):
    """Pass 1 of the chunked contract: (m, t, argmax) over ALL K regions."""
    n = x.shape[0]
    dev, dt = x.device, x.dtype
    m = torch.full((n,), -float("inf"), dtype=dt, device=dev)
    am = torch.zeros(n, dtype=torch.int64, device=dev)
    for lo, hi in _chunks(K, chunk):
        z = -region_sdf_l1(x, y, rects, rect2region, lo, hi) / tau
        cm, ci = z.max(dim=1)
        upd = cm > m                       # strict > keeps first-occurrence tie-break
        m = torch.where(upd, cm, m)
        am = torch.where(upd, ci + lo, am)
    t = torch.zeros(n, dtype=dt, device=dev)
    for lo, hi in _chunks(K, chunk):
        z = -region_sdf_l1(x, y, rects, rect2region, lo, hi) / tau
        e = torch.exp(z - m.unsqueeze(1))
        is_am = (am.unsqueeze(1) == torch.arange(lo, hi, device=dev).unsqueeze(0))
        t = t + (e * (~is_am)).sum(dim=1)   # exclude the argmax term -- never s-1
    return m, t, am


def chunk_p_ell(sdf_chunk, m, t, argmax, k_lo, tau, floor=1e-30):
    dev = sdf_chunk.device
    c = sdf_chunk.shape[1]
    z = -sdf_chunk / tau
    e = torch.exp(z - m.unsqueeze(1))
    s = (1.0 + t).unsqueeze(1)
    p = e / s
    is_am = (argmax.unsqueeze(1) == torch.arange(k_lo, k_lo + c, device=dev).unsqueeze(0))
    omp = torch.where(is_am, (t.unsqueeze(1) / s).expand_as(e), (s - e) / s).clamp_min(floor)
    return p, torch.log(omp)


def d_star_from_m(m, tau):
    """min_k d_k  ==  -tau * max_k z_k."""
    return -tau * m
