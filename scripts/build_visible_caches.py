#!/usr/bin/env python3
"""Build and fully verify the visible recovery Bookshelf caches."""
import hashlib, json, os, platform, sys, time
from pathlib import Path
from ioplace.bench.bookshelf_netlist import build_tiled_netlist_cache, verify_against_bookshelf

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "results/recovery_visible_20260906"
ARR = REC / "arrays"
CACHE = REC / "cache"
SHAPES = ("3x3_n2", "1x2_n2", "2x2_n2")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    for case in SHAPES:
        manifest = ARR / case / f"{case}.manifest.json"
        prefix = ARR / case / case
        out = CACHE / case
        t0 = time.time()
        meta = build_tiled_netlist_cache(str(manifest), str(out))
        ver = verify_against_bookshelf(str(out), str(prefix), mode="full")
        record = {"case": case, "manifest": str(manifest), "prefix": str(prefix),
                  "cache": str(out), "schema_version": meta.get("schema_version"),
                  "build_meta": meta, "verify": ver,
                  "source_code_sha256": sha(Path(__file__)),
                  "python": sys.version, "platform": platform.platform(),
                  "elapsed_total_s": time.time() - t0}
        (out / "verification.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        if not ver.get("ok", False):
            raise SystemExit(f"{case}: full verification failed: {ver.get('errors', [])[:3]}")
        print(f"{case}: cache and full verification OK ({ver.get('elapsed_s', 0):.1f}s)", flush=True)

if __name__ == "__main__": main()
