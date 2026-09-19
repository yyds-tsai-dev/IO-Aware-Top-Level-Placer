"""Shared native initialization and truthful process-wide thread provenance.

IOPLACE_MTKAHYPAR_THREADS=1 is an explicit mitigation for the installed
1.6.2 wheel's intermittent parallel-coarsening crash, not an allocator fix.
Set it before either partitioning or Rent measurement initializes the runtime.
"""
import os
import operator
import threading

_state = {"module": None, "requested": None, "effective": None, "override": None,
          "first_requested": None}
_lock = threading.Lock()

def _env_override():
    raw = os.environ.get("IOPLACE_MTKAHYPAR_THREADS")
    if raw is None: return None
    try: value = int(raw)
    except ValueError: raise ValueError("IOPLACE_MTKAHYPAR_THREADS must be a positive integer")
    if value <= 0: raise ValueError("IOPLACE_MTKAHYPAR_THREADS must be a positive integer")
    return value

def initialize(mtkahypar, requested_threads):
    requested_threads = operator.index(requested_threads)
    if requested_threads <= 0:
        raise ValueError("threads must be positive")
    override = _env_override()
    effective = override if override is not None else requested_threads
    with _lock:
        if _state["module"] is not None:
            if override is not None and override != _state["effective"]:
                raise RuntimeError("IOPLACE_MTKAHYPAR_THREADS cannot change after initialization")
        else:
            owner = mtkahypar.initialize(effective, print_warnings=False)
            _state.update(module=owner, first_requested=requested_threads, effective=effective)
        _state.update(requested=requested_threads, override=override)
        return _state["module"]

def metadata():
    return {"requested_threads": _state["requested"], "override_threads": _state["override"],
            "effective_native_threads": _state["effective"],
            "first_requested_threads": _state["first_requested"]}
