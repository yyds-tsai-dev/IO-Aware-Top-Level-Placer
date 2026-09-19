# Router parallelism adjustment

## Decision and scope

The recovery host has 24 physical cores and 48 logical CPUs. At inspection,
the existing tile detailed router requested 16 threads; two global routers
used approximately one CPU each. The FastRoute maze implementation has no
OpenMP parallel loop in `maze.cpp`; FlexDR explicitly runs worker batches
with `MAX_THREADS`. Raising a GRT thread setting does not parallelize that
maze implementation.

Raise newly launched recovery detailed routers to 24 threads, capped by the
process CPU affinity with four logical CPUs reserved for other stages.
Use a blocking, process-safe lock to admit one new detailed router at a time.
This limits additional memory pressure on a host with about 108 GiB available
at inspection. It is an admission limit, not a guarantee that a single large
design fits memory. The already-running 16-thread tile route is outside this
new lock and keeps its progress; temporary overlap is at most 40 requested
DR threads, plus other placement/global-route work.

Only the recovery supervisor changes. Its active parent retains the original
loaded functions, while each subsequent `--dr` child loads the updated script.
The GP, evaluator, routing algorithms, seeds, and full-default detailed-route
iteration policy stay fixed. No live router is restarted. The original
supervisor source is preserved under
`results/route_gp_20260914/recovery_20260915/supervisor_before_parallelism.txt`.

## Review and validation plan

1. Verify CPU affinity caps and lock exclusion/release on an exception.
2. Verify existing detailed-route adapter tests.
3. Run a real full-default small route using the updated child entry point;
   inspect threads, return code, DRC, unrouted nets, and placement identity.
4. Record the policy and supervisor hash separately from the original protocol.

Thread counts are a performance setting, but thread-count invariance of large
route quality is not assumed. The tile WA arm already runs at 16 threads;
its future joint arm will use 24. Report the difference and avoid claiming a
controlled runtime speedup. Large-case wall time and quality remain pending.
The existing background completion notification remains responsible for
resuming the full FLUTE/evaluator validation goal.

## Validation result

11 unit tests passed; live full-default GCD DR passed: threads=24, IO=584, WL=15647870 DBU, DRC=0, unrouted nontrivial nets=0, identity unchanged; output hashes verified. No large-case speedup is claimed. The user subsequently deferred representative DR to stage two and authorized bounded intermediate GRT as stage-one priority.
