# CLAUDE.md

IO-Aware-Top-Level-Placer: research codebase for GPU-accelerated, IO-aware
top-level placement (10M–30M cell scale, DREAMPlace-based, minimising IO
crossings and feed-throughs). Current milestone plan and phase design live in
[docs/superpowers/plans/](docs/superpowers/plans/) and
[docs/superpowers/specs/](docs/superpowers/specs/); the build/runtime
environment is documented in [docs/dev-env.md](docs/dev-env.md).

## Model orchestration (Fable 5 scheduler)

This repo runs a multi-model team so the expensive scheduler budget is spent on
coordination, not bulk work. **Fable 5 (xhigh effort) is the scheduler**, set at
user level in `~/.claude/settings.json` (`model: claude-fable-5[1m]`,
`effortLevel: xhigh`) — picked up by new sessions, not retroactively. The
scheduler plans, decomposes, delegates, and integrates results; it keeps its own
context lean and pushes the actual work down to the specialists.

| Role | Who | Use for |
|------|-----|---------|
| **Scheduler** | **Fable 5** — this main thread | Understand the request, break it into well-scoped units, route each to the cheapest capable executor, stitch the results together, make the final call. No bulk edits or long pytest grinds here. |
| **Deep reasoning** | `deep-reasoner` subagent → **Opus** ([.claude/agents/deep-reasoner.md](.claude/agents/deep-reasoner.md)) | Hard algorithm/architecture work: partitioning policy (hMETIS export, Mt-KaHyPar), IO/feed-through cost design, evaluator numerics (`evaluator_gpu` vs `evaluator_ref` parity), DREAMPlace-integration root-causing, "does this actually improve placement quality?" trade-offs against the current plan. Returns plans and diagnoses, not sprawling edits. |
| **Mechanical execution** | `fast-worker` subagent → **Sonnet** ([.claude/agents/fast-worker.md](.claude/agents/fast-worker.md)) | Well-specified, low-ambiguity work: applying decided edits, running pytest and reporting, renames/moves, boilerplate, doc sync, fact-gathering. No design decisions. |
| **Peer engineer** | **Codex** (OpenAI), via the official `codex@openai-codex` plugin | Independent implementation or adversarial review from a *different model family* — cross-check the scheduler's / Opus's conclusions on high-stakes changes. |

Dispatch rules for the scheduler:

- **Think first, then delegate.** For anything non-trivial, produce a plan
  (yourself or via `deep-reasoner`) before touching code, scoped against the
  current plan and specs under `docs/superpowers/`.
- **Route by task shape:** needs a *decision or a diagnosis* → `deep-reasoner`
  (Opus); *decided and mechanical* → `fast-worker` (Sonnet); want a *dissenting
  second implementation or review* → Codex. When in doubt about cost, prefer
  delegating over doing it in the scheduler thread.
- **Conserve Fable:** hand bulk edits and build/test loops to `fast-worker`;
  reserve the main thread for orchestration and final judgement.
- **Use Codex as a real peer, not a rubber stamp:** prefer its adversarial
  modes on high-stakes changes (evaluator numerics, partition policy, driver
  flow) to catch what a single model family misses. When Codex is unavailable,
  fall back to a second `deep-reasoner` pass with an adversarial framing.
- **Verify before accepting:** a delegated change counts as done only after
  pytest passes. On this H100 NVL host, run `source src/scripts/env.sh`
  from the repo root, then `"$IOPLACE_PYTHON" -m pytest`. The script exports
  `DREAMPLACE_ROOT=/ldaphome/yyds-tsai-dev/DREAMPlace` by default and selects
  its Python 3.12 venv — see [docs/dev-env.md](docs/dev-env.md).
  `-m "not slow"` skips the real-placement integration tests while iterating;
  run the full suite before calling a task complete. Honor `CUDA_VISIBLE_DEVICES` and
  check `nvidia-smi` before GPU work on this shared host.

Invocation:

- **Subagents** — launch `deep-reasoner` / `fast-worker` with the Agent tool;
  each is bound to its model in its `.claude/agents/*.md` frontmatter and takes
  effect from the next session after edits.
- **Codex** — the `codex@openai-codex` plugin (marketplace `openai-codex`, from
  `openai/codex-plugin-cc`) drives the already-authenticated `codex` CLI on
  this host; run `/codex:setup` once per machine to confirm CLI + auth and the
  stop-review gate. Delegate a task, diagnosis, or alternative implementation
  with `/codex:rescue` or the `codex:codex-rescue` subagent; `/codex:review`
  (read-only) and `/codex:adversarial-review` (steerable challenge that
  pressure-tests design choices) are the second-opinion reviews, with
  `/codex:status` / `/codex:result` managing background jobs. Reviews can also
  run through the plugin's shared runtime:
  `node "$CLAUDE_PLUGIN_ROOT/scripts/codex-companion.mjs" review|adversarial-review`
  (prefer `--background` over `--wait` so a slow review does not look frozen;
  poll with `status`/`result`). Reconcile Codex's take with `deep-reasoner`'s
  before acting on either.
  - Every `[mcp_servers.*]` in `~/.codex/config.toml` must set
    `tool_timeout_sec` — a server without one (e.g. `codebase-memory-mcp`) can
    hang a single tool call indefinitely and freeze the whole review in
    `investigating`.

<!-- codex-project-subagents:start -->
## Codex task execution and review

When dispatching implementation, diagnosis, or review in Codex, read
[`.codex/agents/README.md`](.codex/agents/README.md) and the selected role TOML.
Use `fast-worker` (`gpt-5.6-terra`, `low`) for decided implementation tasks;
use `deep-reasoner` (`gpt-6-astra`, `high`) for diagnosis and independent review.
Keep the latter read-only; the parent persists its returned review report.
Use fresh implementers sequentially and review each task for specification
compliance and code quality before advancing. Preserve these role settings
across session resets; pass explicit ownership and graph evidence in briefs.
The current Codex runtime exposes both roles. Select them by `agent_type`;
do not substitute inherited default roles silently.
<!-- codex-project-subagents:end -->
