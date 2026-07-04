# Architecture

This replaces a previous version of this document that was generic
boilerplate (placeholder classes, a PostgreSQL/Kubernetes stack that never
existed in this codebase). Everything below describes the system as it is
actually implemented.

## 1. Component map

```
CodeFixEnvironment (env.py)      — stateful, one episode per instance
   ├── tasks/ (easy/medium/hard) — 21 hand-written Task definitions
   ├── utils/sandbox.py          — isolated code execution
   ├── rewards.py                — RewardPipeline (rule-based + optional MLP)
   └── models.py                 — pydantic schemas (Action/Observation/State)

server/app.py (FastAPI)          — HTTP wrapper, session-per-client via
   └── server/codefix_environment.py   X-Session-ID header, in-memory SessionManager

client.py                        — async HTTP client + sync wrapper for notebooks
cli.py                           — `codefix-server` console script (serve/tasks/info)
```

## 2. Episode lifecycle

`reset(task_id | difficulty)` loads a `Task` (buggy code + hidden test cases +
hints), snapshots it into `CodeFixState`, and returns an initial
`CodeFixObservation` with no tests run yet.

`step(action)` dispatches one of: `RUN_TESTS`, `EDIT_LINE`, `INSERT_LINE`,
`DELETE_LINE`, `GET_HINT`, `SUBMIT_FIX`, `VIEW_CODE`. Every code-mutating
action produces a unified diff (`difflib.unified_diff` against the original
buggy code) that's returned in the observation — this is what lets an agent
see exactly what it changed without re-diffing itself.

Termination: `SOLVED` (all tests pass), `SUBMITTED` (explicit submit action),
or `MAX_STEPS` (truncation — `min(constructor_max_steps, task.max_steps)`).

## 3. Reward shaping

`RewardPipeline` (rewards.py) combines two signals:
- **Rule-based** (`utils/metrics.py: compute_shaped_reward`) — always active.
  Rewards test-pass-count deltas between the previous and current
  observation, and penalizes hint usage and step count.
- **Optional neural reward model** (`RewardMLP`, small torch MLP) — loaded
  from a checkpoint path if provided, blended with the rule-based score via
  `neural_weight` (default 0.3). This is currently untrained/optional
  scaffolding; there is no shipped checkpoint. Documenting this explicitly
  because the README implies a more complete system than currently exists —
  training this MLP against real trajectory data is a listed next step.

Final episode score (`compute_final_score`) is a function of
`(tests_passed, tests_total, step_count, hints_used)`.

## 4. Sandbox — see SECURITY.md for the full threat model

Three layers: static AST allow-list → restricted `__builtins__` → OS process
isolation with `resource.setrlimit` caps (memory/CPU/nproc/fsize). This is
**process-level isolation, not container/VM-level isolation** — SECURITY.md
documents exactly what that does and doesn't protect against, rather than
overclaiming "fully sandboxed."

## 5. Server / session model

FastAPI app in `server/app.py` is stateless per-request; per-episode state
lives in `SessionManager` (in-memory dict, capped at `MAX_SESSIONS`, keyed by
`X-Session-ID`). This means:
- **No persistence** — restarting the server drops all active sessions.
- **No horizontal scaling** — sessions are pinned to whichever server
  instance created them; there's no shared session store (e.g. Redis).
  Fine for single-instance training/eval; a real limitation for a
  multi-replica deployment.

## 6. What's intentionally NOT here yet (roadmap, not a hidden gap)

- Docker/container-provider packaging conforming to the OpenEnv 0.1 spec
  (`step()/reset()/state()` naming matches, but there's no container
  provider registered, so this isn't yet installable via the OpenEnv Hub).
- A trained reward model checkpoint / an actual RL fine-tune result showing
  the loop closes end-to-end.
- Task set is 21 hand-written examples, not mined from real repositories —
  fine for unit-testing the environment mechanics, not yet a benchmark.

Each of these is tracked as a concrete next step rather than left implicit.
