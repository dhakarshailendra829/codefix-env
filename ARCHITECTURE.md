# Architecture

This document describes CodeFix-Env as it is actually implemented. It replaced an earlier, generic version of this file that described a PostgreSQL/Kubernetes stack that never existed in this codebase — everything below corresponds to real code in this repository.

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
cli.py                           — codefix-server console script (serve/tasks/info)
```

Installable directly from PyPI (`pip install codefix-env`) or from source for development — see `CONTRIBUTING.md`.

## 2. Episode lifecycle

`reset(task_id | difficulty)` loads a `Task` (buggy code, hidden test cases, hints), snapshots it into `CodeFixState`, and returns an initial `CodeFixObservation` with no tests run yet.

`step(action)` dispatches one of: `RUN_TESTS`, `EDIT_LINE`, `INSERT_LINE`, `DELETE_LINE`, `GET_HINT`, `SUBMIT_FIX`, `VIEW_CODE`. Every code-mutating action produces a unified diff (`difflib.unified_diff` against the original buggy code), returned in the observation — this is what lets an agent see exactly what it changed without re-diffing itself.

Termination: `SOLVED` (all tests pass), `SUBMITTED` (explicit submit action), or `MAX_STEPS` (truncation at `min(constructor_max_steps, task.max_steps)`).

## 3. Reward shaping

`RewardPipeline` (`rewards.py`) combines two signals:

- **Rule-based** (`utils/metrics.py: compute_shaped_reward`) — always active. Rewards test-pass-count deltas between the previous and current observation, and penalizes hint usage and step count.
- **Optional neural reward model** (`RewardMLP`, in `utils/reward_model.py`) — a small torch MLP, loaded from a checkpoint path if provided, blended with the rule-based score via `neural_weight` (default 0.3). `RewardMLP` lives in its own module specifically so that importing `codefix_env` does not require importing PyTorch unless the neural component is actually used — this matters for sandboxed worker processes, which re-import the package on every spawn under `multiprocessing`. This component is currently untrained/optional scaffolding — no checkpoint ships with the package. Training it against real trajectory data is a listed next step, not a hidden gap.

Final episode score (`compute_final_score`) is a function of `(tests_passed, tests_total, step_count, hints_used)`.

## 4. Sandbox — see SECURITY.md for the full threat model

Three layers: static AST allow-list → restricted `__builtins__` → OS-level process isolation via `multiprocessing.Process` with `resource.setrlimit` caps on CPU time and file size.

This is **process-level isolation, not container/VM-level isolation**. `SECURITY.md` documents exactly what this does and does not protect against, including two specific hardening attempts (a process-count limit and a virtual-memory cap) that were implemented, tested under concurrent load, found to cause more problems than they solved, and reverted — kept in the record rather than removed.

## 5. Server / session model

The FastAPI app in `server/app.py` is stateless per request; per-episode state lives in `SessionManager` (an in-memory dict, capped at `MAX_SESSIONS`, keyed by `X-Session-ID`). This means:

- **No persistence** — restarting the server drops all active sessions.
- **No horizontal scaling, yet** — sessions are pinned to whichever server instance created them; there is no shared session store (Redis is the planned choice). Adequate for single-instance training or evaluation; a real limitation for a multi-replica deployment.

## 6. What is intentionally not here yet (roadmap, not a hidden gap)

- Docker/container-provider packaging conforming to the OpenEnv specification. The `step()`/`reset()`/`state()` interface already matches in shape, but there is no container provider registered, so the environment is not yet installable via the OpenEnv Hub.
- A trained reward model checkpoint, and a closed-loop RL training result showing a policy actually improving over time against this environment. The current baseline evaluation (`scripts/run_baseline_eval.py`) compares fixed oracle and random policies, which confirms the reward pipeline discriminates skill but does not by itself demonstrate a trainable policy improving.
- A task registry beyond the current 21 hand-written examples. `scripts/mine_tasks.py` mines candidate tasks from real GitHub bug-fix pull requests and is validated against the live API; scaling this to a reviewed set of several hundred tasks is the next step.
- Multi-file task support — every current task is scoped to a single function in a single file.
- Support for a second programming language beyond Python.

Each of these is tracked as a concrete next step; see `README.md`'s Roadmap section.
