<div align="center">

<h1>CodeFix-Env</h1>

<p>A sandboxed reinforcement learning environment for training and evaluating LLM agents on automated code debugging, and the fast, verifiable feedback loop that kind of training requires.</p>

<p align="center">
  <a href="https://github.com/dhakarshailendra829/codefix-env/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/dhakarshailendra829/codefix-env/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://pypi.org/project/codefix-env/"><img src="https://img.shields.io/pypi/v/codefix-env?style=for-the-badge&color=blue" alt="PyPI version"></a>
  <a href="https://github.com/dhakarshailendra829/codefix-env/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="Apache 2.0 License"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-197%20passing-brightgreen?style=for-the-badge" alt="Tests passing"></a>
</p>

<p align="center">
  <strong>
    <a href="#quick-start">Quick Start</a> ·
    <a href="ARCHITECTURE.md">Architecture</a> ·
    <a href="SECURITY.md">Security</a> ·
    <a href="CONTRIBUTING.md">Contributing</a>
  </strong>
</p>

</div>

---

> **Status: active development.** The environment, sandbox, reward pipeline, and task registry are stable and tested. A trained-policy result and a larger, mined task set are in progress — see [Roadmap](#roadmap).

---

## Table of Contents

- [Why CodeFix-Env?](#why-codefix-env)
- [Install](#install)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [How It Works](#how-it-works)
- [Benchmark](#benchmark)
- [Capabilities](#capabilities)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Citations](#citations)

---

## Why CodeFix-Env?

Training a coding agent to debug software requires more than static examples of correct code. It requires a tight loop: propose a fix, execute it against a real test suite, and receive a signal that reflects genuine progress rather than superficial activity.

We do that because SWE-bench<sup>1</sup> gave coding agents a scalable, realistic evaluation standard, built on full repositories and real GitHub issues. What it does not give is a fast enough loop for on-policy reinforcement learning, where thousands of cheap episodes per training step are typically required.

Most environments used for this today sit at one of two extremes: repository-scale benchmarks that are realistic but slow to sandbox, or ad-hoc code execution with no meaningful isolation at all. CodeFix-Env is built for the space between them.

CodeFix-Env is building that middle layer:

> a sandboxed, densely-rewarded reinforcement learning environment for single-function code repair, fast enough to serve as a rollout source, with a stated threat model rather than an assumed one

We do that by:

- executing every agent action inside a layered sandbox — static AST filtering, restricted builtins, and OS-level process isolation — documented in full in [SECURITY.md](SECURITY.md), including two hardening attempts that failed under load and were reverted rather than left unreported
- computing a dense, per-step reward from hidden test-pass-rate deltas, rather than a single pass/fail signal at episode end
- mining task candidates directly from real, merged bug-fix pull requests on GitHub, so tasks can be traced back to an actual historical defect (`scripts/mine_tasks.py`)
- publishing the environment as an installable, continuously tested package rather than a static research repository

Our goal is to grow this into a task set and training result large enough to serve as a genuine benchmark and training ground for code-repair agents, in the same way SWE-bench did for repository-scale evaluation.

<sup>1</sup> https://arXiv:2607.XXXXX, 2026

---

## Install

```bash
pip install codefix-env
```

For development, or to use the training extras:

```bash
git clone https://github.com/dhakarshailendra829/codefix-env.git
cd codefix-env
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -e ".[dev]"       # add ",llm" for transformers/TRL/accelerate
```

Verify:

```bash
codefix-server info
```

---

## Quick Start

**Local environment** — run episodes directly in Python, no server required:

```python
from codefix_env import CodeFixEnvironment, CodeFixAction, ActionType

env = CodeFixEnvironment()
obs = env.reset(task_id="easy-001-missing-return")

result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
print(f"{result.observation.tests_passed}/{result.observation.tests_total} tests passing")

result = env.step(CodeFixAction(
    action_type=ActionType.EDIT_LINE,
    line_number=3,
    new_content="    return result",
))

result = env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
print("solved" if result.observation.all_tests_pass else "not solved", result.reward)
```

**Server mode** — expose the environment over HTTP for remote or multi-language clients:

```bash
codefix-server serve --port 8000
```

**Task inspection** — list and filter the task registry:

```bash
codefix-server tasks
```

**Baseline evaluation** — run oracle and random policies across the full task set:

```bash
python scripts/run_baseline_eval.py --episodes 3
```

---

## Deployment

The repository includes a `Dockerfile` for running the environment as a standalone service:

```bash
docker build -t codefix-env .
docker run -p 8000:8000 codefix-env
```

Set `CODEFIX_REWARD_MODEL_PATH` if using a trained neural reward checkpoint (see [ARCHITECTURE.md](ARCHITECTURE.md)). No other environment variables are required for the default, rule-based reward configuration.

---

## How It Works

When an episode runs, CodeFix-Env:

1. **Loads** a task — a buggy function, a hidden test suite, and graduated hints
2. **Receives** a structured action from the agent: run tests, edit a line, insert, delete, request a hint, view the code, or submit
3. **Executes** any code-mutating action inside the sandbox, isolated from the host process
4. **Computes** a unified diff against the original code and a dense, shaped reward from the test-pass-rate change
5. **Returns** the updated observation, including test output and the diff, so the agent can decide its next action
6. **Terminates** the episode on a full solve, an explicit submission, or the step budget running out

Full design detail, including the reward formula and the sandbox's layered isolation model, is in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Benchmark

Regenerate these numbers with `python scripts/run_baseline_eval.py --episodes 3`.

<!-- BENCHMARK-START -->

Oracle policy (applies the known-correct fix) versus a random policy, across all 21 tasks:

| Policy | Easy | Medium | Hard |
|---|---|---|---|
| Oracle | 100% | 87.5% | 60% |
| Random | 0% | 0% | 20% |

No trained-policy result yet — this compares two fixed baselines to confirm the reward signal discriminates genuine problem-solving from undirected action. A GRPO training result is in progress; see [Roadmap](#roadmap).

<!-- BENCHMARK-END -->

---

## Capabilities

| | |
|---|---|
| **Sandboxed execution** | Static AST filtering, restricted builtins, and OS-level process isolation, with a documented threat model |
| **Dense reward shaping** | Per-step reward from test-pass-rate deltas, hint-usage and step-count penalties, and a solve bonus |
| **Graded task registry** | 21 tasks across easy, medium, and hard tiers, each with hidden tests and progressive hints |
| **Task mining pipeline** | Extracts candidate tasks from real, merged GitHub bug-fix pull requests for human review |
| **Server and async client** | FastAPI HTTP server with a session-scoped API, plus an async client for training loops |
| **Framework-compatible** | Gymnasium-style `reset()`/`step()`/`state()` interface; usable with SFT, DPO, GRPO, and PPO training code |
| **CI-gated** | 197 tests, linting, and type checking on every change, across Python 3.10 through 3.12 |

---

## Roadmap

- Closed-loop RL training result: fine-tune a small open-weight code model against the environment and publish a measured pass-rate curve
- A trained checkpoint for the optional learned reward model, calibrated against oracle judgment
- Task registry expansion from 21 to several hundred tasks, mined and reviewed from real repositories
- Multi-file task support, extending beyond single-function scope
- Container-level sandbox isolation (namespace and capability restrictions, seccomp filtering)
- Support for a second programming language
- Horizontally scalable session handling for distributed rollout collection
- Listing on the OpenEnv Hub

---

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, the task-authoring template, and the pull request workflow.

---

## Security

Execution isolation is treated as a first-class design concern, not an afterthought — including reporting what the current sandbox does not protect against. See [SECURITY.md](SECURITY.md) for the full threat model and for responsible disclosure of any suspected sandbox escape.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citations
<sup>2</sup> Dhakad, S. CodeFix-Env: A Sandboxed Reinforcement Learning Environment for Automated Code Debugging. arXiv:2607.XXXXX, 2026.

