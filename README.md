# CodeFix-Env

A sandboxed reinforcement learning environment for training and evaluating LLM agents on automated code debugging.

[![PyPI](https://img.shields.io/pypi/v/codefix-env?color=blue)](https://pypi.org/project/codefix-env/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-informational)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-197%20passing-brightgreen)](tests/)
[![CI](https://github.com/dhakarshailendra829/codefix-env/actions/workflows/ci.yml/badge.svg)](https://github.com/dhakarshailendra829/codefix-env/actions)

## Why this exists

Training a coding agent to debug software requires more than static examples of correct code. It requires a tight loop: propose a fix, execute it against a real test suite, and receive a signal that reflects genuine progress rather than superficial activity.

Most environments used for this today sit at one of two extremes. Repository-scale benchmarks execute an agent's changes against a full project inside a container, which is realistic but slow — often minutes per episode — and better suited to periodic evaluation than to the high-volume rollout collection that reinforcement learning training requires. At the other end, many agent demonstrations execute model-generated code with little or no isolation from the host system, which is acceptable for a demo and not for anything run unattended.

CodeFix-Env is built for the space between these two extremes: single-function debugging tasks with hidden test suites, executing in well under a second, inside a sandbox with a stated threat model rather than an assumed one.

## What it does

| Goal | Description |
|---|---|
| Train | Fine-tune LLMs on code repair tasks using dense, shaped rewards |
| Evaluate | Score agents against hidden test suites across graded difficulty |
| Benchmark | Compare debugging performance across models and policies |
| Extend | Mine new tasks directly from real GitHub bug-fix commits |

Works with any model accessible through the Hugging Face `transformers` ecosystem or a compatible inference server, including Qwen, Llama, CodeLlama, and DeepSeek-Coder family models.

## Installation

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

## Quick start

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

## How it works

Each episode loads a task — a buggy function, a hidden test suite, and graduated hints — and returns an observation with no tests yet executed. The agent acts through a small, structured action space: run the hidden tests, edit, insert, or delete a line, request a hint, view the current code, or submit a final answer. Every code-mutating action returns a unified diff against the original, so the agent can see exactly what has changed without recomputing it.

Reward is computed at every step, not only at episode end. It combines the change in fraction of tests passing, a bonus for a full solve, and penalties for excess hints and steps, giving a trainable signal throughout an episode rather than a single terminal bit. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design, including the reward formula.

Execution is isolated through a layered sandbox: static analysis rejects unsafe imports and calls, a restricted builtin namespace blocks filesystem access, and the executing process runs under OS-level resource limits. [`SECURITY.md`](SECURITY.md) documents this design in full, including two hardening attempts that were tried, tested under load, found to fail, and reverted — kept in the record rather than left out.

## Task management

```python
from codefix_env import list_tasks, Difficulty, task_count

print(task_count())  # {'easy': 8, 'medium': 8, 'hard': 5, 'total': 21}

easy_tasks = list_tasks(difficulty=Difficulty.EASY)
for task in easy_tasks[:5]:
    print(task.id, task.title)
```

## Running as a server

```bash
codefix-server serve --port 8000
```

Exposes the environment over HTTP with a session-scoped API (`X-Session-ID` header), suitable for remote or multi-language clients and for driving GRPO/PPO training loops via the async `CodeFixClient`:

```python
from codefix_env import CodeFixClient

async def run_episode():
    async with CodeFixClient("http://localhost:8000") as client:
        obs = await client.reset(difficulty="medium")
        result = await client.step(action)
```

## Task library

21 tasks across three difficulty tiers, each with a buggy function, a reference solution, a hidden test suite, and up to three hints.

| Difficulty | Count | Example categories |
|---|---|---|
| Easy | 8 | return bugs, off-by-one, wrong operators, inverted conditions |
| Medium | 8 | algorithmic bugs, scope errors, mutable default arguments |
| Hard | 5 | multi-step logic (LRU cache, graph traversal, tokenization) |

A pipeline for mining additional tasks directly from real, merged bug-fix pull requests on GitHub is included (`scripts/mine_tasks.py`), producing a human-reviewed queue rather than an unreviewed automatic feed.

## Baseline evaluation

```bash
python scripts/run_baseline_eval.py --episodes 3
```

Comparing an oracle policy (applies the known-correct fix) against a random policy across all 21 tasks confirms the reward signal separates genuine problem-solving from undirected action:

| Policy | Easy | Medium | Hard |
|---|---|---|---|
| Oracle | 100% | 87.5% | 60% |
| Random | 0% | 0% | 20% |

No trained-policy result is published yet. Running `examples/grpo_training.py` against a GRPO training loop is the next step toward that; see Roadmap.

## Testing and code quality

```bash
PYTHONPATH=src pytest tests/ -v --timeout=60
ruff check src/ tests/ examples/
black --check src/ tests/ examples/
mypy src/codefix_env/ --ignore-missing-imports
```

197 tests, run on every push across Python 3.10, 3.11, and 3.12.

## Roadmap

- Closed-loop RL training result: fine-tune a small open-weight code model against the environment and publish a measured pass-rate curve
- A trained checkpoint for the optional learned reward model, calibrated against oracle judgment
- Task registry expansion from 21 to several hundred tasks, mined and reviewed from real repositories
- Multi-file task support, extending beyond single-function scope
- Container-level sandbox isolation (namespace and capability restrictions, seccomp filtering)
- Support for a second programming language
- Horizontally scalable session handling for distributed rollout collection
- Listing on the OpenEnv Hub

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design and reward formulation
- [`SECURITY.md`](SECURITY.md) — sandbox threat model
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution workflow

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup and workflow.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
