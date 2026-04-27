# Quick Start Guide

## Installation

### From PyPI (recommended)
```bash
pip install codefix-env
```

### From source
```bash
git clone https://github.com/dhakarshailendra829/codefix-env
cd codefix-env
pip install -e ".[dev]"
```

### With LLM training support
```bash
pip install "codefix-env[llm]"
```

---

## Option A — Direct (No Server)

Use the environment directly in Python. No server needed. Best for local testing.

```python
from codefix_env import CodeFixEnvironment, CodeFixAction, ActionType

env = CodeFixEnvironment()

# Reset — choose a task
obs = env.reset(task_id="easy-001-missing-return")
print(obs.current_code)      # buggy code
print(obs.tests_total)       # number of test cases

# Run tests to see failures
result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
print(result.observation.test_output)
print(result.observation.tests_passed)  # should be 0

# Fix the bug: add missing return statement
result = env.step(CodeFixAction(
    action_type=ActionType.EDIT_LINE,
    line_number=3,
    new_content="    return result",
))

# Run tests again
result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
print(result.observation.tests_passed)  # should be 4

# Submit
result = env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
print(f"Score: {result.reward:.3f}")    # should be ~0.9+
print(f"Done: {result.done}")
```

---

## Option B — HTTP Client (Async)

Start the server, then use the async client. Best for RL training loops.

### Step 1: Start server
```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Use async client
```python
import asyncio
from codefix_env import CodeFixClient, CodeFixAction, ActionType

async def main():
    async with CodeFixClient("http://localhost:8000") as env:
        obs = await env.reset(task_id="medium-002-binary-search")
        print(f"Task: {obs.task_id}")

        result = await env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
        print(f"Tests passing: {result.observation.tests_passed}/{result.observation.tests_total}")

        result = await env.step(CodeFixAction(action_type=ActionType.GET_HINT))
        print(result.observation.feedback)

asyncio.run(main())
```

### Step 3: Or use sync wrapper
```python
from codefix_env import CodeFixClient, CodeFixAction, ActionType

with CodeFixClient("http://localhost:8000").sync() as env:
    obs = env.reset()
    result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
    print(result.observation.test_output)
```

---

## Option C — Docker

```bash
docker pull ghcr.io/dhakarshailendra829/codefix-env:latest
docker run -p 8000:8000 ghcr.io/dhakarshailendra829/codefix-env:latest
```

---

## Browse Available Tasks

```python
from codefix_env import list_tasks, task_count

# List all tasks
for task in list_tasks():
    print(f"{task.id:40s} | {task.difficulty:6s} | {task.bug_category}")

# Count by difficulty
print(task_count())
# {'easy': 8, 'medium': 8, 'hard': 5, 'total': 21}
```

---

## Run the Demo

```bash
python examples/basic_usage.py
```

---

## Run Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Next Steps

- [API Reference](./api_reference.md) — full API docs
- [Custom Tasks](./custom_tasks.md) — add your own debugging tasks
- [GRPO Training](../examples/grpo_training.py) — train LLMs with GRPO
- [Benchmark](../examples/benchmark.py) — compare agents