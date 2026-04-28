<div align="center">

# 🛠️ CodeFix-Env

**A specialized RL environment for training LLMs to autonomously debug and repair code through sandboxed feedback loops.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-197%2F197%20Passing-brightgreen)](tests/)

---
</div>
## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#Usage-and-developement-guide)
- [Task Categories](#task-categories)
- [Usage Examples](#usage-examples)
- [LLM Training](#llm-training)
- [API Reference](#api-reference)
- [CI/CD Pipeline](#cicd-pipeline)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

CodeFix-Env is a **Gymnasium-compatible RL environment** that simulates real-world Python code debugging workflows. It enables researchers and engineers to:

- Train large language models (LLMs) on code repair tasks
- Evaluate AI agents on real debugging scenarios
- Benchmark code generation and reasoning capabilities
- Build automated debugging systems

Perfect for anyone working with LLMs like Qwen, Llama, CodeLlama, or DeepSeek models.
---

### The Challenge
Training Large Language Models (LLMs) for code generation and autonomous repair requires more than static data. Effective learning requires:
* **Authentic Scenarios:** Moving beyond synthetic examples to real-world debugging.
* **Tight Feedback Loops:** Providing immediate test results and error logs.
* **Measurable Progress:** Quantifiable metrics based on test pass rates.
* **Secure Execution:** Sandboxed environments to prevent unsafe code execution.

### The Solution: CodeFix-Env
CodeFix-Env is a specialized environment designed for RL agents and LLMs to master code repair through:
* **Real-World Tasks:** 21 curated Python bugs ranging from logic errors to complex edge cases.
* **Test-Driven Feedback:** Agents receive granular execution data after every action.
* **Multi-Factor Rewards:** Reward signals based on test progress, code quality, and task completion.
* **Isolated Execution:** Built-in sandboxing with timeout protection for safe evaluations.
* **Structured Interaction:** Clean API for structured actions that LLMs can reliably generate.

### Workflow
```mermaid
graph TD
    A[Agent receives buggy code] --> B[Analyze code & test failures]
    B --> C[Generate action: e.g., edit_line]
    C --> D[Environment executes action]
    D --> E[Environment provides feedback]
    
    subgraph Feedback Loop
    E --> F[Test results & pass/fail counts]
    E --> G[Code diff & change analysis]
    E --> H[Shaped reward signals]
    end
    
    F & G & H --> I{Solution found?}
    I -- No --> B
    I -- Yes --> J[Task Complete]
```
---

## Key Features

** Task Library**
- 21 carefully designed debugging tasks
- 3 difficulty levels: Easy (8), Medium (8), Hard (5)
- Each task includes buggy code, solution, test cases, and hints
- Covers real Python patterns: loops, recursion, algorithms, decorators

** LLM Integration**
- Works with HuggingFace Transformers ecosystem
- Compatible with training frameworks: SFT, DPO, GRPO, PPO
- Pre-built support for Qwen, Llama, CodeLlama, DeepSeek models
- Structured JSON action interface for LLM control

** Performance**
- Multiprocessing-based sandbox with timeout enforcement
- Per-test execution isolation
- 197 unit tests with 83% code coverage
- Automated CI/CD pipeline (Python 3.10, 3.11, 3.12)

** Developer Friendly**
- Gymnasium standard interface
- Async client for high-performance training
- Sync wrapper for notebooks
- FastAPI HTTP server for remote access
- Complete type hints and documentation

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/dhakarshailendra829/codefix-env.git
cd codefix-env
```
### Step 2: Create virual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate (On mac)
.venv/Scripts/activate (On Windows)
```
### Step 3: Intall Package
#### For Basic Usage
```bash
pip install -e . 
```
---
#### For Developement
```bash
pip install -e ".[dev]"
```
---
#### For LLM training (includes transformers, TRL, datasets)
```bash
pip install -e ".[llm]"
```
---
#### For everything
```bash
pip install -e ".[dev,llm]"
```
### Step 4: Verify Installation
```bash
python -c "from codefix_env import CodeFixEnvironment; print(' Installation successful!')"
```
---
## Usage & Development Guide

### 1. Basic Environment Usage
```python
python << 'EOF'
from codefix_env import CodeFixEnvironment, CodeFixAction, ActionType

# Create environment
env = CodeFixEnvironment()

# Start episode with random easy task
obs = env.reset(difficulty="easy")
print(f"Task: {obs.task_id}")
print(f"Current Code:\n{obs.current_code}")

# Run tests to see failures
action = CodeFixAction(action_type=ActionType.RUN_TESTS)
result = env.step(action)
print(f"\nTest Results: {result.observation.tests_passed}/{result.observation.tests_total} passing")
print(f"Output:\n{result.observation.test_output}")

print(f"\nReward: {result.reward:.3f}")
print(f"Episode Done: {result.done}")
EOF
```

### 2. Testing & Quality Assurance
To run the full suite:
```bash
cd codefix-env
PYTHONPATH=src pytest tests/ -v --timeout=60
```

To run specific Test Suite
```bash
PYTHONPATH=src pytest tests/test_tasks.py::TestSolutionCorrectness -v
```

To check code standards:
```bash
ruff check src/ tests/ examples/
black --check src/ tests/ examples/
mypy src/codefix_env/ --ignore-missing-imports
```

### 3. API Server Deployment
```bash
cd codefix-env
uvicorn src.codefix_env.server.app:app --reload --port 8000
```
---
## Usage Examples

### 1. Core Debugging Workflow
A complete cycle from task initialization to final submission.

```python
from codefix_env import CodeFixEnvironment, CodeFixAction, ActionType

env = CodeFixEnvironment()
obs = env.reset(task_id="easy-001-missing-return")

# 1. Run tests to identify failures
result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
print(f"Tests: {result.observation.tests_passed}/{result.observation.tests_total} passing")

# 2. Apply a targeted fix
result = env.step(CodeFixAction(
    action_type=ActionType.EDIT_LINE,
    line_number=3,
    new_content="    return result"
))

# 3. Final Verification & Submission
result = env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
print(f"Status: {'Solved' if result.observation.all_tests_pass else 'Failed'}")
print(f"Final Reward: {result.reward:.3f}")
```

### 2. High-Level Task Management
Programmatically interact with the task database.

```python
from codefix_env import list_tasks, Difficulty, task_count

# Get environment stats
print(f"Total tasks by difficulty: {task_count()}")

# Bulk process specific difficulty levels
easy_tasks = list_tasks(difficulty=Difficulty.EASY)
for task in easy_tasks[:5]:
    print(f"Loading {task.id}: {task.title}")
```

---

## LLM Training & Integration

### Server-Client Architecture
Deploy the environment as a service to scale LLM evaluations across multiple agents.

```bash
# Start the API server
uvicorn src.codefix_env.server.app:app --port 8000
```

### Building Training Datasets
Export environment states into `.jsonl` formats for fine-tuning models like Llama or Qwen.

```python
import json
from codefix_env import CodeFixClient

async def collect_trajectories():
    async with CodeFixClient("http://localhost:8000") as client:
        obs = await client.reset(difficulty="medium")
        dataset_entry = {
            "instruction": f"Fix the following code: {obs.current_code}",
            "context": obs.test_output,
            "metadata": {"task_id": obs.task_id}
        }
        # Append to training_data.jsonl...
```

### Zero-Shot Inference (HuggingFace)
Connect an LLM directly to the environment loop.

```python
from transformers import pipeline
from codefix_env import CodeFixEnvironment

# Load a coding model
fixer = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-7B")

env = CodeFixEnvironment()
obs = env.reset(difficulty="easy")

# Generate fix based on environment observation
prompt = f"Fix this Python code based on these errors: {obs.test_output}\nCode:\n{obs.current_code}"
suggestion = fixer(prompt)
```

---

## API Quick Reference


| Class / Method | Description |
| :--- | :--- |
| `CodeFixEnvironment` | Local execution engine for debugging tasks. |
| `CodeFixClient` | Async client for interacting with a remote server. |
| `env.step(action)` | Executes an `ActionType` (EDIT, RUN_TESTS, SUBMIT). |
| `Difficulty` | Enum: `EASY`, `MEDIUM`, `HARD`. |
| `BugCategory` | Categories: `SYNTAX`, `LOGIC`, `DATA_STRUCTURE`, `ALGORITHM`. |

---

## Development & CI/CD
Standard commands for maintaining the environment.

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=src/codefix_env

# Static Analysis
ruff check src/   # Linting
black --check src/ # Formatting
mypy src/         # Type Checking
```
