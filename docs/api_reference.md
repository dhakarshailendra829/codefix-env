# API Reference

## CodeFixEnvironment (Server-Side)

```python
from codefix_env import CodeFixEnvironment
env = CodeFixEnvironment(
    default_difficulty=Difficulty.MEDIUM,  # default task difficulty
    max_steps=20,                          # episode step limit
)
```

### `env.reset(task_id?, difficulty?, seed?) → CodeFixObservation`
Start a new episode. Returns initial observation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str \| None` | `None` | Specific task ID. If None, random task is selected. |
| `difficulty` | `Difficulty \| None` | `None` | Filter random selection by difficulty. |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. |

### `env.step(action) → StepResult`
Execute one action. Returns `StepResult(observation, reward, done, truncated, info)`.

### `env.state() → CodeFixState`
Get internal episode state (non-destructive).

---

## CodeFixAction

```python
from codefix_env import CodeFixAction, ActionType

action = CodeFixAction(
    action_type=ActionType.EDIT_LINE,
    line_number=5,
    new_content="    return result",
    reasoning="Chain-of-thought (optional, logged)",
)
```

| Field | Required | Description |
|---|---|---|
| `action_type` | ✅ | One of the `ActionType` enum values |
| `line_number` | Edit/Insert/Delete only | Target line (1-indexed) |
| `new_content` | Edit/Insert only | New line text |
| `reasoning` | ❌ | Agent's chain-of-thought (logged, not used by env) |

### ActionType Values

| Value | Description |
|---|---|
| `run_tests` | Execute all test cases, get test output |
| `edit_line` | Replace a line with new content |
| `insert_line` | Insert a new line after given line number |
| `delete_line` | Delete a line |
| `get_hint` | Request a hint (costs `hint_penalty` reward) |
| `submit_fix` | Submit final solution, ends episode |
| `view_code` | View current code state (free action) |

---

## CodeFixObservation

```python
obs: CodeFixObservation = env.reset()
obs.current_code    # str  — current (possibly edited) code
obs.original_code   # str  — original buggy code
obs.diff            # str  — unified diff: original → current
obs.test_results    # list[TestResult]
obs.test_output     # str  — human-readable test output
obs.tests_passed    # int
obs.tests_total     # int
obs.all_tests_pass  # bool
obs.score_so_far    # float [0.0, 1.0]
obs.shaped_reward   # float — reward from last step
obs.step_count      # int
obs.max_steps       # int
obs.steps_remaining # int
obs.done            # bool
obs.task_id         # str
obs.difficulty      # Difficulty
obs.bug_category    # BugCategory
obs.hints_used      # int
obs.hint_available  # bool
obs.feedback        # str — human-readable message
obs.error_message   # str — error if action failed
```

---

## StepResult

```python
result = env.step(action)
result.observation   # CodeFixObservation
result.reward        # float — shaped reward for this step
result.done          # bool  — episode ended normally (solved/submitted)
result.truncated     # bool  — episode ended due to max_steps
result.info          # dict  — episode_id, task_id, total_reward
```

---

## Task Management

```python
from codefix_env import load_task, random_task, list_tasks, task_count

task = load_task("easy-001-missing-return")
task.id             # str
task.title          # str
task.description    # str
task.buggy_code     # str
task.solution_code  # str
task.test_cases     # list[TestCase]
task.difficulty     # Difficulty
task.bug_category   # BugCategory
task.tags           # list[str]
task.hints          # list[str]
task.max_steps      # int

random_task(difficulty=Difficulty.HARD, exclude=["hard-001-lru-cache"])
list_tasks(difficulty=Difficulty.EASY)
task_count()  # {'easy': 8, 'medium': 8, 'hard': 5, 'total': 21}
```

---

## RewardPipeline

```python
from codefix_env import RewardPipeline
from codefix_env.utils.metrics import ScoringConfig

pipeline = RewardPipeline(
    cfg=ScoringConfig(
        max_steps=20,
        hint_penalty=0.05,
        step_penalty=0.01,
        solve_bonus=0.20,
        time_decay_gamma=0.99,
    ),
    neural_model_path="path/to/reward_model.pt",  # optional
    neural_weight=0.3,
)

# step reward
reward = pipeline.step_reward(prev_obs, curr_obs, action_type, task)

# episode final reward
final = pipeline.episode_reward(final_obs, task)
```

---

## Neural Reward Model (RewardMLP)

```python
import torch
from codefix_env.utils.metrics import RewardMLP

model = RewardMLP(hidden_dim=64)

# Train it on (features, target_reward) pairs
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.rand(32, RewardMLP.INPUT_DIM)
y = torch.rand(32, 1)
loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
optimizer.step()

# Save for use in pipeline
torch.save(model.state_dict(), "reward_model.pt")
```

---

## HTTP API (Server)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health + uptime |
| `POST` | `/reset` | Start episode, returns session ID in header |
| `POST` | `/step` | Execute action (requires `X-Session-ID` header) |
| `GET` | `/state` | Get episode state (requires `X-Session-ID` header) |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/{id}` | Get task details |
| `GET` | `/metrics` | Server metrics |

Interactive docs: http://localhost:8000/docs