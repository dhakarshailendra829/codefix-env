"""
Baseline Agent Evaluation — proves the RL loop actually closes end-to-end.

Why this exists
----------------
Before this script, the repo had an environment, a reward pipeline, and a
GRPO training *example* — but no evidence anyone had ever actually run an
agent through it and measured results. A reviewer's first question for any
RL-environment project is "ok, but does it work end-to-end?" This script
answers that with a real number, not a claim.

It runs two baseline policies across every task in the registry and plots
pass rate by difficulty:
  1. `oracle` — submits the task's own solution_code line-by-line via
     EDIT_LINE actions, then SUBMIT_FIX. This is a sanity check: if the
     oracle doesn't hit ~100%, the environment or reward pipeline has a bug.
  2. `random` — takes random valid actions. This is the "does the eval
     harness discriminate between good and bad agents at all" check — if
     random performs anywhere close to oracle, the tasks/reward shaping
     are too easy or too noisy to be useful for training.

This does NOT require a GPU or a downloaded LLM checkpoint, which is
deliberate: it's the fast, always-runnable regression test that should live
in CI as a sanity gate on top of the unit tests. Plugging in a real LLM
policy (see examples/grpo_training.py) is the natural next step once this
baseline is green.

Usage:
    PYTHONPATH=src python scripts/run_baseline_eval.py --episodes 5
"""

from __future__ import annotations

import argparse
import random as random_module
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codefix_env import ActionType, CodeFixAction, CodeFixEnvironment, list_tasks  # noqa: E402


def run_oracle_episode(env: CodeFixEnvironment, task_id: str) -> bool:
    """Apply the task's known-correct solution line-by-line, then submit."""
    obs = env.reset(task_id=task_id)
    from codefix_env.tasks import load_task

    task = load_task(task_id)
    solution_lines = task.solution_code.splitlines()
    current_lines = obs.current_code.splitlines()

    # Pad/replace so current code becomes exactly the solution, line by line.
    max_len = max(len(solution_lines), len(current_lines))
    for i in range(max_len):
        target = solution_lines[i] if i < len(solution_lines) else ""
        if i < len(current_lines):
            result = env.step(
                CodeFixAction(action_type=ActionType.EDIT_LINE, line_number=i + 1, new_content=target)
            )
        else:
            result = env.step(
                CodeFixAction(action_type=ActionType.INSERT_LINE, line_number=i, new_content=target)
            )
        if result.done:
            return result.observation.all_tests_pass

    result = env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
    return result.observation.all_tests_pass


def run_random_episode(env: CodeFixEnvironment, task_id: str, max_steps: int = 15) -> bool:
    """Take random valid-shaped actions; used as a floor baseline."""
    obs = env.reset(task_id=task_id)
    for _ in range(max_steps):
        num_lines = len(obs.current_code.splitlines()) or 1
        action_type = random_module.choice(
            [ActionType.RUN_TESTS, ActionType.EDIT_LINE, ActionType.GET_HINT, ActionType.VIEW_CODE]
        )
        line_number = random_module.randint(1, num_lines) if action_type == ActionType.EDIT_LINE else None
        content = "    pass" if action_type == ActionType.EDIT_LINE else None
        result = env.step(CodeFixAction(action_type=action_type, line_number=line_number, new_content=content))
        obs = result.observation
        if result.done:
            return obs.all_tests_pass
    result = env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
    return result.observation.all_tests_pass


def evaluate(policy_name: str, episodes_per_task: int) -> dict:
    env = CodeFixEnvironment(max_steps=30)
    run_fn = run_oracle_episode if policy_name == "oracle" else run_random_episode
    results_by_difficulty: dict[str, list[bool]] = defaultdict(list)

    for task in list_tasks():
        for _ in range(episodes_per_task):
            solved = run_fn(env, task.id)
            results_by_difficulty[str(task.difficulty)].append(solved)

    return {
        diff: sum(r) / len(r) for diff, r in results_by_difficulty.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3, help="episodes per task per policy")
    parser.add_argument("--out", default="baseline_results.png")
    args = parser.parse_args()

    print("Running oracle baseline (sanity check — should be ~100%)...")
    oracle_scores = evaluate("oracle", args.episodes)
    print(oracle_scores)

    print("Running random baseline (floor — should be low)...")
    random_scores = evaluate("random", args.episodes)
    print(random_scores)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        difficulties = sorted(set(oracle_scores) | set(random_scores))
        x = range(len(difficulties))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([i - width / 2 for i in x], [oracle_scores.get(d, 0) for d in difficulties], width, label="oracle")
        ax.bar([i + width / 2 for i in x], [random_scores.get(d, 0) for d in difficulties], width, label="random")
        ax.set_xticks(list(x))
        ax.set_xticklabels(difficulties)
        ax.set_ylabel("Pass rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("CodeFix-Env: oracle vs. random baseline pass rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"Saved chart to {args.out}")
    except ImportError:
        print("matplotlib not installed — skipping chart, numeric results printed above.")


if __name__ == "__main__":
    main()
