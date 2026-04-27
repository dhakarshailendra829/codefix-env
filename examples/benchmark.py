"""
Benchmark — Evaluate Any Agent Against CodeFix-Env
====================================================
Runs an agent through all tasks and reports:
- Tasks solved (%)
- Average steps to solve
- Average score
- Per-difficulty breakdown

Built-in agents:
  - RandomAgent    : takes random actions
  - GreedyAgent    : always edits line 1 then submits
  - OracleAgent    : cheats by copying solution (upper bound)

Run:
    python examples/benchmark.py --agent random
    python examples/benchmark.py --agent oracle
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from codefix_env import (
    ActionType,
    CodeFixAction,
    CodeFixEnvironment,
    CodeFixObservation,
    Difficulty,
    list_tasks,
    task_count,
)
from codefix_env.utils.metrics import EpisodeMetrics

console = Console()


# ── Base Agent ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for benchmark agents."""

    def reset(self, obs: CodeFixObservation) -> None:
        """Called at the start of each episode."""
        pass

    @abstractmethod
    def act(self, obs: CodeFixObservation) -> CodeFixAction:
        """Choose the next action given the current observation."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Built-in Agents ───────────────────────────────────────────────────────────

class RandomAgent(BaseAgent):
    """Takes completely random valid actions."""

    def act(self, obs: CodeFixObservation) -> CodeFixAction:
        lines = obs.current_code.splitlines()
        n_lines = max(len(lines), 1)
        choice = random.choice([
            ActionType.RUN_TESTS,
            ActionType.RUN_TESTS,   # weight run_tests higher
            ActionType.EDIT_LINE,
            ActionType.SUBMIT_FIX,
        ])
        if choice == ActionType.EDIT_LINE:
            ln = random.randint(1, n_lines)
            return CodeFixAction(
                action_type=choice,
                line_number=ln,
                new_content=lines[ln - 1] if ln <= len(lines) else "    pass",
            )
        return CodeFixAction(action_type=choice)


class GreedyRunSubmitAgent(BaseAgent):
    """Runs tests on step 1, submits on step 2. No actual fixing."""

    def __init__(self):
        self._step = 0

    def reset(self, obs):
        self._step = 0

    def act(self, obs: CodeFixObservation) -> CodeFixAction:
        self._step += 1
        if self._step == 1:
            return CodeFixAction(action_type=ActionType.RUN_TESTS)
        return CodeFixAction(action_type=ActionType.SUBMIT_FIX)


class OracleAgent(BaseAgent):
    """
    Cheating oracle: copies solution code line by line.
    Represents the theoretical upper bound.
    """

    def __init__(self):
        self._solution_lines: list[str] = []
        self._edit_idx: int = 0
        self._submitted: bool = False

    def reset(self, obs: CodeFixObservation) -> None:
        # Load the task to get solution (cheating!)
        from codefix_env.tasks import load_task
        task = load_task(obs.task_id)
        self._solution_lines = task.solution_code.splitlines()
        self._edit_idx = 0
        self._submitted = False

    def act(self, obs: CodeFixObservation) -> CodeFixAction:
        if self._submitted:
            return CodeFixAction(action_type=ActionType.SUBMIT_FIX)

        curr_lines = obs.current_code.splitlines()

        # Find first differing line
        for i, (curr, sol) in enumerate(
            zip(curr_lines, self._solution_lines), start=1
        ):
            if curr != sol:
                return CodeFixAction(
                    action_type=ActionType.EDIT_LINE,
                    line_number=i,
                    new_content=sol,
                )

        # Handle length differences
        if len(self._solution_lines) > len(curr_lines):
            return CodeFixAction(
                action_type=ActionType.INSERT_LINE,
                line_number=len(curr_lines),
                new_content=self._solution_lines[len(curr_lines)],
            )
        if len(curr_lines) > len(self._solution_lines):
            return CodeFixAction(
                action_type=ActionType.DELETE_LINE,
                line_number=len(curr_lines),
            )

        # All lines match — submit
        self._submitted = True
        return CodeFixAction(action_type=ActionType.SUBMIT_FIX)


# ── Benchmark Runner ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    task_id:    str
    difficulty: str
    solved:     bool
    steps:      int
    score:      float
    time_s:     float
    hints_used: int


@dataclass
class BenchmarkSummary:
    agent_name: str
    results:    list[BenchmarkResult] = field(default_factory=list)

    @property
    def solve_rate(self) -> float:
        if not self.results: return 0.0
        return sum(r.solved for r in self.results) / len(self.results)

    @property
    def avg_score(self) -> float:
        if not self.results: return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def avg_steps(self) -> float:
        solved = [r for r in self.results if r.solved]
        if not solved: return 0.0
        return sum(r.steps for r in solved) / len(solved)

    def by_difficulty(self) -> dict[str, dict]:
        out = {}
        for diff in ("easy", "medium", "hard"):
            subset = [r for r in self.results if r.difficulty == diff]
            if not subset:
                continue
            out[diff] = {
                "total":      len(subset),
                "solved":     sum(r.solved for r in subset),
                "solve_rate": sum(r.solved for r in subset) / len(subset),
                "avg_score":  sum(r.score for r in subset) / len(subset),
            }
        return out


def run_benchmark(
    agent: BaseAgent,
    difficulty: str = "all",
    max_tasks: int = 999,
) -> BenchmarkSummary:
    """Run agent against all (or filtered) tasks."""
    env = CodeFixEnvironment()

    if difficulty == "all":
        tasks = list_tasks()
    else:
        tasks = list_tasks(difficulty=Difficulty(difficulty))

    tasks = tasks[:max_tasks]
    summary = BenchmarkSummary(agent_name=agent.name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task_bar = progress.add_task(f"[cyan]Benchmarking {agent.name}...", total=len(tasks))

        for task in tasks:
            obs = env.reset(task_id=task.id)
            agent.reset(obs)
            t0 = time.perf_counter()
            total_reward = 0.0
            done = False

            while not done:
                action = agent.act(obs)
                result = env.step(action)
                obs = result.observation
                total_reward += result.reward
                done = result.done or result.truncated

            elapsed = time.perf_counter() - t0
            summary.results.append(BenchmarkResult(
                task_id=task.id,
                difficulty=task.difficulty,
                solved=obs.all_tests_pass,
                steps=obs.step_count,
                score=obs.score_so_far,
                time_s=elapsed,
                hints_used=obs.hints_used,
            ))
            progress.advance(task_bar)

    return summary


def print_summary(summary: BenchmarkSummary):
    """Pretty-print benchmark summary."""
    console.print(f"\n[bold]Agent:[/bold] {summary.agent_name}")

    # Overall table
    table = Table(title="Overall Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric",     style="cyan")
    table.add_column("Value",      justify="right")
    table.add_row("Tasks run",     str(len(summary.results)))
    table.add_row("Solve rate",    f"{summary.solve_rate:.1%}")
    table.add_row("Avg score",     f"{summary.avg_score:.3f}")
    table.add_row("Avg steps (solved)", f"{summary.avg_steps:.1f}")
    console.print(table)

    # Per-difficulty table
    by_diff = summary.by_difficulty()
    if by_diff:
        diff_table = Table(title="By Difficulty", show_header=True, header_style="bold magenta")
        diff_table.add_column("Difficulty")
        diff_table.add_column("Tasks", justify="right")
        diff_table.add_column("Solved", justify="right")
        diff_table.add_column("Solve Rate", justify="right")
        diff_table.add_column("Avg Score", justify="right")

        for diff, stats in by_diff.items():
            color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(diff, "white")
            diff_table.add_row(
                f"[{color}]{diff}[/{color}]",
                str(stats["total"]),
                str(stats["solved"]),
                f"{stats['solve_rate']:.1%}",
                f"{stats['avg_score']:.3f}",
            )
        console.print(diff_table)


# ── CLI ────────────────────────────────────────────────────────────────────────

AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyRunSubmitAgent,
    "oracle": OracleAgent,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark agents on CodeFix-Env")
    parser.add_argument("--agent",      choices=list(AGENTS), default="random")
    parser.add_argument("--difficulty", choices=["all", "easy", "medium", "hard"], default="all")
    parser.add_argument("--max-tasks",  type=int, default=999)
    args = parser.parse_args()

    agent = AGENTS[args.agent]()
    console.print(f"[bold green]CodeFix-Env Benchmark[/bold green]")
    console.print(f"Agent: {agent.name} | Difficulty: {args.difficulty}")
    console.print(f"Total tasks: {task_count()['total']}\n")

    summary = run_benchmark(agent, difficulty=args.difficulty, max_tasks=args.max_tasks)
    print_summary(summary)