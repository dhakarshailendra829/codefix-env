"""
Basic Usage — Direct Environment (No Server Needed)
=====================================================
Shows how to use CodeFixEnvironment directly (server-side, in-process).
Perfect for local development, debugging, and understanding the API.

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import os
import sys

# Add src/ to path if running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codefix_env import (
    ActionType,
    CodeFixAction,
    CodeFixEnvironment,
    task_count,
)

console = Console()


def print_observation(obs, label="Observation"):
    console.print(f"\n[bold cyan]── {label} ──[/bold cyan]")
    console.print(f"  Task:          {obs.task_id}")
    console.print(f"  Step:          {obs.step_count} / {obs.max_steps}")
    console.print(f"  Tests:         {obs.tests_passed} / {obs.tests_total} passing")
    console.print(f"  Score:         {obs.score_so_far:.3f}")
    console.print(f"  Feedback:      {obs.feedback}")
    if obs.test_output:
        console.print(f"  Test output:\n{obs.test_output}")
    if obs.diff:
        console.print(Syntax(obs.diff, "diff", theme="monokai", line_numbers=False))


def demo_easy_task():
    """Demonstrate solving an easy task step by step."""
    console.print(
        Panel.fit(
            "[bold green]CodeFix-Env Basic Demo[/bold green]\n" "Solving: easy-001-missing-return",
            border_style="green",
        )
    )

    env = CodeFixEnvironment()

    # ── Reset ────────────────────────────────────────────────────────────────
    obs = env.reset(task_id="easy-001-missing-return")
    console.print(f"\n[bold]Task:[/bold] {obs.task_id}")
    console.print(
        "[bold]Description:[/bold] The function adds two numbers but never returns the result."
    )
    console.print("\n[bold]Buggy code:[/bold]")
    console.print(Syntax(obs.current_code, "python", theme="monokai", line_numbers=True))

    # ── Step 1: Run tests (see what's failing) ───────────────────────────────
    console.print("\n[yellow]Step 1: Run tests to see failures[/yellow]")
    result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
    print_observation(result.observation, "After run_tests")
    console.print(f"  Reward: [red]{result.reward:.3f}[/red]")

    # ── Step 2: Get a hint ───────────────────────────────────────────────────
    console.print("\n[yellow]Step 2: Get a hint[/yellow]")
    result = env.step(CodeFixAction(action_type=ActionType.GET_HINT))
    console.print(f"  💡 {result.observation.feedback}")
    console.print(f"  Reward: [red]{result.reward:.3f}[/red] (hint penalty)")

    # ── Step 3: Edit the fix ─────────────────────────────────────────────────
    console.print("\n[yellow]Step 3: Fix the bug — add return statement[/yellow]")
    result = env.step(
        CodeFixAction(
            action_type=ActionType.EDIT_LINE,
            line_number=3,
            new_content="    return result",
            reasoning="The function computes result but never returns it. Add return statement.",
        )
    )
    console.print(f"  Reward: [blue]{result.reward:.3f}[/blue]")

    # ── Step 4: Run tests again ───────────────────────────────────────────────
    console.print("\n[yellow]Step 4: Run tests again[/yellow]")
    result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
    print_observation(result.observation, "After fix")
    console.print(f"  Reward: [green]{result.reward:.3f}[/green]")

    # ── Step 5: Submit ────────────────────────────────────────────────────────
    console.print("\n[yellow]Step 5: Submit the fix[/yellow]")
    result = env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
    console.print(f"\n  Final score: [bold green]{result.reward:.3f}[/bold green]")
    console.print(f"  Done: {result.done}")
    console.print(f"  Solved: {result.observation.all_tests_pass}")

    # ── Show fixed code ───────────────────────────────────────────────────────
    console.print("\n[bold]Fixed code:[/bold]")
    console.print(
        Syntax(result.observation.current_code, "python", theme="monokai", line_numbers=True)
    )


def demo_task_listing():
    """Show all available tasks in a table."""
    from codefix_env import list_tasks

    console.print(Panel.fit("[bold]Available Tasks[/bold]", border_style="blue"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Difficulty", justify="center")
    table.add_column("Bug Category")
    table.add_column("Tests", justify="right")

    for task in list_tasks():
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(
            task.difficulty, "white"
        )
        table.add_row(
            task.id,
            task.title,
            f"[{diff_color}]{task.difficulty}[/{diff_color}]",
            task.bug_category,
            str(task.num_tests),
        )

    console.print(table)
    counts = task_count()
    console.print(
        f"\nTotal: {counts['total']} tasks "
        f"({counts['easy']} easy, {counts['medium']} medium, {counts['hard']} hard)"
    )


if __name__ == "__main__":
    demo_task_listing()
    console.print("\n" + "=" * 60 + "\n")
    demo_easy_task()
