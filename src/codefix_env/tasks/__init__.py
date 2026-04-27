"""
Task Registry — Central access point for all CodeFix tasks.
"""

from __future__ import annotations

import random
from typing import Optional

from codefix_env.models import Difficulty, Task
from codefix_env.tasks.easy import EASY_TASKS
from codefix_env.tasks.hard import HARD_TASKS
from codefix_env.tasks.medium import MEDIUM_TASKS

# ── Master registry ──────────────────────────────────────────────────────────
ALL_TASKS: dict[str, Task] = {task.id: task for task in EASY_TASKS + MEDIUM_TASKS + HARD_TASKS}

TASKS_BY_DIFFICULTY: dict[Difficulty, list[Task]] = {
    Difficulty.EASY: EASY_TASKS,
    Difficulty.MEDIUM: MEDIUM_TASKS,
    Difficulty.HARD: HARD_TASKS,
}


def load_task(task_id: str) -> Task:
    """Load a task by its exact ID."""
    if task_id not in ALL_TASKS:
        raise KeyError(f"Task '{task_id}' not found. Use list_tasks() to see available tasks.")
    return ALL_TASKS[task_id]


def random_task(
    difficulty: Optional[Difficulty] = None,
    exclude: Optional[list[str]] = None,
) -> Task:
    """
    Return a random task, optionally filtered by difficulty and
    excluding already-seen task IDs.
    """
    pool = list(ALL_TASKS.values())
    if difficulty:
        pool = [t for t in pool if t.difficulty == difficulty]
    if exclude:
        pool = [t for t in pool if t.id not in exclude]
    if not pool:
        raise ValueError("No tasks available after applying filters.")
    return random.choice(pool)


def list_tasks(difficulty: Optional[Difficulty] = None) -> list[Task]:
    """List all tasks, optionally filtered by difficulty."""
    if difficulty:
        return TASKS_BY_DIFFICULTY.get(difficulty, [])
    return list(ALL_TASKS.values())


def task_count() -> dict[str, int]:
    """Return count of tasks by difficulty."""
    return {
        "easy": len(EASY_TASKS),
        "medium": len(MEDIUM_TASKS),
        "hard": len(HARD_TASKS),
        "total": len(ALL_TASKS),
    }


__all__ = [
    "ALL_TASKS",
    "EASY_TASKS",
    "MEDIUM_TASKS",
    "HARD_TASKS",
    "load_task",
    "random_task",
    "list_tasks",
    "task_count",
]
