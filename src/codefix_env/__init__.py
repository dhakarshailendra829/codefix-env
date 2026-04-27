"""
CodeFix-Env — RL Environment for Automated Code Debugging
==========================================================

Quick start::

    from codefix_env import CodeFixEnvironment, CodeFixAction, CodeFixClient

    # Server-side (direct, no HTTP)
    env = CodeFixEnvironment()
    obs = env.reset(task_id="easy-001-missing-return")
    result = env.step(CodeFixAction(action_type="run_tests"))

    # Client-side (HTTP)
    async with CodeFixClient("http://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(CodeFixAction(action_type="submit_fix"))
"""

__version__ = "0.2.0"
__author__ = "Shailendra Dhakar"

from codefix_env.client import CodeFixClient, SyncCodeFixClient
from codefix_env.env import CodeFixEnvironment
from codefix_env.models import (
    ActionType,
    BugCategory,
    CodeFixAction,
    CodeFixObservation,
    CodeFixState,
    Difficulty,
    StepResult,
    Task,
    TestCase,
    TestResult,
)
from codefix_env.rewards import RewardPipeline
from codefix_env.tasks import list_tasks, load_task, random_task, task_count
from codefix_env.utils.metrics import EpisodeMetrics, RewardMLP, ScoringConfig

__all__ = [
    # Core
    "CodeFixEnvironment",
    "CodeFixClient",
    "SyncCodeFixClient",
    # Models
    "CodeFixAction",
    "CodeFixObservation",
    "CodeFixState",
    "StepResult",
    "Task",
    "TestCase",
    "TestResult",
    "ActionType",
    "Difficulty",
    "BugCategory",
    # Rewards
    "RewardPipeline",
    "ScoringConfig",
    "RewardMLP",
    "EpisodeMetrics",
    # Tasks
    "load_task",
    "random_task",
    "list_tasks",
    "task_count",
]
