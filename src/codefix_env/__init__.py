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
from codefix_env.utils.metrics import EpisodeMetrics, ScoringConfig

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


def __getattr__(name):
    """
    Lazy attribute access for RewardMLP only. RewardMLP lives in
    utils/reward_model.py (a torch-only module kept separate from the
    rest of the package) so that `import codefix_env` never pays torch's
    import cost unless someone actually touches RewardMLP. This matters
    most for sandboxed worker processes (utils/sandbox.py) that re-import
    the whole package on every spawn, especially on Windows where
    multiprocessing uses "spawn" instead of "fork".
    """
    if name == "RewardMLP":
        from codefix_env.utils.reward_model import RewardMLP

        return RewardMLP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")