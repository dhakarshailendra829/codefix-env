from codefix_env.utils.metrics import (
    EpisodeMetrics,
    ScoringConfig,
    compute_diff_score,
    compute_final_score,
    compute_shaped_reward,
    compute_test_score,
)
from codefix_env.utils.sandbox import ExecutionResult, run_all_tests, run_code

__all__ = [
    "run_code",
    "run_all_tests",
    "ExecutionResult",
    "compute_test_score",
    "compute_shaped_reward",
    "compute_final_score",
    "compute_diff_score",
    "ScoringConfig",
    "RewardMLP",
    "EpisodeMetrics",
]


def __getattr__(name):
    """Lazy: RewardMLP moved to reward_model.py (torch-only module) so
    importing codefix_env.utils doesn't force torch to load. See
    codefix_env/__init__.py for the matching top-level lazy accessor."""
    if name == "RewardMLP":
        from codefix_env.utils.reward_model import RewardMLP

        return RewardMLP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
