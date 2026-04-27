"""
Metrics & Scoring
==================
All reward and score computation for CodeFix-Env.

Design:
- Base reward = fraction of tests passing
- Shaped reward = dense signal per step (partial credit)
- Penalties  = hint usage, extra steps, time
- Neural reward model (optional) = learned reward via PyTorch MLP
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Score Config
# ─────────────────────────────────────────────


@dataclass
class ScoringConfig:
    """Hyper-parameters for reward computation."""

    max_steps: int = 20
    hint_penalty: float = 0.05  # per hint used
    step_penalty: float = 0.01  # per extra step beyond min
    solve_bonus: float = 0.20  # bonus for solving all tests
    partial_credit_alpha: float = 0.5  # weight of partial progress in shaped reward
    time_decay_gamma: float = 0.99  # exponential decay per step


# ─────────────────────────────────────────────
# Rule-Based Rewards
# ─────────────────────────────────────────────


def compute_test_score(tests_passed: int, tests_total: int) -> float:
    """Fraction of tests passing. Core task signal."""
    if tests_total == 0:
        return 0.0
    return tests_passed / tests_total


def compute_shaped_reward(
    prev_passed: int,
    curr_passed: int,
    tests_total: int,
    step_count: int,
    hints_used: int,
    action_type: str,
    cfg: ScoringConfig = ScoringConfig(),
) -> float:
    """
    Dense, shaped reward for a single step.

    Positive signal:
    - Progress: new tests passing → reward proportional to delta
    - Solve bonus: all tests pass

    Negative signal:
    - Regression: tests were broken
    - Hint penalty
    - Step penalty for idle actions (run_tests without changes)
    """
    reward = 0.0

    if tests_total > 0:
        delta = (curr_passed - prev_passed) / tests_total
        # Progress signal with partial credit
        reward += delta * (1.0 + cfg.partial_credit_alpha)

    # Solve bonus
    if curr_passed == tests_total and tests_total > 0:
        reward += cfg.solve_bonus

    # Hint penalty
    reward -= hints_used * cfg.hint_penalty

    # Time decay: later steps get slightly less reward
    reward *= cfg.time_decay_gamma**step_count

    return float(reward)


def compute_final_score(
    tests_passed: int,
    tests_total: int,
    step_count: int,
    hints_used: int,
    cfg: ScoringConfig = ScoringConfig(),
) -> float:
    """
    Final episode score in [0, 1].

    Full solution at minimum steps = 1.0
    Partial solutions penalised by steps and hints.
    """
    if tests_total == 0:
        return 0.0

    base = tests_passed / tests_total

    # Step efficiency: solved in fewer steps → higher score
    # Normalised to [0, 1]: 1.0 if solved in 1 step, 0.5 at max_steps/2
    step_factor = math.exp(-cfg.step_penalty * max(0, step_count - 1))

    hint_deduction = hints_used * cfg.hint_penalty

    score = base * step_factor - hint_deduction
    return float(max(0.0, min(1.0, score)))


def compute_diff_score(original: str, current: str) -> float:
    """
    Compute how much the code has changed from the original.
    Returns 0.0 (unchanged) to 1.0 (completely different).
    Used as a signal to detect agent activity.
    """
    orig_lines = set(original.strip().splitlines())
    curr_lines = set(current.strip().splitlines())
    if not orig_lines:
        return 0.0
    jaccard = len(orig_lines & curr_lines) / len(orig_lines | curr_lines)
    return 1.0 - jaccard


# ─────────────────────────────────────────────
# Neural Reward Model (PyTorch)
# ─────────────────────────────────────────────


class RewardMLP(nn.Module):
    """
    Learned reward model for CodeFix-Env.

    Input features (8-dim):
      [tests_passed_ratio, step_ratio, hints_used, diff_score,
       prev_score, action_type_id, code_len_ratio, test_count]

    Output: scalar reward in [0, 1] (sigmoid)

    Train with human feedback or RL traces to replace/augment rule-based reward.
    """

    INPUT_DIM = 8
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.OUTPUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, INPUT_DIM) → (batch, 1) rewards"""
        return self.net(x)

    def predict(
        self,
        tests_passed_ratio: float,
        step_ratio: float,
        hints_used: int,
        diff_score: float,
        prev_score: float,
        action_type_id: int,
        code_len_ratio: float,
        test_count: int,
    ) -> float:
        """Convenience: predict a single reward scalar."""
        features = torch.tensor(
            [
                [
                    tests_passed_ratio,
                    step_ratio,
                    float(hints_used),
                    diff_score,
                    prev_score,
                    float(action_type_id) / 7.0,  # normalise 0..7
                    code_len_ratio,
                    float(test_count) / 10.0,
                ]
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            return self.forward(features).item()


@dataclass
class EpisodeMetrics:
    """Summary metrics for a completed episode — logged to W&B / stdout."""

    task_id: str
    solved: bool
    final_score: float
    total_steps: int
    hints_used: int
    tests_passed: int
    tests_total: int
    total_reward: float
    efficiency: float = 0.0  # solved? then steps_used / max_steps, else 0

    def __post_init__(self):
        if self.solved and self.total_steps > 0:
            self.efficiency = 1.0 - (self.total_steps / 20.0)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "solved": self.solved,
            "final_score": round(self.final_score, 4),
            "total_steps": self.total_steps,
            "hints_used": self.hints_used,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "total_reward": round(self.total_reward, 4),
            "efficiency": round(self.efficiency, 4),
        }
