"""
Neural Reward Model (PyTorch)
==============================
Learned reward model for CodeFix-Env, kept in its own module so that
importing codefix_env's core (env, sandbox, tasks) never pays torch's
import cost. torch only loads when someone actually uses RewardMLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
