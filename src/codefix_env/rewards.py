"""
Reward Shaping Pipeline
========================
High-level reward computation that combines:
1. Rule-based rewards (always active)
2. Neural reward model (optional, loaded from checkpoint)
3. Reward normalisation across an episode

Used by the environment server on every step.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from codefix_env.models import ActionType, CodeFixObservation, Task
from codefix_env.utils.metrics import (
    EpisodeMetrics,
    RewardMLP,
    ScoringConfig,
    compute_diff_score,
    compute_final_score,
    compute_shaped_reward,
)

logger = logging.getLogger(__name__)


class RewardPipeline:
    """
    Combines rule-based reward with an optional neural reward model.

    Usage::

        pipeline = RewardPipeline(cfg=ScoringConfig())
        reward = pipeline.step_reward(prev_obs, curr_obs, action_type, task)
        final  = pipeline.episode_reward(curr_obs, task)
    """

    def __init__(
        self,
        cfg: ScoringConfig = ScoringConfig(),
        neural_model_path: Optional[str] = None,
        neural_weight: float = 0.3,
    ):
        self.cfg = cfg
        self.neural_weight = neural_weight
        self._neural_model: Optional[RewardMLP] = None

        # Load pre-trained neural reward model if provided
        if neural_model_path and Path(neural_model_path).exists():
            try:
                self._neural_model = RewardMLP()
                state = torch.load(neural_model_path, map_location="cpu", weights_only=True)
                self._neural_model.load_state_dict(state)
                self._neural_model.eval()
                logger.info("Neural reward model loaded from %s", neural_model_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load neural reward model: %s", e)
                self._neural_model = None

    @property
    def uses_neural(self) -> bool:
        return self._neural_model is not None

    def step_reward(
        self,
        prev_obs: CodeFixObservation,
        curr_obs: CodeFixObservation,
        action_type: ActionType,
        task: Task,
    ) -> float:
        """
        Compute dense reward for a single step.
        Blend of rule-based + optional neural signal.
        """
        rule_reward = compute_shaped_reward(
            prev_passed=prev_obs.tests_passed,
            curr_passed=curr_obs.tests_passed,
            tests_total=curr_obs.tests_total,
            step_count=curr_obs.step_count,
            hints_used=curr_obs.hints_used,
            action_type=action_type.value if hasattr(action_type, "value") else str(action_type),
            cfg=self.cfg,
        )

        if self._neural_model is None:
            return rule_reward

        # Neural component
        diff_score = compute_diff_score(curr_obs.original_code, curr_obs.current_code)
        action_id = list(ActionType).index(action_type) if action_type in list(ActionType) else 0
        neural_reward = self._neural_model.predict(
            tests_passed_ratio=curr_obs.tests_passed / max(curr_obs.tests_total, 1),
            step_ratio=curr_obs.step_count / max(task.max_steps, 1),
            hints_used=curr_obs.hints_used,
            diff_score=diff_score,
            prev_score=prev_obs.score_so_far,
            action_type_id=action_id,
            code_len_ratio=len(curr_obs.current_code) / max(len(curr_obs.original_code), 1),
            test_count=curr_obs.tests_total,
        )

        # Weighted blend: mostly rules, neural provides nuance
        blended = (1.0 - self.neural_weight) * rule_reward + self.neural_weight * neural_reward
        return float(blended)

    def episode_reward(
        self,
        final_obs: CodeFixObservation,
        task: Task,
    ) -> float:
        """Final episode reward on termination."""
        return compute_final_score(
            tests_passed=final_obs.tests_passed,
            tests_total=final_obs.tests_total,
            step_count=final_obs.step_count,
            hints_used=final_obs.hints_used,
            cfg=self.cfg,
        )

    def build_metrics(
        self,
        task: Task,
        final_obs: CodeFixObservation,
        total_reward: float,
    ) -> EpisodeMetrics:
        return EpisodeMetrics(
            task_id=task.id,
            solved=final_obs.all_tests_pass,
            final_score=final_obs.score_so_far,
            total_steps=final_obs.step_count,
            hints_used=final_obs.hints_used,
            tests_passed=final_obs.tests_passed,
            tests_total=final_obs.tests_total,
            total_reward=total_reward,
        )


# Singleton default pipeline (no neural model)
default_pipeline = RewardPipeline(
    cfg=ScoringConfig(),
    neural_model_path=os.environ.get("CODEFIX_REWARD_MODEL_PATH"),
)
