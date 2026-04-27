"""
Tests for reward computation — rule-based and neural model.
"""

from __future__ import annotations

import torch

from codefix_env.models import ActionType, BugCategory, CodeFixObservation, Difficulty
from codefix_env.rewards import RewardPipeline
from codefix_env.utils.metrics import (
    EpisodeMetrics,
    RewardMLP,
    ScoringConfig,
    compute_diff_score,
    compute_final_score,
    compute_shaped_reward,
    compute_test_score,
)

# ── compute_test_score ────────────────────────────────────────────────────────


class TestComputeTestScore:
    def test_all_pass(self):
        assert compute_test_score(4, 4) == 1.0

    def test_none_pass(self):
        assert compute_test_score(0, 4) == 0.0

    def test_partial(self):
        assert compute_test_score(2, 4) == 0.5

    def test_zero_total(self):
        assert compute_test_score(0, 0) == 0.0


# ── compute_shaped_reward ─────────────────────────────────────────────────────


class TestShapedReward:
    def test_progress_gives_positive_reward(self):
        r = compute_shaped_reward(
            prev_passed=0,
            curr_passed=2,
            tests_total=4,
            step_count=1,
            hints_used=0,
            action_type="run_tests",
        )
        assert r > 0.0

    def test_regression_gives_negative_reward(self):
        r = compute_shaped_reward(
            prev_passed=3,
            curr_passed=1,
            tests_total=4,
            step_count=1,
            hints_used=0,
            action_type="run_tests",
        )
        assert r < 0.0

    def test_solve_bonus_applied(self):
        cfg = ScoringConfig(solve_bonus=0.5)
        r = compute_shaped_reward(
            prev_passed=3,
            curr_passed=4,
            tests_total=4,
            step_count=1,
            hints_used=0,
            action_type="submit_fix",
            cfg=cfg,
        )
        assert r > 0.5

    def test_hint_penalty_applied(self):
        cfg = ScoringConfig(hint_penalty=0.1)
        r_no_hint = compute_shaped_reward(
            prev_passed=2,
            curr_passed=2,
            tests_total=4,
            step_count=1,
            hints_used=0,
            action_type="run_tests",
            cfg=cfg,
        )
        r_with_hint = compute_shaped_reward(
            prev_passed=2,
            curr_passed=2,
            tests_total=4,
            step_count=1,
            hints_used=1,
            action_type="run_tests",
            cfg=cfg,
        )
        assert r_with_hint < r_no_hint

    def test_time_decay(self):
        """Later steps get smaller rewards for same progress."""
        cfg = ScoringConfig(time_decay_gamma=0.9)
        r_early = compute_shaped_reward(
            prev_passed=0,
            curr_passed=1,
            tests_total=4,
            step_count=1,
            hints_used=0,
            action_type="edit_line",
            cfg=cfg,
        )
        r_late = compute_shaped_reward(
            prev_passed=0,
            curr_passed=1,
            tests_total=4,
            step_count=15,
            hints_used=0,
            action_type="edit_line",
            cfg=cfg,
        )
        assert r_early > r_late


# ── compute_final_score ───────────────────────────────────────────────────────


class TestFinalScore:
    def test_perfect_score(self):
        s = compute_final_score(4, 4, step_count=1, hints_used=0)
        assert s > 0.9

    def test_zero_score_on_no_tests_pass(self):
        s = compute_final_score(0, 4, step_count=20, hints_used=5)
        assert s == 0.0

    def test_partial_score(self):
        s = compute_final_score(2, 4, step_count=5, hints_used=0)
        assert 0.0 < s < 0.6

    def test_score_clipped_to_one(self):
        s = compute_final_score(4, 4, step_count=1, hints_used=0)
        assert s <= 1.0

    def test_score_clipped_to_zero(self):
        s = compute_final_score(0, 4, step_count=20, hints_used=100)
        assert s >= 0.0

    def test_fewer_steps_better_score(self):
        s_fast = compute_final_score(4, 4, step_count=2, hints_used=0)
        s_slow = compute_final_score(4, 4, step_count=18, hints_used=0)
        assert s_fast > s_slow


# ── compute_diff_score ────────────────────────────────────────────────────────


class TestDiffScore:
    def test_identical_code_zero(self):
        code = "def foo():\n    return 1\n"
        assert compute_diff_score(code, code) == 0.0

    def test_completely_different_nonzero(self):
        orig = "def foo():\n    return 1\n"
        curr = "def bar():\n    x = 42\n    print(x)\n"
        score = compute_diff_score(orig, curr)
        assert score > 0.0

    def test_empty_original(self):
        score = compute_diff_score("", "some code")
        assert score == 0.0  # guarded division


# ── RewardMLP ─────────────────────────────────────────────────────────────────


class TestRewardMLP:
    def test_model_output_shape(self):
        model = RewardMLP()
        x = torch.rand(4, RewardMLP.INPUT_DIM)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_in_zero_one(self):
        model = RewardMLP()
        x = torch.rand(10, RewardMLP.INPUT_DIM)
        out = model(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_predict_returns_float(self):
        model = RewardMLP()
        r = model.predict(
            tests_passed_ratio=0.5,
            step_ratio=0.2,
            hints_used=0,
            diff_score=0.3,
            prev_score=0.4,
            action_type_id=1,
            code_len_ratio=1.0,
            test_count=4,
        )
        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0

    def test_model_gradients_flow(self):
        model = RewardMLP()
        x = torch.rand(2, RewardMLP.INPUT_DIM)
        out = model(x).sum()
        out.backward()
        for p in model.parameters():
            assert p.grad is not None


# ── RewardPipeline ────────────────────────────────────────────────────────────


def _make_obs(**kwargs) -> CodeFixObservation:
    defaults = dict(
        current_code="def f(): pass",
        original_code="def f(): pass",
        tests_passed=0,
        tests_total=4,
        score_so_far=0.0,
        step_count=1,
        max_steps=20,
        steps_remaining=19,
        task_id="easy-001-missing-return",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.RETURN_BUG,
        hints_used=0,
    )
    defaults.update(kwargs)
    return CodeFixObservation(**defaults)


class TestRewardPipeline:
    def test_step_reward_progress(self):
        pipeline = RewardPipeline()
        from codefix_env.tasks import load_task

        task = load_task("easy-001-missing-return")
        prev = _make_obs(tests_passed=0)
        curr = _make_obs(tests_passed=3)
        r = pipeline.step_reward(prev, curr, ActionType.EDIT_LINE, task)
        assert r > 0.0

    def test_episode_reward_solved(self):
        pipeline = RewardPipeline()
        from codefix_env.tasks import load_task

        task = load_task("easy-001-missing-return")
        obs = _make_obs(tests_passed=4, all_tests_pass=True)
        r = pipeline.episode_reward(obs, task)
        assert r > 0.5

    def test_no_neural_model_by_default(self):
        pipeline = RewardPipeline()
        assert not pipeline.uses_neural


# ── EpisodeMetrics ────────────────────────────────────────────────────────────


class TestEpisodeMetrics:
    def test_efficiency_when_solved(self):
        m = EpisodeMetrics(
            task_id="t1",
            solved=True,
            final_score=0.9,
            total_steps=5,
            hints_used=0,
            tests_passed=4,
            tests_total=4,
            total_reward=2.0,
        )
        assert m.efficiency > 0.0

    def test_efficiency_zero_when_not_solved(self):
        m = EpisodeMetrics(
            task_id="t1",
            solved=False,
            final_score=0.2,
            total_steps=20,
            hints_used=2,
            tests_passed=1,
            tests_total=4,
            total_reward=0.5,
        )
        assert m.efficiency == 0.0

    def test_to_dict_keys(self):
        m = EpisodeMetrics(
            task_id="t1",
            solved=True,
            final_score=1.0,
            total_steps=3,
            hints_used=0,
            tests_passed=4,
            tests_total=4,
            total_reward=3.0,
        )
        d = m.to_dict()
        expected_keys = {
            "task_id",
            "solved",
            "final_score",
            "total_steps",
            "hints_used",
            "tests_passed",
            "tests_total",
            "total_reward",
            "efficiency",
        }
        assert expected_keys == set(d.keys())
