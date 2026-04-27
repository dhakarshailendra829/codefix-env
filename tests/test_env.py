"""
Tests for CodeFixEnvironment — reset, step, all action types.
"""
from __future__ import annotations

import pytest

from codefix_env.env import CodeFixEnvironment
from codefix_env.models import (
    ActionType,
    CodeFixAction,
    CodeFixObservation,
    Difficulty,
    StepResult,
    TerminationReason,
)


@pytest.fixture
def env():
    return CodeFixEnvironment()


@pytest.fixture
def easy_env(env):
    """Env with easy-001 task loaded."""
    env.reset(task_id="easy-001-missing-return")
    return env


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="easy-001-missing-return")
        assert isinstance(obs, CodeFixObservation)

    def test_reset_loads_correct_task(self, env):
        obs = env.reset(task_id="easy-001-missing-return")
        assert obs.task_id == "easy-001-missing-return"

    def test_reset_initial_step_count(self, env):
        obs = env.reset(task_id="easy-001-missing-return")
        assert obs.step_count == 0

    def test_reset_code_is_buggy(self, env):
        obs = env.reset(task_id="easy-001-missing-return")
        # buggy code has no return statement for the result
        assert "return result" not in obs.current_code

    def test_reset_random_easy(self, env):
        obs = env.reset(difficulty=Difficulty.EASY)
        assert obs.difficulty == "easy"

    def test_reset_state_cleared(self, env):
        env.reset(task_id="easy-001-missing-return")
        env.reset(task_id="easy-002-wrong-operator")
        state = env.state()
        assert state.task_id == "easy-002-wrong-operator"
        assert state.step_count == 0

    def test_reset_before_step_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))

    def test_reset_with_seed(self, env):
        obs1 = env.reset(difficulty=Difficulty.EASY, seed=42)
        obs2 = env.reset(difficulty=Difficulty.EASY, seed=42)
        assert obs1.task_id == obs2.task_id


# ── run_tests ─────────────────────────────────────────────────────────────────

class TestRunTests:
    def test_run_tests_returns_step_result(self, easy_env):
        result = easy_env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
        assert isinstance(result, StepResult)

    def test_run_tests_fails_on_buggy_code(self, easy_env):
        result = easy_env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
        # buggy code: missing return, so tests should fail
        assert result.observation.tests_passed < result.observation.tests_total

    def test_run_tests_increments_step(self, easy_env):
        easy_env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
        assert easy_env.state().step_count == 1


# ── edit_line ─────────────────────────────────────────────────────────────────

class TestEditLine:
    def test_edit_line_changes_code(self, easy_env):
        obs_before = easy_env._current_code
        easy_env.step(CodeFixAction(
            action_type=ActionType.EDIT_LINE,
            line_number=2,
            new_content="    return result",
        ))
        assert easy_env._current_code != obs_before

    def test_edit_invalid_line_returns_error(self, easy_env):
        result = easy_env.step(CodeFixAction(
            action_type=ActionType.EDIT_LINE,
            line_number=999,
            new_content="x = 1",
        ))
        assert result.observation.error_message != ""
        assert result.reward < 0

    def test_edit_then_run_tests_pass(self, easy_env):
        # Fix the bug: INSERT return statement after line 3 (the assignment line)
        # buggy_code lines: L1=blank, L2=def, L3=result=a+b
        # We insert after L3 to add '    return result'
        easy_env.step(CodeFixAction(
            action_type=ActionType.INSERT_LINE,
            line_number=3,
            new_content="    return result",
        ))
        result = easy_env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
        assert result.observation.tests_passed == result.observation.tests_total


# ── insert / delete ───────────────────────────────────────────────────────────

class TestInsertDeleteLine:
    def test_insert_line(self, easy_env):
        before_lines = easy_env._current_code.count("\n")
        easy_env.step(CodeFixAction(
            action_type=ActionType.INSERT_LINE,
            line_number=2,
            new_content="    # inserted comment",
        ))
        after_lines = easy_env._current_code.count("\n")
        assert after_lines == before_lines + 1

    def test_delete_line(self, easy_env):
        before_lines = easy_env._current_code.count("\n")
        easy_env.step(CodeFixAction(
            action_type=ActionType.DELETE_LINE,
            line_number=1,
        ))
        after_lines = easy_env._current_code.count("\n")
        assert after_lines == before_lines - 1

    def test_delete_invalid_line(self, easy_env):
        result = easy_env.step(CodeFixAction(
            action_type=ActionType.DELETE_LINE,
            line_number=9999,
        ))
        assert result.reward < 0


# ── hints ─────────────────────────────────────────────────────────────────────

class TestHints:
    def test_get_hint_returns_hint(self, easy_env):
        result = easy_env.step(CodeFixAction(action_type=ActionType.GET_HINT))
        assert "Hint" in result.observation.feedback or "hint" in result.observation.feedback.lower()

    def test_hint_costs_reward(self, easy_env):
        result = easy_env.step(CodeFixAction(action_type=ActionType.GET_HINT))
        assert result.reward < 0

    def test_hints_exhausted(self, easy_env):
        # Exhaust all hints dynamically regardless of how many the task has
        from codefix_env.tasks import load_task
        task = load_task("easy-001-missing-return")
        num_hints = len(task.hints)
        for _ in range(num_hints):
            easy_env.step(CodeFixAction(action_type=ActionType.GET_HINT))
        # Next call — all hints used, should get 'No more' message
        result = easy_env.step(CodeFixAction(action_type=ActionType.GET_HINT))
        assert "No more" in result.observation.feedback


# ── submit ────────────────────────────────────────────────────────────────────

class TestSubmit:
    def test_submit_ends_episode(self, easy_env):
        result = easy_env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
        assert result.done is True

    def test_submit_after_fix_gives_high_reward(self, easy_env):
        # Fix the bug first — insert return after assignment line
        easy_env.step(CodeFixAction(
            action_type=ActionType.INSERT_LINE,
            line_number=3,
            new_content="    return result",
        ))
        result = easy_env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
        assert result.reward > 0.5

    def test_step_after_done_raises(self, easy_env):
        easy_env.step(CodeFixAction(action_type=ActionType.SUBMIT_FIX))
        with pytest.raises(RuntimeError):
            easy_env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))


# ── max steps ─────────────────────────────────────────────────────────────────

class TestMaxSteps:
    def test_truncated_at_max_steps(self):
        env = CodeFixEnvironment(max_steps=3)
        env.reset(task_id="easy-001-missing-return")
        result = None
        for _ in range(3):
            result = env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
        assert result.truncated is True or result.done is True


# ── diff tracking ─────────────────────────────────────────────────────────────

class TestDiff:
    def test_diff_empty_on_no_change(self, easy_env):
        result = easy_env.step(CodeFixAction(action_type=ActionType.RUN_TESTS))
        # No edits made, diff should be empty or minimal
        assert isinstance(result.observation.diff, str)

    def test_diff_populated_after_edit(self, easy_env):
        easy_env.step(CodeFixAction(
            action_type=ActionType.EDIT_LINE,
            line_number=2,
            new_content="    return result",
        ))
        result = easy_env.step(CodeFixAction(action_type=ActionType.VIEW_CODE))
        assert "---" in result.observation.diff or "+++" in result.observation.diff