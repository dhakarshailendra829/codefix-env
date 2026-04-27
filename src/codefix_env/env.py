"""
CodeFixEnvironment — Core Server-Side Environment
===================================================
Implements the OpenEnv-compatible interface: reset(), step(), state().

Features:
- Full action validation
- Sandbox code execution per step
- Dense shaped rewards
- Diff tracking
- Hint system with penalty
- Episode state management
"""
from __future__ import annotations

import difflib
import logging
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Optional

import structlog

from codefix_env.models import (
    ActionType,
    BugCategory,
    CodeFixAction,
    CodeFixObservation,
    CodeFixState,
    Difficulty,
    StepResult,
    Task,
    TerminationReason,
    TestResult,
)
from codefix_env.rewards import RewardPipeline, default_pipeline
from codefix_env.tasks import load_task, random_task
from codefix_env.utils.metrics import compute_final_score, compute_test_score
from codefix_env.utils.sandbox import ExecutionResult, run_all_tests

logger = structlog.get_logger(__name__)


class CodeFixEnvironment:
    """
    Server-side RL environment for automated code debugging.

    Gymnasium-compatible interface:
        obs             = env.reset(task_id="easy-001-missing-return")
        result: StepResult = env.step(action)
        state:  CodeFixState = env.state()

    The environment is stateful (one episode at a time per instance).
    For parallel training, spawn one instance per worker.
    """

    def __init__(
        self,
        reward_pipeline: RewardPipeline = default_pipeline,
        default_difficulty: Difficulty  = Difficulty.MEDIUM,
        max_steps: int                  = 20,
    ):
        self.reward_pipeline    = reward_pipeline
        self.default_difficulty = default_difficulty
        self.default_max_steps  = max_steps

        # Episode state (initialised on reset)
        self._task:    Optional[Task]              = None
        self._state:   Optional[CodeFixState]      = None
        self._obs:     Optional[CodeFixObservation]= None
        self._current_code: str                    = ""
        self._prev_obs: Optional[CodeFixObservation] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id:    Optional[str]        = None,
        difficulty: Optional[Difficulty] = None,
        seed:       Optional[int]        = None,
    ) -> CodeFixObservation:
        """
        Start a new episode.

        Args:
            task_id:    Specific task to load. If None, random task is chosen.
            difficulty: Filter random task by difficulty.
            seed:       Random seed (for reproducibility).

        Returns:
            Initial CodeFixObservation.
        """
        if seed is not None:
            import random
            random.seed(seed)

        # Load task
        if task_id:
            self._task = load_task(task_id)
        else:
            self._task = random_task(difficulty=difficulty or self.default_difficulty)

        # Initialise episode
        # constructor max_steps is a hard cap: min(constructor_cap, task_default)
        # e.g. CodeFixEnvironment(max_steps=3) will always truncate at 3 steps
        task_max = self._task.max_steps or self.default_max_steps
        effective_max = min(self.default_max_steps, task_max)
        self._current_code = self._task.buggy_code
        self._state = CodeFixState(
            episode_id=str(uuid.uuid4()),
            task_id=self._task.id,
            max_steps=effective_max,
        )

        # Build initial observation (no tests run yet)
        self._obs = self._build_observation(
            tests_run=False,
            test_results=[],
            shaped_reward=0.0,
            feedback=f"Episode started. Task: '{self._task.title}'\n{self._task.description}",
        )
        self._prev_obs = deepcopy(self._obs)

        logger.info(
            "episode_reset",
            episode_id=self._state.episode_id,
            task_id=self._task.id,
            difficulty=self._task.difficulty,
        )
        return self._obs

    def step(self, action: CodeFixAction) -> StepResult:
        """
        Execute one action in the environment.

        Returns StepResult with (observation, reward, done, info).
        """
        if self._state is None or self._task is None:
            raise RuntimeError("Call reset() before step().")
        # Allow GET_HINT even on a done episode (graceful feedback, no crash)
        # use_enum_values=True means action_type is stored as plain string not enum member
        action_type_str = action.action_type if isinstance(action.action_type, str) else action.action_type.value
        if self._state.done and action_type_str != ActionType.GET_HINT.value:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._prev_obs = deepcopy(self._obs)
        self._state.step_count += 1

        # Dispatch action
        action_type = ActionType(action.action_type)
        try:
            obs, immediate_reward = self._dispatch(action_type, action)
        except Exception as e:  # noqa: BLE001
            obs = self._build_observation(
                tests_run=False,
                test_results=[],
                shaped_reward=0.0,
                feedback=f"⚠️  Error executing action: {e}",
                error_message=str(e),
            )
            immediate_reward = -0.05

        # Record in state
        self._state.action_history.append({
            "step": self._state.step_count,
            "action_type": action_type.value,
            "line_number": action.line_number,
            "reward": immediate_reward,
        })
        self._state.total_reward += immediate_reward

        # Update obs ref
        self._obs = obs

        # Check terminal conditions
        done, truncated, reason = self._check_terminal(obs)
        if done or truncated:
            self._state.done   = True
            self._state.solved = obs.all_tests_pass
            final_score = compute_final_score(
                obs.tests_passed, obs.tests_total,
                self._state.step_count, self._state.hints_used,
                self.reward_pipeline.cfg,
            )
            obs.score_so_far = final_score
            obs.done = True
            obs.termination_reason = reason
            metrics = self.reward_pipeline.build_metrics(self._task, obs, self._state.total_reward)
            logger.info("episode_done", **metrics.to_dict())

        result = StepResult(
            observation=obs,
            reward=immediate_reward,
            done=done or truncated,
            truncated=truncated,
            info={
                "episode_id": self._state.episode_id,
                "task_id": self._task.id,
                "total_reward": self._state.total_reward,
            },
        )
        return result

    def state(self) -> CodeFixState:
        """Return internal episode state (non-destructive)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return deepcopy(self._state)

    # ── Action Handlers ───────────────────────────────────────────────────────

    def _dispatch(
        self,
        action_type: ActionType,
        action: CodeFixAction,
    ) -> tuple[CodeFixObservation, float]:
        """Route action to its handler. Returns (obs, reward)."""

        if action_type == ActionType.RUN_TESTS:
            return self._action_run_tests()

        elif action_type == ActionType.EDIT_LINE:
            return self._action_edit_line(action.line_number, action.new_content or "")

        elif action_type == ActionType.INSERT_LINE:
            return self._action_insert_line(action.line_number, action.new_content or "")

        elif action_type == ActionType.DELETE_LINE:
            return self._action_delete_line(action.line_number)

        elif action_type == ActionType.GET_HINT:
            return self._action_get_hint()

        elif action_type == ActionType.SUBMIT_FIX:
            return self._action_submit()

        elif action_type == ActionType.VIEW_CODE:
            return self._action_view_code()

        else:
            return self._build_observation(
                tests_run=False,
                test_results=[],
                shaped_reward=0.0,
                feedback=f"Unknown action: {action_type}",
                error_message="UnknownAction",
            ), -0.02

    def _action_run_tests(self) -> tuple[CodeFixObservation, float]:
        results = run_all_tests(self._current_code, self._task.test_cases)
        test_results = self._parse_results(results)
        passed = sum(1 for r in test_results if r.passed)
        obs = self._build_observation(
            tests_run=True,
            test_results=test_results,
            shaped_reward=0.0,
            feedback=f"Tests run: {passed}/{len(test_results)} passing.",
        )
        reward = self.reward_pipeline.step_reward(
            self._prev_obs, obs, ActionType.RUN_TESTS, self._task
        )
        obs.shaped_reward = reward
        return obs, reward

    def _action_edit_line(self, line_num: int, new_content: str) -> tuple[CodeFixObservation, float]:
        lines = self._current_code.splitlines(keepends=True)
        if line_num < 1 or line_num > len(lines):
            obs = self._build_observation(
                tests_run=False, test_results=[],
                shaped_reward=-0.02,
                feedback=f"Invalid line number {line_num}. Code has {len(lines)} lines.",
                error_message="InvalidLineNumber",
            )
            return obs, -0.02

        # Preserve original line ending
        ending = "\n" if not new_content.endswith("\n") else ""
        lines[line_num - 1] = new_content + ending
        self._current_code = "".join(lines)

        obs = self._build_observation(
            tests_run=False, test_results=[],
            shaped_reward=0.0,
            feedback=f"Line {line_num} edited.",
        )
        reward = self.reward_pipeline.step_reward(
            self._prev_obs, obs, ActionType.EDIT_LINE, self._task
        )
        obs.shaped_reward = reward
        return obs, reward

    def _action_insert_line(self, after_line: int, content: str) -> tuple[CodeFixObservation, float]:
        lines = self._current_code.splitlines(keepends=True)
        idx   = min(after_line, len(lines))
        if not content.endswith("\n"):
            content += "\n"
        lines.insert(idx, content)
        self._current_code = "".join(lines)
        obs = self._build_observation(
            tests_run=False, test_results=[],
            shaped_reward=0.0,
            feedback=f"Line inserted after line {after_line}.",
        )
        reward = self.reward_pipeline.step_reward(
            self._prev_obs, obs, ActionType.INSERT_LINE, self._task
        )
        obs.shaped_reward = reward
        return obs, reward

    def _action_delete_line(self, line_num: int) -> tuple[CodeFixObservation, float]:
        lines = self._current_code.splitlines(keepends=True)
        if line_num < 1 or line_num > len(lines):
            obs = self._build_observation(
                tests_run=False, test_results=[], shaped_reward=-0.02,
                feedback=f"Invalid line number {line_num}.", error_message="InvalidLineNumber",
            )
            return obs, -0.02
        del lines[line_num - 1]
        self._current_code = "".join(lines)
        obs = self._build_observation(
            tests_run=False, test_results=[], shaped_reward=0.0,
            feedback=f"Line {line_num} deleted.",
        )
        reward = self.reward_pipeline.step_reward(
            self._prev_obs, obs, ActionType.DELETE_LINE, self._task
        )
        obs.shaped_reward = reward
        return obs, reward

    def _action_get_hint(self) -> tuple[CodeFixObservation, float]:
        hints_used = self._state.hints_used
        hints = self._task.hints
        if not hints:
            feedback = "No hints available for this task."
        elif hints_used < len(hints):
            feedback = f"💡 Hint {hints_used + 1}/{len(hints)}: {hints[hints_used]}"
            self._state.hints_used += 1
        else:
            feedback = "No more hints available."

        penalty = -self._task.hint_penalty if hints_used < len(hints) else 0.0
        obs = self._build_observation(
            tests_run=False, test_results=[], shaped_reward=penalty, feedback=feedback,
        )
        return obs, penalty

    def _action_submit(self) -> tuple[CodeFixObservation, float]:
        """Run all tests and compute final reward."""
        results = run_all_tests(self._current_code, self._task.test_cases)
        test_results = self._parse_results(results)
        passed = sum(1 for r in test_results if r.passed)

        obs = self._build_observation(
            tests_run=True, test_results=test_results, shaped_reward=0.0,
            feedback=(
                f"✅ Submitted! {passed}/{len(test_results)} tests passing."
                if passed == len(test_results)
                else f"❌ Submitted with {passed}/{len(test_results)} tests passing."
            ),
        )
        obs.done = True
        obs.termination_reason = TerminationReason.SUBMITTED

        reward = self.reward_pipeline.episode_reward(obs, self._task)
        obs.shaped_reward = reward
        obs.score_so_far  = reward
        self._state.done  = True
        return obs, reward

    def _action_view_code(self) -> tuple[CodeFixObservation, float]:
        obs = self._build_observation(
            tests_run=False, test_results=[], shaped_reward=0.0,
            feedback="Code viewed.",
        )
        return obs, 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        tests_run:    bool,
        test_results: list[TestResult],
        shaped_reward: float,
        feedback:     str = "",
        error_message: str = "",
    ) -> CodeFixObservation:
        """Construct a full CodeFixObservation from current state."""
        assert self._task and self._state

        passed      = sum(1 for r in test_results if r.passed) if tests_run else (
            self._obs.tests_passed if self._obs else 0
        )
        total       = len(self._task.test_cases)
        all_pass    = (passed == total and tests_run)
        score       = compute_test_score(passed, total) if tests_run else (
            self._obs.score_so_far if self._obs else 0.0
        )

        # Build unified diff
        diff = "".join(difflib.unified_diff(
            self._task.buggy_code.splitlines(keepends=True),
            self._current_code.splitlines(keepends=True),
            fromfile="original.py",
            tofile="current.py",
        ))

        return CodeFixObservation(
            current_code=self._current_code,
            original_code=self._task.buggy_code,
            diff=diff,
            test_results=test_results,
            test_output="\n".join(
                f"[{'✓' if r.passed else '✗'}] {r.name}: {r.output or r.error}"
                for r in test_results
            ) if test_results else "",
            tests_passed=passed,
            tests_total=total,
            all_tests_pass=all_pass,
            score_so_far=score,
            shaped_reward=shaped_reward,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            steps_remaining=max(0, self._state.max_steps - self._state.step_count),
            done=False,
            task_id=self._task.id,
            difficulty=self._task.difficulty,
            bug_category=self._task.bug_category,
            hints_used=self._state.hints_used,
            hint_available=self._state.hints_used < len(self._task.hints),
            feedback=feedback,
            error_message=error_message,
        )

    def _parse_results(self, exec_results: list[ExecutionResult]) -> list[TestResult]:
        """Convert raw ExecutionResult list → TestResult list."""
        assert self._task
        results = []
        for tc, er in zip(self._task.test_cases, exec_results):
            # Guard: er must be ExecutionResult not a list or other type
            if not isinstance(er, ExecutionResult):
                er = ExecutionResult(
                    passed=False,
                    exception=f"InternalError: expected ExecutionResult, got {type(er).__name__}",
                )
            results.append(TestResult(
                name=tc.name,
                passed=er.passed,
                output=er.stdout[:500] if er.stdout else "",
                error=(er.exception or er.stderr)[:500],
                runtime_ms=er.runtime_ms,
            ))
        return results

    def _check_terminal(
        self, obs: CodeFixObservation
    ) -> tuple[bool, bool, Optional[TerminationReason]]:
        """Returns (done, truncated, reason)."""
        if obs.all_tests_pass:
            return True, False, TerminationReason.SOLVED
        if obs.done:
            return True, False, TerminationReason.SUBMITTED
        if self._state.step_count >= self._state.max_steps:
            return False, True, TerminationReason.MAX_STEPS
        return False, False, None