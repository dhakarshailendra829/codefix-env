"""
CodeFix-Env Data Models
=======================
All Pydantic v2 models for Actions, Observations, State, and Tasks.
Strongly typed — no `Any`, no raw dicts.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────


class ActionType(str, Enum):
    RUN_TESTS = "run_tests"
    EDIT_LINE = "edit_line"
    INSERT_LINE = "insert_line"
    DELETE_LINE = "delete_line"
    GET_HINT = "get_hint"
    SUBMIT_FIX = "submit_fix"
    RESET = "reset"
    VIEW_CODE = "view_code"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BugCategory(str, Enum):
    SYNTAX = "syntax_error"
    LOGIC = "logic_bug"
    OFF_BY_ONE = "off_by_one"
    TYPE_ERROR = "type_error"
    MISSING_IMPORT = "missing_import"
    WRONG_OPERATOR = "wrong_operator"
    RECURSION_BUG = "recursion_bug"
    ALGORITHM_BUG = "algorithm_bug"
    MULTI_FUNCTION = "multi_function"
    RETURN_BUG = "return_bug"
    INDEX_ERROR = "index_error"
    SCOPE_BUG = "scope_bug"


class TerminationReason(str, Enum):
    SOLVED = "solved"
    MAX_STEPS = "max_steps_reached"
    SUBMITTED = "submitted"
    TIMEOUT = "timeout"


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────


class CodeFixAction(BaseModel):
    """
    An action taken by the agent in the CodeFix environment.

    Examples::

        CodeFixAction(action_type=ActionType.EDIT_LINE, line_number=5, new_content="    return x + 1")
        CodeFixAction(action_type=ActionType.RUN_TESTS)
        CodeFixAction(action_type=ActionType.SUBMIT_FIX)
    """

    action_type: ActionType = Field(..., description="Type of action to perform")
    line_number: Optional[int] = Field(
        None, ge=1, description="Target line (1-indexed, for edit/insert/delete)"
    )
    new_content: Optional[str] = Field(None, description="New line content (for edit/insert)")
    reasoning: Optional[str] = Field(
        None, description="Agent's chain-of-thought (logged, not used by env)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("line_number")
    @classmethod
    def line_required_for_edits(cls, v: Optional[int], info) -> Optional[int]:
        action = info.data.get("action_type")
        if action in (ActionType.EDIT_LINE, ActionType.INSERT_LINE, ActionType.DELETE_LINE):
            if v is None:
                raise ValueError(f"line_number is required for action_type={action}")
        return v

    @field_validator("new_content")
    @classmethod
    def content_required_for_edits(cls, v: Optional[str], info) -> Optional[str]:
        action = info.data.get("action_type")
        if action in (ActionType.EDIT_LINE, ActionType.INSERT_LINE):
            if v is None:
                raise ValueError(f"new_content is required for action_type={action}")
        return v

    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────


class TestResult(BaseModel):
    """Result of a single test case."""

    name: str
    passed: bool
    output: str = ""
    error: str = ""
    runtime_ms: float = 0.0


class CodeFixObservation(BaseModel):
    """
    Full observation returned after every env.step() or env.reset().
    Contains everything the agent needs to make the next decision.
    """

    # Code state
    current_code: str = Field(..., description="Current (possibly edited) source code")
    original_code: str = Field(..., description="Original buggy code (never changes)")
    diff: str = Field("", description="Unified diff: original → current")

    # Test feedback
    test_results: list[TestResult] = Field(default_factory=list)
    test_output: str = Field("", description="Raw stdout/stderr from test runner")
    tests_passed: int = 0
    tests_total: int = 0
    all_tests_pass: bool = False

    # Scoring
    score_so_far: float = Field(0.0, ge=0.0, le=1.0, description="Cumulative normalised score")
    shaped_reward: float = Field(0.0, description="Dense reward for this step")

    # Episode metadata
    step_count: int = 0
    max_steps: int = 20
    steps_remaining: int = 20
    done: bool = False
    termination_reason: Optional[TerminationReason] = None

    # Task info
    task_id: str = ""
    difficulty: Difficulty = Difficulty.MEDIUM
    bug_category: BugCategory = BugCategory.LOGIC
    hint_available: bool = True
    hints_used: int = 0

    # Agent feedback
    feedback: str = ""
    error_message: str = ""


# ─────────────────────────────────────────────
# Episode State
# ─────────────────────────────────────────────


class CodeFixState(BaseModel):
    """Internal episode state (returned by env.state())."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 20
    total_reward: float = 0.0
    action_history: list[dict] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    done: bool = False
    solved: bool = False
    hints_used: int = 0


# ─────────────────────────────────────────────
# Step Result (wraps observation)
# ─────────────────────────────────────────────


class StepResult(BaseModel):
    """Full result returned by env.step()."""

    observation: CodeFixObservation
    reward: float
    done: bool
    truncated: bool = False
    info: dict = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Task Definition
# ─────────────────────────────────────────────


class TestCase(BaseModel):
    """A single test case for a task."""

    name: str
    code: str = Field(..., description="Test code (function call + assertion)")
    expected: str = ""
    timeout_s: float = 5.0


class Task(BaseModel):
    """A complete debugging task."""

    id: str
    title: str
    description: str
    buggy_code: str
    solution_code: str
    test_cases: list[TestCase]
    difficulty: Difficulty
    bug_category: BugCategory
    tags: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    max_steps: int = 20
    hint_penalty: float = 0.05
    time_penalty_per_step: float = 0.01
    author: str = "codefix-env"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def num_tests(self) -> int:
        return len(self.test_cases)
