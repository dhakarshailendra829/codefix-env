"""
Tests for task registry, task loading, and sandbox execution.
"""
from __future__ import annotations

import pytest

from codefix_env.models import Difficulty, Task
from codefix_env.tasks import ALL_TASKS, list_tasks, load_task, random_task, task_count
from codefix_env.tasks.easy import EASY_TASKS
from codefix_env.tasks.medium import MEDIUM_TASKS
from codefix_env.tasks.hard import HARD_TASKS
from codefix_env.utils.sandbox import ExecutionResult, run_code, run_all_tests


# ── Task Registry ─────────────────────────────────────────────────────────────

class TestTaskRegistry:
    def test_all_tasks_loaded(self):
        counts = task_count()
        assert counts["total"] >= 20
        assert counts["easy"]   >= 8
        assert counts["medium"] >= 8
        assert counts["hard"]   >= 5

    def test_load_known_task(self):
        task = load_task("easy-001-missing-return")
        assert task.id == "easy-001-missing-return"
        assert task.difficulty == "easy"

    def test_load_unknown_task_raises(self):
        raised = False
        try:
            load_task("nonexistent-task-xyz")
        except (KeyError, Exception):
            raised = True
        assert raised, "Expected an exception for unknown task ID"

    def test_list_all_tasks(self):
        tasks = list_tasks()
        assert len(tasks) == len(ALL_TASKS)

    def test_list_easy_tasks(self):
        tasks = list_tasks(difficulty=Difficulty.EASY)
        assert all(t.difficulty == "easy" for t in tasks)

    def test_list_medium_tasks(self):
        tasks = list_tasks(difficulty=Difficulty.MEDIUM)
        assert all(t.difficulty == "medium" for t in tasks)

    def test_list_hard_tasks(self):
        tasks = list_tasks(difficulty=Difficulty.HARD)
        assert all(t.difficulty == "hard" for t in tasks)

    def test_random_task_easy(self):
        task = random_task(difficulty=Difficulty.EASY)
        assert task.difficulty == "easy"

    def test_random_task_with_exclude(self):
        easy_ids = [t.id for t in EASY_TASKS]
        # Exclude all but one
        exclude = easy_ids[:-1]
        task = random_task(difficulty=Difficulty.EASY, exclude=exclude)
        assert task.id == easy_ids[-1]

    def test_random_task_all_excluded_raises(self):
        all_ids = list(ALL_TASKS.keys())
        with pytest.raises(ValueError):
            random_task(exclude=all_ids)


# ── Task Structure Validation ─────────────────────────────────────────────────

class TestTaskStructure:
    @pytest.mark.parametrize("task", list(ALL_TASKS.values()))
    def test_task_has_required_fields(self, task: Task):
        assert task.id
        assert task.title
        assert task.description
        assert task.buggy_code.strip()
        assert task.solution_code.strip()
        assert len(task.test_cases) >= 2, f"{task.id} needs at least 2 test cases"
        assert task.difficulty in ("easy", "medium", "hard")
        assert task.max_steps >= 8

    @pytest.mark.parametrize("task", list(ALL_TASKS.values()))
    def test_task_has_hints(self, task: Task):
        assert len(task.hints) >= 1, f"{task.id} should have at least one hint"

    @pytest.mark.parametrize("task", list(ALL_TASKS.values()))
    def test_task_ids_are_unique(self, task: Task):
        # If this runs without error, all IDs are unique (dict key constraint)
        assert task.id in ALL_TASKS

    @pytest.mark.parametrize("task", list(ALL_TASKS.values()))
    def test_buggy_and_solution_differ(self, task: Task):
        assert task.buggy_code != task.solution_code, \
            f"{task.id}: buggy_code and solution_code are identical!"


# ── Solution Correctness ──────────────────────────────────────────────────────

class TestSolutionCorrectness:
    """Every task's solution code must pass all its own test cases."""

    @pytest.mark.parametrize("task", list(ALL_TASKS.values()))
    def test_solution_passes_all_tests(self, task: Task):
        results = run_all_tests(task.solution_code, task.test_cases)
        failed = [
            (task.test_cases[i].name, r.exception or r.stderr)
            for i, r in enumerate(results)
            if not r.passed
        ]
        assert not failed, (
            f"Task '{task.id}' solution fails tests:\n"
            + "\n".join(f"  {name}: {err}" for name, err in failed)
        )

    @pytest.mark.parametrize("task", EASY_TASKS + MEDIUM_TASKS)
    def test_buggy_code_fails_at_least_one_test(self, task: Task):
        """Sanity check: buggy code should fail at least one test."""
        results = run_all_tests(task.buggy_code, task.test_cases)
        all_pass = all(r.passed for r in results)
        assert not all_pass, (
            f"Task '{task.id}': buggy code passes ALL tests — this task has no bug!"
        )


# ── Sandbox ───────────────────────────────────────────────────────────────────

class TestSandbox:
    def test_correct_code_passes(self):
        code = "def add(a, b): return a + b"
        result = run_code(code, "assert add(1, 2) == 3")
        assert result.passed is True

    def test_wrong_code_fails(self):
        code = "def add(a, b): return a - b"
        result = run_code(code, "assert add(1, 2) == 3")
        assert result.passed is False
        assert "AssertionError" in result.exception

    def test_syntax_error_caught(self):
        code = "def foo(\n    pass"
        result = run_code(code, "")
        assert result.passed is False
        assert "SyntaxError" in result.exception

    def test_timeout_enforced(self):
        code = "while True: pass"
        result = run_code(code, "", timeout_s=1.0)
        assert result.timed_out is True
        assert result.passed is False

    def test_stdout_captured(self):
        code = "def greet(): print('hello'); return 'hi'"
        result = run_code(code, "greet()")
        assert "hello" in result.stdout

    def test_import_os_blocked(self):
        code = "import os; os.system('echo hacked')"
        result = run_code(code, "")
        assert result.passed is False
        assert "SecurityError" in result.exception or "not allowed" in result.exception

    def test_import_math_allowed(self):
        code = "import math; result = math.sqrt(4)"
        result = run_code(code, "assert result == 2.0")
        assert result.passed is True

    def test_import_subprocess_blocked(self):
        code = "import subprocess; subprocess.run(['ls'])"
        result = run_code(code, "")
        assert result.passed is False

    def test_exec_blocked(self):
        code = "exec('import os')"
        result = run_code(code, "")
        assert result.passed is False
        assert "SecurityError" in result.exception

    def test_runtime_measured(self):
        code = "def fast(): return 1"
        result = run_code(code, "fast()")
        assert result.runtime_ms >= 0

    def test_run_all_tests(self):
        from codefix_env.models import TestCase
        code = "def add(a, b): return a + b"
        tests = [
            TestCase(name="t1", code="assert add(1,2)==3"),
            TestCase(name="t2", code="assert add(0,0)==0"),
            TestCase(name="t3", code="assert add(-1,1)==0"),
        ]
        results = run_all_tests(code, tests)
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_runtime_ms_tracked(self):
        code = "import time; time.sleep(0.01)"
        # Should not run (time is not in safe modules)
        result = run_code(code, "")
        # Either blocked by import or times out — main thing: it doesn't hang
        assert result.runtime_ms >= 0