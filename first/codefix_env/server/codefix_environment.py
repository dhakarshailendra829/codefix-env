import uuid
import traceback
from typing import Dict, Any, Optional, List
from openenv.core.env_server import Environment
from pydantic import BaseModel
from codefix_env.models import CodeFixAction, CodeFixObservation, CodeFixState


class CodeFixEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.episode_id: Optional[str] = None
        self._state: Optional[CodeFixState] = None
        self.current_task: Optional[str] = None
        self.tasks = {
            "easy-fix-syntax": self._create_easy_task(),
            "medium-algorithm": self._create_medium_task(),
            "hard-multi-function": self._create_hard_task(),
        }

    def _create_easy_task(self):
        return {
            "name": "easy-fix-syntax",
            "original_code": "def add(a, b):\n    return a + b",
            "test_cases": [
                {"input": (3, 5), "expected": 8},
                {"input": (0, 0), "expected": 0},
            ],
            "description": "Fix simple syntax error in add function",
        }

    def _create_medium_task(self):
        return {
            "name": "medium-algorithm",
            "original_code": "def find_max(nums):\n    max_val = 0\n    for n in nums:\n        if n > max_val:\n            max_val = n\n    return max_val",
            "test_cases": [
                {"input": [1, 3, 2], "expected": 3},
                {"input": [-5, -1, -10], "expected": -1},
            ],
            "description": "Fix bug in finding maximum (handles negatives)",
        }

    def _create_hard_task(self):
        return {
            "name": "hard-multi-function",
            "original_code": 'def process_data(data):\n    total = sum(data)\n    avg = total / len(data)\n    return {"total": total, "avg": avg}\n\ndef filter_positive(nums):\n    return [n for n in nums if n > 0]',
            "test_cases": [
                {"input": [1, 2, 3], "expected": {"total": 6, "avg": 2.0}},
            ],
            "description": "Fix bugs across multiple functions",
        }

    def reset(self, task_name: Optional[str] = None, **kwargs):
        if task_name is None or task_name not in self.tasks:
            task_name = "easy-fix-syntax"
        self.episode_id = str(uuid.uuid4())
        self.current_task = task_name
        task_data = self.tasks[task_name]
        self._state = CodeFixState(
            task_name=task_name,
            original_code=task_data["original_code"],
            current_code=task_data["original_code"],
            test_cases=task_data["test_cases"],
            hidden_tests_passed=0,
            steps_taken=0,
            max_steps=12,
        )
        obs = CodeFixObservation(
            current_code=self._state.current_code,
            test_output="No tests run yet. Call run_tests to begin.",
            error_message=None,
            score_so_far=0.0,
            feedback=f"Task: {task_data['description']}. Available actions: run_tests, edit_line, submit_fix.",
            task_name=task_name,
            reward=0.0,
            done=False,
        )
        return obs

    def step(self, action: CodeFixAction):
        if self._state is None:
            raise ValueError("Call reset() first")
        self._state.steps_taken += 1
        reward = 0.0
        feedback = ""
        error_message = None
        test_output = "No tests run"
        try:
            if action.action_type == "run_tests":
                test_output, passed, score = self._run_tests()
                feedback = (
                    f"Passed {passed}/{len(self._state.test_cases)} tests. Score: {score:.2f}"
                )
                reward = score * 0.3
            elif (
                action.action_type == "edit_line"
                and action.line_number is not None
                and action.new_line is not None
            ):
                lines = self._state.current_code.splitlines()
                if 0 <= action.line_number < len(lines):
                    lines[action.line_number] = action.new_line.rstrip()
                    self._state.current_code = "\n".join(lines)
                    reward = 0.15
                    feedback = f"Line {action.line_number} updated."
                else:
                    error_message = "Invalid line number"
            elif action.action_type == "submit_fix":
                if action.code:
                    self._state.current_code = action.code
                test_output, passed, score = self._run_tests()
                reward = score
                feedback = f"Submitted fix. Final test score: {score:.2f}"
                self._state.steps_taken = self._state.max_steps
            else:
                feedback = "Action received."
        except Exception as e:
            error_message = str(e)[:200]
            reward = -0.05
        reward = max(-0.1, min(reward, 1.0))
        obs = CodeFixObservation(
            current_code=self._state.current_code,
            test_output=test_output,
            error_message=error_message,
            score_so_far=reward,
            feedback=feedback,
            task_name=self._state.task_name,
            reward=reward,
            done=(self._state.steps_taken >= self._state.max_steps or reward >= 0.9),
        )
        return obs

    def _run_tests(self):
        try:
            safe_globals = {"__builtins__": {"sum": sum, "len": len}}
            exec(self._state.current_code, safe_globals)
            func_name = None
            for name in list(safe_globals.keys()):
                if callable(safe_globals.get(name)) and not name.startswith("_"):
                    func_name = name
                    break
            if not func_name:
                return "No function defined", 0, 0.0
            passed = 0
            output = ""
            for i, test in enumerate(self._state.test_cases):
                try:
                    inp = test["input"]
                    result = (
                        safe_globals[func_name](*inp)
                        if isinstance(inp, (list, tuple))
                        else safe_globals[func_name](inp)
                    )
                    if result == test.get("expected"):
                        passed += 1
                        output += f"Test {i+1}: PASS\n"
                    else:
                        output += f"Test {i+1}: FAIL (got {result})\n"
                except Exception as e:
                    output += f"Test {i+1}: ERROR - {str(e)}\n"
            score = passed / len(self._state.test_cases) if self._state.test_cases else 0.0
            return output.strip() or "All tests executed", passed, score
        except Exception as e:
            return f"Execution failed: {str(e)}", 0, 0.0

    @property
    def state(self):
        if self._state is None:
            raise ValueError("No active episode. Call reset first.")
        return self._state
