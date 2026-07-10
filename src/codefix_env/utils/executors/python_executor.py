"""
Python Executor
================
Wraps the existing, already-tested sandbox.run_code() behind the
LanguageExecutor interface. A thin adapter, not a reimplementation —
sandbox.py's AST allow-list, restricted builtins, and resource limits
are unchanged and continue to be exercised by the existing test suite.
"""

from __future__ import annotations

from codefix_env.utils.executors.base import LanguageExecutor
from codefix_env.utils.result_types import ExecutionResult


class PythonExecutor(LanguageExecutor):
    language_id = "python"

    def run_test(self, source_code: str, test_code: str, timeout_s: float) -> ExecutionResult:
        from codefix_env.utils.sandbox import run_code

        return run_code(source_code, test_code, timeout_s=timeout_s)
