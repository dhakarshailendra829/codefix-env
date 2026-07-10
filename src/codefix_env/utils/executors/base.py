"""
Language Executor Interface
=============================
Every supported language (Python, C++, ...) implements this interface.
sandbox.py dispatches to the correct executor based on Task.language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from codefix_env.utils.result_types import ExecutionResult


class LanguageExecutor(ABC):
    """One executor instance handles one language. Must be stateless
    across calls — no shared mutable state between run_test calls."""

    language_id: str = ""

    @abstractmethod
    def run_test(self, source_code: str, test_code: str, timeout_s: float) -> ExecutionResult:
        """
        Execute `source_code` against `test_code`, inside whatever
        isolation this language requires, enforcing `timeout_s`.
        Must never raise — all failures are reported via the returned
        ExecutionResult so env.py never needs a try/except around this.
        """
        raise NotImplementedError

    def is_available(self) -> tuple[bool, str]:
        """Check whether this executor's toolchain is installed/usable.
        Returns (available, reason_if_not)."""
        return True, ""
