"""
Shared execution result type, used by every language executor and by
sandbox.py. Kept in its own module so executors/*.py can import it
without a circular import against sandbox.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of a single sandboxed test execution, for any language."""

    stdout: str = ""
    stderr: str = ""
    exception: str = ""
    passed: bool = False
    runtime_ms: float = 0.0
    timed_out: bool = False
