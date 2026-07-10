"""
Executor Registry
==================
Single source of truth mapping Task.language -> LanguageExecutor.
Adding a new language: write executors/<lang>_executor.py implementing
LanguageExecutor, then add one line here.
"""

from __future__ import annotations

from codefix_env.utils.executors.base import LanguageExecutor
from codefix_env.utils.executors.cpp_executor import CppExecutor
from codefix_env.utils.executors.python_executor import PythonExecutor

_REGISTRY: dict[str, LanguageExecutor] = {
    "python": PythonExecutor(),
    "cpp": CppExecutor(),
}


def get_executor(language: str) -> LanguageExecutor:
    executor = _REGISTRY.get(language)
    if executor is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"No executor registered for language={language!r}. Supported: {supported}"
        )
    return executor


def supported_languages() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["LanguageExecutor", "get_executor", "supported_languages"]
