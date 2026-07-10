"""
C++ Executor
=============
Compiles the agent's C++ solution together with a hidden test snippet
into one translation unit, compiles with g++, runs the binary under a
timeout.

Isolation model — read before using this in an adversarial setting:
Python's sandbox can statically reject dangerous constructs via AST
inspection before anything runs. C++ has no equivalent: by the time you
have a compiled binary, static filtering is not meaningful the same
way, and adversarial native code can do far more than the restricted
Python sandbox. This executor relies on: (1) a textual pre-compile
denylist for obvious dangerous includes/calls — weaker than Python's
allow-list, stated as such; (2) a wall-clock timeout on compile and
run; (3) resource limits on POSIX via preexec_fn (no-op on Windows,
which falls back to the timeout alone). For genuinely adversarial C++
input, add container-level isolation (see SECURITY.md) — this is not
safe by default beyond what's described here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

from codefix_env.utils.executors.base import LanguageExecutor
from codefix_env.utils.result_types import ExecutionResult

_DENYLIST_TOKENS = [
    "#include <fstream>",
    "#include <cstdlib>",
    "#include <unistd.h>",
    "#include <sys/",
    "system(",
    "popen(",
    "exec(",
    "fork(",
    "remove(",
    "rename(",
]

_COMPILE_TIMEOUT_S = 10.0


def _preexec_limits():
    """POSIX-only resource limits for the compiled binary's process."""
    try:
        import resource

        def _set():
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

        return _set
    except ImportError:
        return None


class CppExecutor(LanguageExecutor):
    language_id = "cpp"

    def is_available(self) -> tuple[bool, str]:
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            return False, "No C++ compiler (g++ or clang++) found on PATH."
        return True, compiler

    def run_test(self, source_code: str, test_code: str, timeout_s: float) -> ExecutionResult:
        available, info = self.is_available()
        if not available:
            return ExecutionResult(exception=f"CompilerNotFound: {info}")

        combined = self._build_translation_unit(source_code, test_code)

        denylist_hit = self._check_denylist(combined)
        if denylist_hit:
            return ExecutionResult(
                exception=f"SecurityError: disallowed construct '{denylist_hit}'"
            )

        with tempfile.TemporaryDirectory(prefix="codefix_cpp_") as tmpdir:
            src_path = os.path.join(tmpdir, "solution.cpp")
            bin_path = os.path.join(tmpdir, "solution_bin")
            with open(src_path, "w") as f:
                f.write(combined)

            compiler = shutil.which("g++") or shutil.which("clang++")
            try:
                compile_proc = subprocess.run(
                    [compiler, "-std=c++17", "-O1", "-o", bin_path, src_path],
                    capture_output=True,
                    text=True,
                    timeout=_COMPILE_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(exception="CompileTimeout: compilation exceeded time limit")

            if compile_proc.returncode != 0:
                return ExecutionResult(exception="CompileError", stderr=compile_proc.stderr[:2000])

            preexec = _preexec_limits()
            start = time.perf_counter()
            try:
                run_proc = subprocess.run(
                    [bin_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    preexec_fn=preexec() if preexec else None,
                )
                runtime_ms = (time.perf_counter() - start) * 1000
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    timed_out=True,
                    exception=f"TimeoutError: exceeded {timeout_s}s",
                    runtime_ms=(time.perf_counter() - start) * 1000,
                )

            passed = run_proc.returncode == 0
            return ExecutionResult(
                stdout=run_proc.stdout[:2000],
                stderr=run_proc.stderr[:2000] if not passed else "",
                exception="" if passed else f"AssertionFailed (exit code {run_proc.returncode})",
                passed=passed,
                runtime_ms=runtime_ms,
            )

    @staticmethod
    def _check_denylist(code: str) -> str:
        for token in _DENYLIST_TOKENS:
            if token in code:
                return token
        return ""

    @staticmethod
    def _build_translation_unit(source_code: str, test_code: str) -> str:
        return f"""\
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>

{source_code}

int main() {{
{test_code}
    return 0;
}}
"""
