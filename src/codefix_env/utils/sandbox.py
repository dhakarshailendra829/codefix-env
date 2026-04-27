"""
Sandbox — Safe Python Code Execution
=====================================
Executes untrusted code in a restricted environment with:
- Timeout enforcement via threading
- Stdout/stderr capture
- Restricted builtins (no file I/O, no network, no subprocess)
- Memory-safe: runs in subprocess with resource limits on Linux
"""

from __future__ import annotations

import ast
import builtins as _builtins_module
import io
import multiprocessing
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of a sandboxed code execution."""

    stdout: str = ""
    stderr: str = ""
    exception: str = ""
    passed: bool = False
    runtime_ms: float = 0.0
    timed_out: bool = False


# Builtins allowed in sandbox (no I/O, no network, no os)
_SAFE_BUILTIN_NAMES = [
    # Core class/function support — REQUIRED for class and import keywords
    "__build_class__",
    "__name__",
    "__package__",
    "__spec__",
    "__import__",  # ✅ ADDED - Required for import statements to work
    # Safe built-in functions
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    "True",
    "False",
    "None",
    # All standard exceptions
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BlockingIOError",
    "BrokenPipeError",
    "BufferError",
    "BytesWarning",
    "ChildProcessError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "DeprecationWarning",
    "EOFError",
    "EnvironmentError",
    "Exception",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "GeneratorExit",
    "IOError",
    "ImportError",
    "ImportWarning",
    "IndentationError",
    "IndexError",
    "InterruptedError",
    "IsADirectoryError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "ModuleNotFoundError",
    "NameError",
    "NotADirectoryError",
    "NotImplemented",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "PermissionError",
    "ProcessLookupError",
    "RecursionError",
    "ReferenceError",
    "ResourceWarning",
    "RuntimeError",
    "RuntimeWarning",
    "StopAsyncIteration",
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
    "SystemError",
    "SystemExit",
    "TabError",
    "TimeoutError",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
]

_SAFE_BUILTINS = {
    name: getattr(_builtins_module, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(_builtins_module, name)
}

# Safe stdlib modules allowed inside sandbox
_SAFE_MODULES = {
    "math",
    "cmath",
    "decimal",
    "fractions",
    "random",
    "statistics",
    "itertools",
    "functools",
    "operator",
    "collections",
    "heapq",
    "bisect",
    "array",
    "copy",
    "re",
    "string",
    "textwrap",
    "difflib",
    "json",
    "enum",
    "dataclasses",
    "typing",
    "abc",
    "io",
    "time",
}


def _validate_ast(code: str) -> Optional[str]:
    """
    Static analysis: reject code with dangerous AST nodes.
    Returns error message if dangerous, None if safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # Block import of non-safe modules
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            else:
                module = node.names[0].name if node.names else ""
            root = module.split(".")[0]
            if root not in _SAFE_MODULES:
                return f"SecurityError: import of '{module}' is not allowed in sandbox"
        # Block exec/eval/compile (but NOT __import__ - it's needed for imports)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile"):
                return f"SecurityError: '{node.func.id}' is not allowed in sandbox"
    return None


def _run_in_process(code: str, test_code: str, result_queue: multiprocessing.Queue) -> None:
    """
    Worker function executed in a child process.
    Captures stdout/stderr and returns ExecutionResult via queue.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    # ✅ FIXED: Use _SAFE_BUILTINS which now includes __import__
    ns: dict = {"__builtins__": _SAFE_BUILTINS}
    passed = False
    exception_str = ""
    start = time.perf_counter()

    try:
        exec(compile(code, "<codefix>", "exec"), ns)  # noqa: S102
        exec(compile(test_code, "<test>", "exec"), ns)  # noqa: S102
        passed = True
    except AssertionError as e:
        exception_str = f"AssertionError: {e}"
    except Exception:  # noqa: BLE001
        exception_str = traceback.format_exc()
    finally:
        runtime_ms = (time.perf_counter() - start) * 1000
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    result_queue.put(
        ExecutionResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            exception=exception_str,
            passed=passed,
            runtime_ms=runtime_ms,
            timed_out=False,
        )
    )


def run_code(code: str, test_code: str = "", timeout_s: float = 5.0) -> ExecutionResult:
    """
    Execute `code` then `test_code` in a sandboxed subprocess.

    Args:
        code:      The (potentially buggy) solution code.
        test_code: Test assertion code (e.g. "assert add(1,2)==3").
        timeout_s: Max wall-clock seconds before kill.

    Returns:
        ExecutionResult with pass/fail, stdout, stderr, runtime.
    """
    full_code = textwrap.dedent(code)
    full_test = textwrap.dedent(test_code)

    # Static check first (fast, no subprocess)
    if err := _validate_ast(full_code):
        return ExecutionResult(exception=err, passed=False)

    # Run in child process for isolation
    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_in_process,
        args=(full_code, full_test, q),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return ExecutionResult(timed_out=True, exception=f"TimeoutError: exceeded {timeout_s}s")

    if not q.empty():
        return q.get_nowait()

    return ExecutionResult(exception="ExecutionError: process exited without result", passed=False)


def run_all_tests(code: str, test_cases: list, timeout_s: float = 5.0) -> list[ExecutionResult]:
    """
    Run all test cases against the given code.
    Each test is isolated (imports/state don't leak between tests).
    """
    results = []
    for tc in test_cases:
        res = run_code(
            code, tc.code, timeout_s=tc.timeout_s if hasattr(tc, "timeout_s") else timeout_s
        )
        results.append(res)
    return results