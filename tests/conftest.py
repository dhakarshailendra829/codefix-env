"""
pytest configuration — runs before any tests.
Fixes Windows multiprocessing spawn method for sandbox isolation.
"""

import multiprocessing
import sys


def pytest_configure(config):
    """Called after command line options have been parsed."""
    if sys.platform == "win32":
        # Windows requires 'spawn' for multiprocessing in pytest
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set
