# Contributing to CodeFix-Env

Thanks for contributing. This guide covers local setup, adding tasks, code style, and the pull request process.

## Development setup

```bash
git clone https://github.com/dhakarshailendra829/codefix-env.git
cd codefix-env

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

If you only want to use the package rather than develop it, the published release is on PyPI:

```bash
pip install codefix-env
```

Confirm either install with:

```bash
codefix-server info
```

## Running tests

```bash
# All tests
PYTHONPATH=src pytest tests/ -v

# Specific file
PYTHONPATH=src pytest tests/test_env.py -v

# With coverage
PYTHONPATH=src pytest tests/ --cov=src/codefix_env --cov-report=html
open htmlcov/index.html

# Sandbox tests only (fast, no heavy deps)
PYTHONPATH=src pytest tests/test_tasks.py::TestSandbox -v
```

All 197 existing tests should pass before you start, and before you open a pull request.

## Adding a new debugging task

Tasks live in `src/codefix_env/tasks/` — add to `easy.py`, `medium.py`, or `hard.py` depending on difficulty.

### Task template

```python
Task(
    id="easy-009-your-task-name",        # must be unique, follow the existing patterns 
    title="Clear Short Title",
    description="One sentence: what is the bug?",
    difficulty=Difficulty.EASY,
    bug_category=BugCategory.LOGIC,
    tags=["relevant", "tags"],
    max_steps=10,                         # 8-12 easy, 12-18 medium, 20-25 hard
    buggy_code="""\
def your_function(x):
    return x + 1   # BUG: should be x - 1
""",
    solution_code="""\
def your_function(x):
    return x - 1
""",
    hints=[
        "Hint 1: general direction without giving it away",
        "Hint 2: more specific, nearly the answer",
    ],
    test_cases=[
        TestCase(name="test_basic",    code="assert your_function(5) == 4"),
        TestCase(name="test_zero",     code="assert your_function(0) == -1"),
        TestCase(name="test_negative", code="assert your_function(-3) == -4"),
    ],
)
```

### Checklist before submitting a task

- [ ] Task ID is unique (not already present in the registry)
- [ ] At least 3 test cases
- [ ] At least 1 hint
- [ ] `buggy_code` fails at least one test
- [ ] `solution_code` passes all tests
- [ ] `pytest tests/test_tasks.py -v` passes
- [ ] Added to the appropriate `EASY_TASKS` / `MEDIUM_TASKS` / `HARD_TASKS` list

Tasks mined from real bug-fix commits via `scripts/mine_tasks.py` are also welcome, following the same checklist after human review — see the script's docstring for usage.

## Code style

Enforced with `ruff`, `black`, and `isort`. Run before committing:

```bash
ruff check src/ tests/ --fix
black src/ tests/ examples/
isort src/ tests/ examples/
```

Or, with pre-commit installed:

```bash
pre-commit run --all-files
```

## Pull request process

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/add-task-fibonacci-v2`.
3. Make your changes.
4. Run the full test suite: `PYTHONPATH=src pytest tests/ -v`.
5. Update `README.md` or `ARCHITECTURE.md` if the change affects what they describe — documentation that drifts from the code is treated as a bug here.
6. Open the pull request with a clear description of what changed and why, and fill out the PR template.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Python version
- CodeFix-Env version: `python -c "import codefix_env; print(codefix_env.__version__)"`
- Minimal reproduction code
- Full traceback

For a suspected sandbox escape or other security-relevant defect, do not open a public issue — see `SECURITY.md` for private reporting instructions.

## Code of conduct

Be respectful. See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).
