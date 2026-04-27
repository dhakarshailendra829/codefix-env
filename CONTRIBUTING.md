# Contributing to CodeFix-Env

Thank you for contributing! This guide covers everything you need.

---

## Development Setup

```bash
git clone https://github.com/dhakarshailendra829/codefix-env
cd codefix-env

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode with dev deps
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_env.py -v

# With coverage
pytest tests/ --cov=src/codefix_env --cov-report=html
open htmlcov/index.html

# Just sandbox tests (fast, no heavy deps)
pytest tests/test_tasks.py::TestSandbox -v
```

---

## Adding a New Debugging Task

Tasks live in `src/codefix_env/tasks/` — pick `easy.py`, `medium.py`, or `hard.py`.

### Task Template

```python
Task(
    id="easy-009-your-task-name",        # must be unique, follow pattern
    title="Clear Short Title",
    description="One sentence: what is the bug?",
    difficulty=Difficulty.EASY,
    bug_category=BugCategory.LOGIC,
    tags=["relevant", "tags"],
    max_steps=10,                         # 8-12 for easy, 12-18 for medium, 20-25 for hard
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

### Checklist Before Submitting

- [ ] Task ID is unique (`task_id` not in `ALL_TASKS`)
- [ ] At least 3 test cases
- [ ] At least 1 hint
- [ ] `buggy_code` fails at least one test
- [ ] `solution_code` passes ALL tests
- [ ] Run `pytest tests/test_tasks.py -v` and confirm all pass
- [ ] Added to the appropriate `EASY_TASKS` / `MEDIUM_TASKS` / `HARD_TASKS` list

---

## Code Style

We use `ruff`, `black`, and `isort`. Run before committing:

```bash
ruff check src/ tests/ --fix
black src/ tests/ examples/
isort src/ tests/ examples/
```

Or just:
```bash
pre-commit run --all-files
```

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/add-task-fibonacci-v2`
3. Make your changes
4. Run the full test suite: `pytest tests/ -v`
5. Submit PR with a clear description
6. Fill out the PR template

---

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md).

Please include:
- Python version
- CodeFix-Env version (`python -c "import codefix_env; print(codefix_env.__version__)"`)
- Minimal reproduction code
- Full traceback

---

## Code of Conduct

Be kind. See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).