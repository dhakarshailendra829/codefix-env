"""
Easy Tasks — Syntax errors, simple fixes, missing colons, wrong indentation.
Each task has 3+ test cases and a clear hint.
"""

from __future__ import annotations

from codefix_env.models import BugCategory, Difficulty, Task, TestCase

EASY_TASKS: list[Task] = [
    # ── Task E1: Missing return ──────────────────────────────────────────
    Task(
        id="easy-001-missing-return",
        title="Missing Return Statement",
        description="The function adds two numbers but never returns the result.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.RETURN_BUG,
        tags=["return", "arithmetic"],
        max_steps=10,
        buggy_code="""\
def add_numbers(a, b):
    result = a + b
""",
        solution_code="""\
def add_numbers(a, b):
    result = a + b
    return result
""",
        hints=[
            "The function computes a result but never sends it back to the caller.",
            "You need a `return` statement at the end of the function.",
        ],
        test_cases=[
            TestCase(name="test_positive", code="assert add_numbers(2, 3) == 5"),
            TestCase(name="test_negative", code="assert add_numbers(-1, -4) == -5"),
            TestCase(name="test_zero", code="assert add_numbers(0, 0) == 0"),
            TestCase(name="test_float", code="assert add_numbers(1.5, 2.5) == 4.0"),
        ],
    ),
    # ── Task E2: Wrong comparison operator ──────────────────────────────
    Task(
        id="easy-002-wrong-operator",
        title="Wrong Comparison Operator",
        description="The is_even function uses assignment `=` instead of equality `==`.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.WRONG_OPERATOR,
        tags=["operators", "boolean"],
        max_steps=8,
        buggy_code="""\
def is_even(n):
    return n % 2 = 0
""",
        solution_code="""\
def is_even(n):
    return n % 2 == 0
""",
        hints=[
            "Python uses `==` for comparison, `=` is only for assignment.",
            "Look at the return expression carefully — one character is wrong.",
        ],
        test_cases=[
            TestCase(name="test_even_2", code="assert is_even(2) == True"),
            TestCase(name="test_odd_3", code="assert is_even(3) == False"),
            TestCase(name="test_zero", code="assert is_even(0) == True"),
            TestCase(name="test_neg", code="assert is_even(-4) == True"),
        ],
    ),
    # ── Task E3: Off-by-one in range ─────────────────────────────────────
    Task(
        id="easy-003-off-by-one-range",
        title="Off-by-One in Range",
        description="sum_to_n should sum numbers 1 through n inclusive, but misses the last number.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.OFF_BY_ONE,
        tags=["range", "loop", "off-by-one"],
        max_steps=10,
        buggy_code="""\
def sum_to_n(n):
    total = 0
    for i in range(1, n):
        total += i
    return total
""",
        solution_code="""\
def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
""",
        hints=[
            "Python's range(a, b) excludes b. Think about whether n should be included.",
            "Change `range(1, n)` to include n itself.",
        ],
        test_cases=[
            TestCase(name="test_sum_5", code="assert sum_to_n(5) == 15"),
            TestCase(name="test_sum_10", code="assert sum_to_n(10) == 55"),
            TestCase(name="test_sum_1", code="assert sum_to_n(1) == 1"),
            TestCase(name="test_sum_0", code="assert sum_to_n(0) == 0"),
        ],
    ),
    # ── Task E4: Wrong indentation ───────────────────────────────────────
    Task(
        id="easy-004-wrong-indent",
        title="Wrong Indentation in Loop",
        description="The return statement is inside the loop — it exits too early.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.LOGIC,
        tags=["indentation", "loop"],
        max_steps=10,
        buggy_code="""\
def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
        return max_val
""",
        solution_code="""\
def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
""",
        hints=[
            "The return statement is indented inside the for loop — it exits after checking the first element.",
            "De-indent `return max_val` by one level so it runs after the loop finishes.",
        ],
        test_cases=[
            TestCase(name="test_basic", code="assert find_max([3, 1, 4, 1, 5, 9]) == 9"),
            TestCase(name="test_sorted", code="assert find_max([1, 2, 3, 4, 5]) == 5"),
            TestCase(name="test_single", code="assert find_max([42]) == 42"),
            TestCase(name="test_negative", code="assert find_max([-5, -1, -3]) == -1"),
        ],
    ),
    # ── Task E5: Missing colon ───────────────────────────────────────────
    Task(
        id="easy-005-missing-colon",
        title="Missing Colon in Function Definition",
        description="A function definition is missing its colon, causing a SyntaxError.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.SYNTAX,
        tags=["syntax", "colon"],
        max_steps=8,
        buggy_code="""\
def greet(name)
    return f"Hello, {name}!"
""",
        solution_code="""\
def greet(name):
    return f"Hello, {name}!"
""",
        hints=[
            "Every `def` statement must end with a colon `:`.",
        ],
        test_cases=[
            TestCase(name="test_hello", code='assert greet("Alice") == "Hello, Alice!"'),
            TestCase(name="test_empty", code='assert greet("") == "Hello, !"'),
        ],
    ),
    # ── Task E6: String not converted to int ────────────────────────────
    Task(
        id="easy-006-type-mismatch",
        title="Type Mismatch — String + Integer",
        description="The function tries to add a string to an integer without converting.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.TYPE_ERROR,
        tags=["type", "casting"],
        max_steps=10,
        buggy_code="""\
def add_age(base_age, years_str):
    return base_age + years_str
""",
        solution_code="""\
def add_age(base_age, years_str):
    return base_age + int(years_str)
""",
        hints=[
            "years_str is a string. You need to convert it to int before adding.",
            "Use `int(years_str)` to convert.",
        ],
        test_cases=[
            TestCase(name="test_basic", code='assert add_age(25, "5") == 30'),
            TestCase(name="test_zero", code='assert add_age(30, "0") == 30'),
            TestCase(name="test_large", code='assert add_age(0, "100") == 100'),
        ],
    ),
    # ── Task E7: Wrong list index ────────────────────────────────────────
    Task(
        id="easy-007-wrong-index",
        title="Wrong List Index",
        description="get_last should return the last element but uses index 0.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.INDEX_ERROR,
        tags=["list", "index"],
        max_steps=8,
        buggy_code="""\
def get_last(items):
    return items[0]
""",
        solution_code="""\
def get_last(items):
    return items[-1]
""",
        hints=[
            "Python index -1 refers to the last element of a list.",
        ],
        test_cases=[
            TestCase(name="test_basic", code="assert get_last([1, 2, 3]) == 3"),
            TestCase(name="test_single", code="assert get_last(['x']) == 'x'"),
            TestCase(name="test_strings", code="assert get_last(['a', 'b', 'c']) == 'c'"),
        ],
    ),
    # ── Task E8: Boolean logic inverted ─────────────────────────────────
    Task(
        id="easy-008-inverted-bool",
        title="Inverted Boolean Condition",
        description="is_adult returns True for ages under 18 and False for adults.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.LOGIC,
        tags=["boolean", "condition"],
        max_steps=8,
        buggy_code="""\
def is_adult(age):
    return age < 18
""",
        solution_code="""\
def is_adult(age):
    return age >= 18
""",
        hints=[
            "The condition is backwards — an adult is someone 18 or older.",
            "Change `<` to `>=`.",
        ],
        test_cases=[
            TestCase(name="test_adult", code="assert is_adult(18) == True"),
            TestCase(name="test_child", code="assert is_adult(10) == False"),
            TestCase(name="test_exactly", code="assert is_adult(17) == False"),
            TestCase(name="test_old", code="assert is_adult(65) == True"),
        ],
    ),
]
