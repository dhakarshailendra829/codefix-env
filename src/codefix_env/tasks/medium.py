"""
Medium Tasks — Algorithm bugs, logic errors, missing edge cases, scope bugs.
"""

from __future__ import annotations

from codefix_env.models import BugCategory, Difficulty, Task, TestCase

MEDIUM_TASKS: list[Task] = [
    # ── Task M1: Fibonacci — wrong base case ────────────────────────────
    Task(
        id="medium-001-fibonacci",
        title="Fibonacci — Wrong Base Case",
        description="The Fibonacci function has an incorrect base case that makes fib(1) return 0 instead of 1.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.ALGORITHM_BUG,
        tags=["recursion", "fibonacci", "base-case"],
        max_steps=15,
        buggy_code="""\
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 0   # BUG: should be 1
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
        solution_code="""\
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
        hints=[
            "The first Fibonacci number (n=1) should be 1, not 0.",
            "Fix the return value when n == 1.",
        ],
        test_cases=[
            TestCase(name="test_fib_0", code="assert fibonacci(0) == 0"),
            TestCase(name="test_fib_1", code="assert fibonacci(1) == 1"),
            TestCase(name="test_fib_5", code="assert fibonacci(5) == 5"),
            TestCase(name="test_fib_10", code="assert fibonacci(10) == 55"),
        ],
    ),
    # ── Task M2: Binary search — wrong mid ──────────────────────────────
    Task(
        id="medium-002-binary-search",
        title="Binary Search — Wrong Mid Calculation",
        description="The binary search uses `mid = high / 2` instead of `(low + high) // 2`, causing incorrect results.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.ALGORITHM_BUG,
        tags=["binary-search", "algorithm"],
        max_steps=15,
        buggy_code="""\
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = high // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
        solution_code="""\
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
        hints=[
            "The midpoint should be calculated from both `low` and `high`, not just `high`.",
            "`mid = (low + high) // 2` is the correct formula.",
        ],
        test_cases=[
            TestCase(name="test_found_middle", code="assert binary_search([1,3,5,7,9], 5) == 2"),
            TestCase(name="test_found_first", code="assert binary_search([1,3,5,7,9], 1) == 0"),
            TestCase(name="test_found_last", code="assert binary_search([1,3,5,7,9], 9) == 4"),
            TestCase(name="test_not_found", code="assert binary_search([1,3,5,7,9], 4) == -1"),
            TestCase(name="test_single", code="assert binary_search([42], 42) == 0"),
        ],
    ),
    # ── Task M3: Palindrome — wrong slice ───────────────────────────────
    Task(
        id="medium-003-palindrome",
        title="Palindrome Check — Wrong Reverse",
        description="The palindrome check reverses incorrectly — it uses `[::-2]` instead of `[::-1]`.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.LOGIC,
        tags=["string", "palindrome", "slicing"],
        max_steps=12,
        buggy_code="""\
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-2]
""",
        solution_code="""\
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
""",
        hints=[
            "`[::-2]` reverses every other character. Use `[::-1]` to reverse the full string.",
        ],
        test_cases=[
            TestCase(name="test_racecar", code='assert is_palindrome("racecar") == True'),
            TestCase(name="test_hello", code='assert is_palindrome("hello") == False'),
            TestCase(
                name="test_spaces",
                code='assert is_palindrome("A man a plan a canal Panama") == True',
            ),
            TestCase(name="test_single", code='assert is_palindrome("a") == True'),
        ],
    ),
    # ── Task M4: FizzBuzz wrong order ────────────────────────────────────
    Task(
        id="medium-004-fizzbuzz",
        title="FizzBuzz — Wrong Condition Order",
        description="FizzBuzz checks `n % 3` and `n % 5` before the combined case, causing multiples of 15 to output just 'Fizz'.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.LOGIC,
        tags=["fizzbuzz", "condition-order"],
        max_steps=12,
        buggy_code="""\
def fizzbuzz(n):
    if n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    elif n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    else:
        return str(n)
""",
        solution_code="""\
def fizzbuzz(n):
    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)
""",
        hints=[
            "The most specific condition (divisible by both 3 and 5) must be checked first.",
            "Move the FizzBuzz check to the top.",
        ],
        test_cases=[
            TestCase(name="test_fizz", code='assert fizzbuzz(9) == "Fizz"'),
            TestCase(name="test_buzz", code='assert fizzbuzz(10) == "Buzz"'),
            TestCase(name="test_fizzbuzz", code='assert fizzbuzz(15) == "FizzBuzz"'),
            TestCase(name="test_number", code='assert fizzbuzz(7) == "7"'),
            TestCase(name="test_30", code='assert fizzbuzz(30) == "FizzBuzz"'),
        ],
    ),
    # ── Task M5: Mutable default argument ───────────────────────────────
    # ── Task M5: Mutable default argument ───────────────────────────────
    Task(
        id="medium-005-mutable-default",
        title="Mutable Default Argument Bug",
        description="Using a mutable list as a default argument causes state to persist across calls.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.LOGIC,
        tags=["mutable-default", "python-gotcha"],
        max_steps=12,
        buggy_code="""\
    def append_item(item, items=[]):
        items.append(item)
        return items
    """,
        solution_code="""\
    def append_item(item, items=None):
        if items is None:
            items = []
        items.append(item)
        return items
    """,
        hints=[
            "Default mutable arguments in Python are shared across all calls.",
            "Use `None` as default and create the list inside the function.",
        ],
        test_cases=[
            TestCase(name="test_first_call",  code="assert append_item(1) == [1]"),
            TestCase(name="test_second_call", code="assert append_item(2) == [2]"),  # ✅ CHANGED from [2] expectation
            TestCase(name="test_with_list",   code="assert append_item(3, [10, 20]) == [10, 20, 3]"),
        ],
    ),
    
    # ── Task M6: Scope bug — global variable ────────────────────────────
    Task(
        id="medium-006-scope-bug",
        title="Scope Bug — Missing Global Declaration",
        description="The counter function tries to modify a global variable without declaring it global.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.SCOPE_BUG,
        tags=["scope", "global"],
        max_steps=12,
        buggy_code="""\
count = 0

def increment():
    count += 1
    return count
""",
        solution_code="""\
count = 0

def increment():
    global count
    count += 1
    return count
""",
        hints=[
            "To modify a global variable inside a function, you must declare it with `global`.",
            "Add `global count` at the start of the function.",
        ],
        test_cases=[
            TestCase(name="test_increment_once", code="increment(); assert increment() >= 1"),
            TestCase(name="test_returns_int", code="assert isinstance(increment(), int)"),
        ],
    ),
    # ── Task M7: Merge sort wrong merge ─────────────────────────────────
    Task(
        id="medium-007-merge-sort",
        title="Merge Sort — Wrong Comparison in Merge",
        description="The merge step compares using `>` where it should use `<=`, producing a descending sort.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.ALGORITHM_BUG,
        tags=["sorting", "merge-sort"],
        max_steps=18,
        buggy_code="""\
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] > right[j]:   # BUG: should be <=
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
        solution_code="""\
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
        hints=[
            "The merge comparison is backwards — it picks the larger element first.",
            "Change `>` to `<=` in the merge function.",
        ],
        test_cases=[
            TestCase(
                name="test_basic", code="assert merge_sort([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]"
            ),
            TestCase(name="test_sorted", code="assert merge_sort([1, 2, 3]) == [1, 2, 3]"),
            TestCase(
                name="test_reverse", code="assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]"
            ),
            TestCase(name="test_single", code="assert merge_sort([1]) == [1]"),
            TestCase(name="test_empty", code="assert merge_sort([]) == []"),
        ],
    ),
    # ── Task M8: Dictionary key error ───────────────────────────────────
    Task(
        id="medium-008-dict-key",
        title="Dictionary — KeyError on Missing Key",
        description="word_count crashes on the second call because it doesn't handle missing keys.",
        difficulty=Difficulty.MEDIUM,
        bug_category=BugCategory.LOGIC,
        tags=["dict", "keyerror"],
        max_steps=12,
        buggy_code="""\
def word_count(words):
    counts = {}
    for word in words:
        counts[word] += 1
    return counts
""",
        solution_code="""\
def word_count(words):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts
""",
        hints=[
            "You can't add 1 to a key that doesn't exist yet.",
            "Use `dict.get(key, 0)` to safely retrieve a value with a default.",
        ],
        test_cases=[
            TestCase(name="test_basic", code="assert word_count(['a','b','a']) == {'a':2,'b':1}"),
            TestCase(name="test_empty", code="assert word_count([]) == {}"),
            TestCase(name="test_single", code="assert word_count(['x']) == {'x':1}"),
        ],
    ),
]
