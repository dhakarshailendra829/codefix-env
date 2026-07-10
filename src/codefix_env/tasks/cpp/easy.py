"""
C++ tasks — easy tier. Test code is a sequence of assert(...)
statements, wrapped by CppExecutor into a main() that runs them in
order.
"""

from __future__ import annotations

from codefix_env.models import BugCategory, Difficulty, Task, TestCase

CPP_EASY_TASKS = [
    Task(
        id="cpp-easy-001-off-by-one",
        title="Off-by-One in Array Sum (C++)",
        description="A function that sums an array's elements has an off-by-one bug in its loop bound.",
        difficulty=Difficulty.EASY,
        bug_category=BugCategory.OFF_BY_ONE,
        language="cpp",
        tags=["cpp", "arrays", "loops"],
        max_steps=10,
        buggy_code="""\
#include <vector>
int sum_array(const std::vector<int>& arr) {
    int total = 0;
    for (int i = 0; i < (int)arr.size() - 1; i++) {  // BUG: should be i < arr.size()
        total += arr[i];
    }
    return total;
}
""",
        solution_code="""\
#include <vector>
int sum_array(const std::vector<int>& arr) {
    int total = 0;
    for (int i = 0; i < (int)arr.size(); i++) {
        total += arr[i];
    }
    return total;
}
""",
        hints=[
            "Check the loop's stopping condition against the actual array size.",
            "The loop is skipping the last element — compare `<` against `arr.size()` directly.",
        ],
        test_cases=[
            TestCase(name="test_basic", code="assert(sum_array({1, 2, 3}) == 6);"),
            TestCase(name="test_single", code="assert(sum_array({5}) == 5);"),
            TestCase(name="test_empty", code="assert(sum_array({}) == 0);"),
            TestCase(name="test_negative", code="assert(sum_array({-1, -2, 3}) == 0);"),
        ],
    ),
]
