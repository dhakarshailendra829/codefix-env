"""
Hard Tasks — Multi-function bugs, complex algorithms, subtle logic errors.
These tasks require understanding interactions between multiple functions.
"""

from __future__ import annotations

from codefix_env.models import BugCategory, Difficulty, Task, TestCase

HARD_TASKS: list[Task] = [
    # ── Task H1: LRU Cache — wrong eviction ─────────────────────────────
    Task(
        id="hard-001-lru-cache",
        title="LRU Cache — Wrong Eviction Policy",
        description=(
            "A manual LRU cache implementation evicts the most recently used item "
            "instead of the least recently used. The `get` method also fails to update "
            "recency on a cache hit."
        ),
        difficulty=Difficulty.HARD,
        bug_category=BugCategory.MULTI_FUNCTION,
        tags=["lru", "cache", "data-structures", "ordered-dict"],
        max_steps=25,
        buggy_code="""\
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # BUG: does not move to end (not updating recency)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=True)  # BUG: evicts MRU not LRU
""",
        solution_code="""\
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # FIX: mark as recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # FIX: evict LRU (first item)
""",
        hints=[
            "When `get` is called, the accessed item should become the most recently used.",
            "To evict the *least* recently used, use `popitem(last=False)` — removes the first (oldest) item.",
            "Both `get` and `put` have bugs. Fix them independently.",
        ],
        test_cases=[
            TestCase(
                name="test_basic_put_get",
                code="""\
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
""",
            ),
            TestCase(
                name="test_eviction",
                code="""\
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.get(1)       # 1 is now MRU
cache.put(3, 3)    # evicts 2 (LRU)
assert cache.get(2) == -1
assert cache.get(3) == 3
""",
            ),
            TestCase(
                name="test_update_existing",
                code="""\
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.put(1, 10)   # update key 1
assert cache.get(1) == 10
""",
            ),
            TestCase(
                name="test_miss",
                code="""\
cache = LRUCache(1)
assert cache.get(99) == -1
""",
            ),
        ],
    ),
    # ── Task H2: Graph BFS — wrong visited tracking ─────────────────────
    Task(
        id="hard-002-graph-bfs",
        title="Graph BFS — Infinite Loop from Wrong Visited Tracking",
        description=(
            "BFS adds nodes to `visited` after dequeuing instead of before enqueuing, "
            "causing nodes to be visited multiple times and potentially looping."
        ),
        difficulty=Difficulty.HARD,
        bug_category=BugCategory.ALGORITHM_BUG,
        tags=["graph", "bfs", "visited"],
        max_steps=25,
        buggy_code="""\
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbour in graph.get(node, []):
                queue.append(neighbour)  # BUG: should check visited before enqueuing
    return order
""",
        solution_code="""\
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return order
""",
        hints=[
            "Nodes should be added to `visited` when they are *enqueued*, not when dequeued.",
            "Initialise `visited` with the start node.",
        ],
        test_cases=[
            TestCase(
                name="test_linear",
                code="""\
g = {0: [1], 1: [2], 2: [3], 3: []}
assert bfs(g, 0) == [0, 1, 2, 3]
""",
            ),
            TestCase(
                name="test_branching",
                code="""\
g = {0: [1, 2], 1: [3], 2: [3], 3: []}
result = bfs(g, 0)
assert result[0] == 0
assert set(result) == {0, 1, 2, 3}
assert len(result) == 4   # no duplicates
""",
            ),
            TestCase(
                name="test_single_node",
                code="""\
g = {0: []}
assert bfs(g, 0) == [0]
""",
            ),
        ],
    ),
    # ── Task H3: Decorator — wrong wraps usage ──────────────────────────
    Task(
        id="hard-003-decorator-bug",
        title="Decorator — Lost Function Metadata",
        description=(
            "A timing decorator is missing `@functools.wraps(func)`, which causes the "
            "wrapped function to lose its `__name__` and `__doc__`, breaking introspection."
        ),
        difficulty=Difficulty.HARD,
        bug_category=BugCategory.MULTI_FUNCTION,
        tags=["decorator", "functools", "wraps", "metadata"],
        max_steps=15,
        buggy_code="""\
import functools
import time

def timer(func):
    def wrapper(*args, **kwargs):   # BUG: missing @functools.wraps(func)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result
    return wrapper

@timer
def compute(n):
    \"\"\"Compute n^2.\"\"\"
    return n * n
""",
        solution_code="""\
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result
    return wrapper

@timer
def compute(n):
    \"\"\"Compute n^2.\"\"\"
    return n * n
""",
        hints=[
            "Without `@functools.wraps(func)`, the wrapper function replaces the original's metadata.",
            "Add `@functools.wraps(func)` directly above `def wrapper`.",
        ],
        test_cases=[
            TestCase(name="test_result", code="assert compute(5) == 25"),
            TestCase(name="test_name", code='assert compute.__name__ == "compute"'),
            TestCase(name="test_doc", code='assert compute.__doc__ == "Compute n^2."'),
        ],
    ),
    # ── Task H4: Stack with linked list — pop bug ───────────────────────
    Task(
        id="hard-004-linked-stack",
        title="Linked List Stack — Pop Returns Wrong Value",
        description=(
            "A stack implemented with a linked list has a `pop` method that updates the head "
            "pointer but returns the new head's value instead of the removed node's value."
        ),
        difficulty=Difficulty.HARD,
        bug_category=BugCategory.MULTI_FUNCTION,
        tags=["linked-list", "stack", "data-structures"],
        max_steps=25,
        buggy_code="""\
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class Stack:
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, val):
        node = Node(val)
        node.next = self.head
        self.head = node
        self.size += 1

    def pop(self):
        if self.head is None:
            raise IndexError("pop from empty stack")
        self.head = self.head.next   # BUG: loses the removed node before capturing value
        self.size -= 1
        return self.head.val if self.head else None

    def peek(self):
        if self.head is None:
            raise IndexError("peek from empty stack")
        return self.head.val

    def is_empty(self):
        return self.head is None
""",
        solution_code="""\
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class Stack:
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, val):
        node = Node(val)
        node.next = self.head
        self.head = node
        self.size += 1

    def pop(self):
        if self.head is None:
            raise IndexError("pop from empty stack")
        removed_val = self.head.val   # FIX: capture value before moving head
        self.head = self.head.next
        self.size -= 1
        return removed_val

    def peek(self):
        if self.head is None:
            raise IndexError("peek from empty stack")
        return self.head.val

    def is_empty(self):
        return self.head is None
""",
        hints=[
            "You need to save the current head's value before moving `self.head` to the next node.",
            "Store `removed_val = self.head.val` before updating `self.head`.",
        ],
        test_cases=[
            TestCase(
                name="test_push_pop",
                code="""\
s = Stack()
s.push(1); s.push(2); s.push(3)
assert s.pop() == 3
assert s.pop() == 2
assert s.pop() == 1
""",
            ),
            TestCase(
                name="test_peek",
                code="""\
s = Stack()
s.push(42)
assert s.peek() == 42
assert s.size == 1
""",
            ),
            TestCase(
                name="test_empty_pop",
                code="""\
s = Stack()
try:
    s.pop()
    assert False, "Should have raised"
except IndexError:
    pass
""",
            ),
            TestCase(
                name="test_size_tracking",
                code="""\
s = Stack()
s.push(1); s.push(2)
s.pop()
assert s.size == 1
""",
            ),
        ],
    ),
    # ── Task H5: Tokenizer — wrong string parsing ────────────────────────
    Task(
        id="hard-005-tokenizer",
        title="Mini Tokenizer — String Literal Handling Bug",
        description=(
            "A simple expression tokenizer fails on string literals because it doesn't "
            "handle quoted strings — it splits on spaces inside quotes."
        ),
        difficulty=Difficulty.HARD,
        bug_category=BugCategory.MULTI_FUNCTION,
        tags=["parsing", "tokenizer", "string-literals"],
        max_steps=25,
        buggy_code="""\
def tokenize(expr):
    \"\"\"Tokenize a simple expression into a list of tokens.\"\"\"
    tokens = []
    current = ''
    for ch in expr:
        if ch == ' ':
            if current:
                tokens.append(current)
                current = ''
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens
""",
        solution_code="""\
def tokenize(expr):
    \"\"\"Tokenize a simple expression into a list of tokens.\"\"\"
    tokens = []
    current = ''
    in_string = False
    for ch in expr:
        if ch == '\"' :
            in_string = not in_string
            current += ch
        elif ch == ' ' and not in_string:
            if current:
                tokens.append(current)
                current = ''
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens
""",
        hints=[
            "The tokenizer doesn't track whether it's inside a quoted string.",
            'Add an `in_string` flag that toggles when a `"` is encountered.',
            "Only split on spaces when `in_string` is False.",
        ],
        test_cases=[
            TestCase(name="test_simple", code='assert tokenize("a + b") == ["a", "+", "b"]'),
            TestCase(
                name="test_string_literal",
                code="assert tokenize('print \"hello world\"') == ['print', '\"hello world\"']",
            ),
            TestCase(name="test_no_spaces", code='assert tokenize("abc") == ["abc"]'),
            TestCase(name="test_empty", code='assert tokenize("") == []'),
        ],
    ),
]
