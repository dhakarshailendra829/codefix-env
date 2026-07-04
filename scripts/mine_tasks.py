"""
Task Miner — grow the task set from real bugfix commits instead of hand-writing more.

Why this exists
----------------
21 hand-written tasks proves the environment MECHANICS work. It does not
demonstrate the environment scales, and reviewers who know SWE-bench will
immediately ask "did you just make these up?" This script mines candidate
tasks from real, merged GitHub PRs so the task set has provenance you can
point to (commit URL, real diff, real repo).

What it does
------------
1. Searches GitHub for merged PRs matching bugfix-shaped commit messages
   (e.g. "fix off-by-one", "fix null check") in small, popular pure-Python
   repos.
2. Filters to PRs that touch exactly ONE function in ONE file (so the
   resulting task has a clean "buggy version" / "fixed version" pair without
   needing multi-file context).
3. Uses the pre-fix version as `buggy_code` and the post-fix version as
   `solution_code`.
4. Emits a review queue (JSONL) — a human (you) still approves each one
   before it's promoted into tasks/mined.py, because auto-generated test
   cases from diffs are not reliable enough to trust blindly.

This is intentionally a semi-automated pipeline, not a fully autonomous one.
A reviewed queue of 50 real tasks is worth more on a resume than 500
unreviewed ones — the review step IS the point, don't skip it.

Usage
-----
    export GITHUB_TOKEN=ghp_xxx   # avoid the 60 req/hr unauthenticated limit
    python scripts/mine_tasks.py --repo psf/requests --limit 20
    python scripts/mine_tasks.py --repo pallets/flask --limit 20

Output: scripts/mined_candidates.jsonl (one candidate task per line, for
manual review before promotion to tasks/mined.py).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import httpx

GITHUB_API = "https://api.github.com"
BUGFIX_KEYWORDS = [
    "fix off-by-one",
    "fix null",
    "fix none check",
    "fix incorrect",
    "fix wrong",
    "fix bug",
    "fix logic",
    "fix edge case",
    "fix typo in",
    "fix index",
]


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def search_bugfix_prs(repo: str, limit: int) -> list[dict]:
    """
    Search merged PRs in `repo` whose title matches bugfix patterns.

    GitHub's search API rejects complex parenthesized OR groups combined
    with `in:title` (422 Unprocessable Entity). The reliable approach is one
    query per keyword, deduped by PR number, stopping once `limit` is hit.
    """
    seen: dict[int, dict] = {}
    for kw in BUGFIX_KEYWORDS:
        if len(seen) >= limit:
            break
        q = f'repo:{repo} is:pr is:merged in:title "{kw}"'
        resp = httpx.get(
            f"{GITHUB_API}/search/issues",
            params={"q": q, "per_page": 10},
            headers=_headers(),
            timeout=20.0,
        )
        if resp.status_code != 200:
            print(
                f"  search failed for '{kw}': {resp.status_code} {resp.text[:200]}",
                file=sys.stderr,
            )
            continue
        for item in resp.json().get("items", []):
            seen[item["number"]] = item
    return list(seen.values())[:limit]


def get_pr_files(repo: str, pr_number: int) -> list[dict]:
    resp = httpx.get(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
        headers=_headers(),
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def is_single_function_python_change(files: list[dict]) -> bool:
    """Keep only PRs touching exactly one .py file with a small, focused diff."""
    py_files = [f for f in files if f["filename"].endswith(".py")]
    if len(py_files) != 1:
        return False
    f = py_files[0]
    # Small, focused diffs are far more likely to be a single logic-bug fix
    # rather than a refactor, a new feature, or a multi-concern change.
    return f.get("changes", 0) <= 20 and f.get("status") == "modified"


def extract_touched_function(patch: str) -> str | None:
    """Best-effort: pull the function name from a unified diff hunk header."""
    for line in patch.splitlines():
        if line.startswith("@@") and "def " in line:
            try:
                return line.split("def ")[1].split("(")[0].strip()
            except IndexError:
                continue
    return None


def validate_python_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def mine(repo: str, limit: int, out_path: Path) -> int:
    candidates = []
    prs = search_bugfix_prs(repo, limit)
    print(f"Found {len(prs)} candidate PR titles in {repo}")

    for pr in prs:
        pr_number = pr["number"]
        try:
            files = get_pr_files(repo, pr_number)
        except httpx.HTTPStatusError as e:
            print(f"  skip PR #{pr_number}: {e}", file=sys.stderr)
            continue

        if not is_single_function_python_change(files):
            continue

        f = files[0]
        fn_name = extract_touched_function(f.get("patch", ""))
        if not fn_name:
            continue

        candidates.append(
            {
                "source": "github_pr",
                "repo": repo,
                "pr_number": pr_number,
                "pr_title": pr["title"],
                "pr_url": pr["html_url"],
                "file": f["filename"],
                "function_name": fn_name,
                "patch": f.get("patch", ""),
                "status": "NEEDS_MANUAL_REVIEW",
                # A human must fill these in after reading the patch:
                "buggy_code": None,
                "solution_code": None,
                "test_cases": [],
            }
        )

    with out_path.open("a") as fh:
        for c in candidates:
            fh.write(json.dumps(c) + "\n")

    print(f"Wrote {len(candidates)} candidates to {out_path}")
    print("Next: open the file, fill in buggy_code/solution_code/test_cases for")
    print("each candidate by reading the linked PR, then promote reviewed ones")
    print("into tasks/mined.py following the Task schema in models.py.")
    return len(candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. psf/requests")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--out", default="scripts/mined_candidates.jsonl", type=Path
    )
    args = parser.parse_args()
    mine(args.repo, args.limit, args.out)
