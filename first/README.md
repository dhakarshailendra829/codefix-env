# CodeFix-Env

Real-world Python code debugging environment for agent evaluation.

## Tasks
- easy-fix-syntax (easy)
- medium-algorithm (medium)
- hard-multi-function (hard)

## Action Space
CodeFixAction with action_type (run_tests, edit_line, submit_fix, ...)

## Observation Space
CodeFixObservation with current_code, test_output, feedback, score_so_far

## Setup
```bash
uvicorn codefix_env.server.app:app --reload