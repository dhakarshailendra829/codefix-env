"""
GRPO Training with TRL + CodeFix-Env
======================================
Train an LLM to fix Python bugs using Group Relative Policy Optimization (GRPO).

The LLM receives buggy code + feedback and produces structured CodeFixAction JSON.
The environment executes the action and returns a reward signal.

Architecture:
    LLM (policy) ──→ CodeFixAction JSON ──→ CodeFix-Env ──→ reward
         ↑_________________________________________________|

Requirements:
    pip install codefix-env[llm]
    # Start server first:
    uvicorn server.app:app --port 8000

Run:
    python examples/grpo_training.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Optional

from codefix_env.models import CodeFixAction, CodeFixClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logger = logging.getLogger(__name__)

# ── Prompt Builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python debugger. You receive buggy Python code and test failure feedback.
Your job is to identify and fix bugs by producing structured actions.

You MUST respond with a valid JSON object in this exact format:
{
  "action_type": "<one of: run_tests, edit_line, insert_line, delete_line, get_hint, submit_fix>",
  "line_number": <integer or null>,
  "new_content": "<string or null>",
  "reasoning": "<your chain-of-thought explanation>"
}

Rules:
- edit_line and insert_line require both line_number and new_content
- delete_line requires line_number
- run_tests, get_hint, submit_fix require neither
- Use run_tests first to understand failures
- Use submit_fix when confident the bug is fixed
- Think step by step in reasoning field
"""


def build_prompt(obs) -> str:
    """Build the user prompt from a CodeFixObservation."""
    lines = obs.current_code.splitlines()
    numbered = "\n".join(f"{i+1:3d} | {line}" for i, line in enumerate(lines))

    prompt = f"""## Task: {obs.task_id}
## Difficulty: {obs.difficulty}

## Current Code (with line numbers):
```python
{numbered}
```

## Test Results ({obs.tests_passed}/{obs.tests_total} passing):
{obs.test_output or "No tests run yet."}

## Feedback:
{obs.feedback}

## Steps remaining: {obs.steps_remaining}/{obs.max_steps}
## Current score: {obs.score_so_far:.3f}

What action will you take? Respond with JSON only."""
    return prompt


# ── Reward Function for GRPO ──────────────────────────────────────────────────


class CodeFixRewardFunction:
    """
    Reward function compatible with TRL's GRPOTrainer.

    For each generated response:
    1. Parse the JSON action
    2. Execute it in the environment
    3. Return the shaped reward
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        """
        Called by TRL for each batch of (prompt, completion) pairs.
        Returns a list of rewards.
        """
        rewards = []
        for prompt, completion in zip(prompts, completions):
            reward = asyncio.run(self._eval_single(prompt, completion, kwargs))
            rewards.append(reward)
        return rewards

    async def _eval_single(self, prompt: str, completion: str, context: dict) -> float:
        """Execute one action and return reward."""
        # Parse action from LLM output
        action = self._parse_action(completion)
        if action is None:
            return -0.1  # malformed JSON penalty

        # Get session from context (set up per-example by training loop)
        session_id = context.get("session_id")
        if not session_id:
            return 0.0

        try:
            async with CodeFixClient(self.base_url) as client:
                # Restore session
                result = await client.step(action)
                return result.reward
        except Exception as e:
            logger.warning("Reward eval failed: %s", e)
            return 0.0

    @staticmethod
    def _parse_action(text: str) -> CodeFixAction | None:
        """Parse LLM output into a CodeFixAction. Returns None if invalid."""
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
            return CodeFixAction(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug("Failed to parse action: %s | text: %s", e, text[:200])
            return None


# ── Training Loop ─────────────────────────────────────────────────────────────


def build_dataset(num_episodes: int = 100, task_ids: Optional[list[str]] = None):
    """
    Build a HuggingFace Dataset of (prompt, session_id) pairs for GRPO.
    Each row = one environment reset + initial observation.
    """
    try:
        from datasets import Dataset
    except Exception as e:
        raise RuntimeError("Failed to import datasets") from e

    import httpx

    rows = []
    for i in range(num_episodes):
        # Reset environment and get initial observation
        try:
            r = httpx.post("http://localhost:8000/reset", json={}, timeout=10)
            r.raise_for_status()
            obs_data = r.json()
            session_id = r.headers.get("x-session-id", "")

            # Build observation object
            from codefix_env import CodeFixObservation

            obs = CodeFixObservation.model_validate(obs_data)

            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(obs)},
                    ],
                    "session_id": session_id,
                    "task_id": obs.task_id,
                    "difficulty": obs.difficulty,
                }
            )
        except Exception as e:
            logger.warning("Failed to reset env for row %d: %s", i, e)

    return Dataset.from_list(rows)


def train(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./outputs/grpo_codefix",
    num_train_steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    server_url: str = "http://localhost:8000",
):
    """
    Run GRPO training with TRL.

    Args:
        model_name:       HuggingFace model ID
        output_dir:       Where to save checkpoints
        num_train_steps:  Total gradient steps
        batch_size:       Per-device batch size
        learning_rate:    LR for optimizer
        server_url:       CodeFix-Env server URL
    """
    try:
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:
        raise RuntimeError("Bad input") from e

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset (resetting environments)...")
    dataset = build_dataset(num_episodes=num_train_steps * batch_size)

    reward_fn = CodeFixRewardFunction(base_url=server_url)

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        max_new_tokens=256,
        temperature=0.7,
        logging_steps=10,
        save_steps=100,
        report_to="none",  # set to "wandb" for W&B logging
    )

    trainer = GRPOTrainer(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training with CodeFix-Env")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="./outputs/grpo_codefix")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument(
        "--demo", action="store_true", help="Run a quick demo without actual training"
    )
    args = parser.parse_args()

    if args.demo:
        # Quick parse test
        test_json = '{"action_type": "run_tests", "line_number": null, "new_content": null, "reasoning": "Check failures first"}'
        action = CodeFixRewardFunction._parse_action(test_json)
        print(f"Parsed action: {action}")
        print("Demo successful — install trl + transformers to run full training.")
    else:
        train(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            server_url=args.server_url,
        )
