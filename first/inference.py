import asyncio
import os
import json
from typing import List, Optional
from openai import OpenAI

from codefix_env.server.codefix_environment import CodeFixEnvironment
from codefix_env.models import CodeFixAction

# === REQUIRED VARIABLES (exactly as per official sample) ===
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")                    # Must come from environment variable

TASK_NAME = os.getenv("TASK_NAME", "easy-fix-syntax")
MAX_STEPS = 12

# OpenAI Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env=codefix-env model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = min(max(score, 0.0), 1.0)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    env = CodeFixEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    last_obs = None

    log_start(TASK_NAME, "codefix-env", MODEL_NAME)

    try:
        last_obs = env.reset(task_name=TASK_NAME)

        for step in range(1, MAX_STEPS + 1):

            prompt = "You are a Python code debugger.\n\n"
            prompt += "Task: " + str(last_obs.task_name) + "\n\n"
            prompt += "Current code:\n```python\n" + str(last_obs.current_code) + "\n```\n\n"
            prompt += "Test output:\n" + str(last_obs.test_output) + "\n\n"
            prompt += "Feedback: " + str(last_obs.feedback) + "\n\n"
            prompt += 'Reply ONLY with JSON. Examples: {"action_type": "run_tests"} or {"action_type": "submit_fix", "code": "def add(a,b): return a+b"}\n'

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=300,
                )
                action_text = completion.choices[0].message.content.strip()
                action_dict = json.loads(action_text)
                action = CodeFixAction(**action_dict)
            except:
                action = CodeFixAction(action_type="run_tests")

            new_obs = env.step(action)

            reward = float(getattr(new_obs, "score_so_far", 0.0))
            done = bool(reward >= 0.9 or step >= MAX_STEPS)

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))

            log_step(step, action_str, reward, done, None)

            rewards.append(reward)
            steps_taken = step
            last_obs = new_obs

            if done:
                break

        success = (rewards[-1] >= 0.85) if rewards else False
        log_end(success, steps_taken, rewards)

    except Exception as e:
        print("[DEBUG] Error:", str(e))
        log_end(False, steps_taken, rewards)


if __name__ == "__main__":
    asyncio.run(main())