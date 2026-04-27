"""
Server-Side Environment Wrapper
================================
Thin wrapper around CodeFixEnvironment for use inside the FastAPI server.
Manages per-session state, serialisation, and concurrent session support.
"""
from __future__ import annotations

import uuid
from typing import Optional

from codefix_env.env import CodeFixEnvironment
from codefix_env.models import (
    CodeFixAction,
    CodeFixObservation,
    CodeFixState,
    Difficulty,
    StepResult,
)
from codefix_env.rewards import RewardPipeline, default_pipeline


class ServerEnvironment:
    """
    Stateful environment session.
    One instance per connected client (created on /reset).
    Thread-safe under asyncio (single-threaded event loop).
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self._env = CodeFixEnvironment(reward_pipeline=default_pipeline)

    def reset(
        self,
        task_id:    Optional[str]        = None,
        difficulty: Optional[Difficulty] = None,
        seed:       Optional[int]        = None,
    ) -> CodeFixObservation:
        return self._env.reset(task_id=task_id, difficulty=difficulty, seed=seed)

    def step(self, action: CodeFixAction) -> StepResult:
        return self._env.step(action)

    def state(self) -> CodeFixState:
        return self._env.state()

    @property
    def is_done(self) -> bool:
        try:
            return self._env.state().done
        except RuntimeError:
            return False


class SessionManager:
    """
    In-memory session store.
    Maps session_id → ServerEnvironment.
    For production, swap with Redis-backed store.
    """

    def __init__(self, max_sessions: int = 1000):
        self._sessions: dict[str, ServerEnvironment] = {}
        self.max_sessions = max_sessions

    def create(self) -> ServerEnvironment:
        if len(self._sessions) >= self.max_sessions:
            # Evict oldest done session
            for sid, env in list(self._sessions.items()):
                if env.is_done:
                    del self._sessions[sid]
                    break
        session = ServerEnvironment()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[ServerEnvironment]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)