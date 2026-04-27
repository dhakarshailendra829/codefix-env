"""
CodeFixClient — Async & Sync HTTP Client
=========================================
Communicates with the CodeFix-Env FastAPI server via HTTP + WebSocket.

Usage::

    # Async (recommended for RL training loops)
    async with CodeFixClient("http://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(CodeFixAction(action_type="run_tests"))

    # Sync (via .sync() wrapper — for notebooks and scripts)
    with CodeFixClient("http://localhost:8000").sync() as env:
        obs = env.reset()
        result = env.step(CodeFixAction(action_type="run_tests"))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from codefix_env.models import (
    CodeFixAction,
    CodeFixObservation,
    CodeFixState,
    Difficulty,
    StepResult,
)

logger = logging.getLogger(__name__)


class CodeFixClient:
    """
    Async HTTP client for the CodeFix-Env server.
    Provides the same reset() / step() / state() interface as the server-side env.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "CodeFixClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        # Health check
        try:
            r = await self._client.get("/health")
            r.raise_for_status()
            logger.info("Connected to CodeFix-Env at %s", self.base_url)
        except Exception as e:
            logger.warning("Health check failed: %s. Proceeding anyway.", e)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Core API ──────────────────────────────────────────────────────────────

    async def reset(
        self,
        task_id: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
        seed: Optional[int] = None,
    ) -> CodeFixObservation:
        """Reset the environment and return the initial observation."""
        payload: dict = {}
        if task_id:
            payload["task_id"] = task_id
        if difficulty:
            payload["difficulty"] = difficulty
        if seed is not None:
            payload["seed"] = seed
        data = await self._post("/reset", payload)
        return CodeFixObservation.model_validate(data)

    async def step(self, action: CodeFixAction) -> StepResult:
        """Execute an action and return the step result."""
        data = await self._post("/step", action.model_dump())
        return StepResult.model_validate(data)

    async def state(self) -> CodeFixState:
        """Get internal episode state."""
        data = await self._get("/state")
        return CodeFixState.model_validate(data)

    async def list_tasks(self, difficulty: Optional[str] = None) -> list[dict]:
        """List all available tasks."""
        params = {}
        if difficulty:
            params["difficulty"] = difficulty
        data = await self._get("/tasks", params=params)
        return data

    async def health(self) -> dict:
        """Ping the server."""
        return await self._get("/health")

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    async def _post(self, path: str, body: dict) -> dict:
        assert self._client, "Use `async with CodeFixClient(...) as env:`"
        for attempt in range(self.max_retries):
            try:
                r = await self._client.post(path, json=body)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "HTTP %d on POST %s: %s", e.response.status_code, path, e.response.text
                )
                raise
            except httpx.TransportError:
                logger.error(
                    "Transport error on POST %s (attempt %d/%d)",
                    path,
                    attempt + 1,
                    self.max_retries,
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Failed after {self.max_retries} retries")

    async def _get(self, path: str, params: dict | None = None) -> dict:
        assert self._client, "Use `async with CodeFixClient(...) as env:`"
        r = await self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    # ── Sync wrapper ──────────────────────────────────────────────────────────

    def sync(self) -> "SyncCodeFixClient":
        """Return a synchronous wrapper around this async client."""
        return SyncCodeFixClient(self)


class SyncCodeFixClient:
    """
    Synchronous wrapper around CodeFixClient.
    Runs the async client in a dedicated event loop.

    Usage::

        with CodeFixClient("http://localhost:8000").sync() as env:
            obs = env.reset()
            result = env.step(action)
    """

    def __init__(self, async_client: CodeFixClient):
        self._async = async_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "SyncCodeFixClient":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async.__aenter__())
        return self

    def __exit__(self, *args) -> None:
        if self._loop:
            self._loop.run_until_complete(self._async.__aexit__(*args))
            self._loop.close()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def reset(self, **kwargs) -> CodeFixObservation:
        return self._run(self._async.reset(**kwargs))

    def step(self, action: CodeFixAction) -> StepResult:
        return self._run(self._async.step(action))

    def state(self) -> CodeFixState:
        return self._run(self._async.state())

    def list_tasks(self, **kwargs) -> list[dict]:
        return self._run(self._async.list_tasks(**kwargs))
