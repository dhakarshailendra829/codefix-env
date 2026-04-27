"""
CodeFix-Env FastAPI Server
===========================
REST API server for the CodeFix RL environment.

Endpoints:
  GET  /health          → Health check
  POST /reset           → Start new episode (creates session)
  POST /step            → Execute action
  GET  /state           → Get episode state
  GET  /tasks           → List available tasks
  GET  /tasks/{task_id} → Get task details
  GET  /metrics         → Server metrics

Session management via X-Session-ID header.

Run::

    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from codefix_env.models import CodeFixAction, Difficulty
from codefix_env.tasks import list_tasks, load_task, task_count
from server.codefix_environment import ServerEnvironment, SessionManager

# ── Logging ───────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

# ── Session store (global) ────────────────────────────────────────────────────

session_manager = SessionManager(max_sessions=int(os.getenv("MAX_SESSIONS", "500")))
_server_start_time = time.time()

# ── Request/Response Schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:    Optional[str]        = None
    difficulty: Optional[Difficulty] = None
    seed:       Optional[int]        = None


class HealthResponse(BaseModel):
    status:        str
    version:       str
    uptime_s:      float
    active_sessions: int
    task_counts:   dict


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("codefix_env_server_starting", task_counts=task_count())
    yield
    logger.info("codefix_env_server_stopping", sessions=len(session_manager))


app = FastAPI(
    title="CodeFix-Env",
    description=(
        "High-performance RL environment for automated code debugging. "
        "Built on Meta PyTorch OpenEnv."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request timing ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.debug("request", method=request.method, path=request.url.path, ms=round(ms, 1))
    response.headers["X-Process-Time-Ms"] = str(round(ms, 1))
    return response


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


# ── Session helper ────────────────────────────────────────────────────────────

def _get_session(session_id: Optional[str]) -> ServerEnvironment:
    """Retrieve session or raise 404."""
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Session-ID header is required. Call /reset first.",
        )
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Call /reset to create one.",
        )
    return session


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Server health check."""
    return HealthResponse(
        status="ok",
        version="0.2.0",
        uptime_s=round(time.time() - _server_start_time, 1),
        active_sessions=len(session_manager),
        task_counts=task_count(),
    )


@app.post("/reset", tags=["environment"])
async def reset(body: ResetRequest = ResetRequest()):
    """
    Start a new episode.

    Returns an observation + a session ID that must be sent
    in the `X-Session-ID` header for all subsequent requests.
    """
    session = session_manager.create()
    obs = session.reset(
        task_id=body.task_id,
        difficulty=body.difficulty,
        seed=body.seed,
    )
    response = JSONResponse(content=obs.model_dump(mode="json"))
    response.headers["X-Session-ID"] = session.session_id
    logger.info("session_created", session_id=session.session_id, task_id=obs.task_id)
    return response


@app.post("/step", tags=["environment"])
async def step(
    action: CodeFixAction,
    x_session_id: Optional[str] = Header(None),
):
    """Execute one action in the environment."""
    session = _get_session(x_session_id)
    result  = session.step(action)
    return result.model_dump(mode="json")


@app.get("/state", tags=["environment"])
async def state(x_session_id: Optional[str] = Header(None)):
    """Get internal episode state (non-destructive)."""
    session = _get_session(x_session_id)
    return session.state().model_dump(mode="json")


@app.get("/tasks", tags=["tasks"])
async def get_tasks(
    difficulty: Optional[Difficulty] = Query(None, description="Filter by difficulty"),
):
    """List all available tasks."""
    tasks = list_tasks(difficulty=difficulty)
    return {
        "tasks": [
            {
                "id":          t.id,
                "title":       t.title,
                "description": t.description,
                "difficulty":  t.difficulty,
                "bug_category": t.bug_category,
                "tags":        t.tags,
                "num_tests":   t.num_tests,
                "max_steps":   t.max_steps,
            }
            for t in tasks
        ],
        "total": len(tasks),
    }


@app.get("/tasks/{task_id}", tags=["tasks"])
async def get_task(task_id: str):
    """Get full details of a specific task (without solution)."""
    task = load_task(task_id)
    return {
        "id":           task.id,
        "title":        task.title,
        "description":  task.description,
        "difficulty":   task.difficulty,
        "bug_category": task.bug_category,
        "tags":         task.tags,
        "buggy_code":   task.buggy_code,
        "num_tests":    task.num_tests,
        "max_steps":    task.max_steps,
        "hints_count":  len(task.hints),
    }


@app.get("/metrics", tags=["system"])
async def metrics():
    """Server-level metrics."""
    return {
        "active_sessions": len(session_manager),
        "task_counts":     task_count(),
        "uptime_s":        round(time.time() - _server_start_time, 1),
    }