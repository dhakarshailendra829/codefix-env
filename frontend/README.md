# CodeFix-Env Frontend

This is a separate React + Vite frontend for the existing CodeFix-Env FastAPI backend. It lives entirely in `frontend/` and does not require backend code changes.

## Setup

Start the backend from the repository root:

```bash
pip install -e .
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend in another terminal:

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

On macOS or Linux, use `cp .env.example .env` instead of `copy .env.example .env`.

The default frontend environment value is:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

During local development, Vite proxies `/api/*` to `VITE_API_BASE_URL`. That keeps the app independent while allowing the browser to read the `X-Session-ID` header returned by `POST /reset`.

## What It Uses

- React with JavaScript/JSX
- Vite
- Custom CSS only
- Existing REST endpoints: `GET /health`, `GET /tasks`, `GET /tasks/{id}`, `POST /reset`, and `POST /step`

## Workflow

1. Select a task.
2. Start or reset a session.
3. Edit a line, insert a line, or delete a line.
4. Use the action panel to run tests, request a hint, submit the fix, or refresh the current code.
