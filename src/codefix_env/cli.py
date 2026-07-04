"""
CodeFix-Env CLI
================
Entry point registered in pyproject.toml as `codefix-server`.

Previously pyproject.toml declared:
    [project.scripts]
    codefix-server = "codefix_env.cli:app"

...but this module did not exist, so `pip install -e .` produced a broken
console script. This file makes that entry point real.

Usage:
    codefix-server serve                 # start the API server
    codefix-server serve --port 9000
    codefix-server tasks                 # list all tasks
    codefix-server info                  # print version + task counts
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from codefix_env import __version__
from codefix_env.tasks import list_tasks, task_count

app = typer.Typer(help="CodeFix-Env command line interface.")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)."),
) -> None:
    """Start the CodeFix-Env FastAPI server."""
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=reload)


@app.command()
def tasks(
    difficulty: str = typer.Option(None, help="Filter: easy | medium | hard"),
) -> None:
    """List all available debugging tasks."""
    from codefix_env.models import Difficulty

    diff_filter = Difficulty(difficulty) if difficulty else None
    table = Table(title="CodeFix-Env Tasks")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Difficulty")
    table.add_column("Category")
    for t in list_tasks(difficulty=diff_filter):
        table.add_row(t.id, t.title, str(t.difficulty), str(t.bug_category))
    console.print(table)


@app.command()
def info() -> None:
    """Print version and task counts."""
    console.print(f"[bold]codefix-env[/bold] v{__version__}")
    console.print(task_count())


if __name__ == "__main__":
    app()
