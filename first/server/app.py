from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app
from codefix_env.models import CodeFixAction, CodeFixObservation
from codefix_env.server.codefix_environment import CodeFixEnvironment

app = create_fastapi_app(CodeFixEnvironment, CodeFixAction, CodeFixObservation)


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "codefix-env"}


def main():
    """Entry point required by OpenEnv validator"""
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
