FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY codefix_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt openenv-core && rm /tmp/requirements.txt
COPY codefix_env /app/codefix_env
COPY inference.py /app/
COPY pyproject.toml /app/
COPY server/app.py /app/server/app.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
