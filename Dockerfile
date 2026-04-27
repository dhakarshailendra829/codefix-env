# Root Dockerfile — delegates to server/Dockerfile
# Usage: docker build -t codefix-env .

FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir --prefix=/install . || pip install --no-cache-dir --prefix=/install -r /dev/stdin <<'EOF'
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
httpx>=0.27.0
torch>=2.2.0
numpy>=1.26.0
python-dotenv>=1.0.0
structlog>=24.1.0
rich>=13.7.0
EOF

FROM python:3.11-slim AS runtime
RUN groupadd -r codefix && useradd -r -g codefix -m codefix
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=codefix:codefix src/ ./src/
COPY --chown=codefix:codefix server/ ./server/
COPY --chown=codefix:codefix pyproject.toml openenv.yaml .env.example ./
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src HOST=0.0.0.0 PORT=8000
USER codefix
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers 2"]