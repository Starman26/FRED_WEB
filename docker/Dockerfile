FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AGENT_LOG_LEVEL=INFO \
    AGENT_LOG_ENABLED=true \
    HOST=0.0.0.0 \
    PORT=8000

    
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

# Copiar e instalar dependencias primero (mejor cache)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copiar código
COPY src/ ./src/
COPY langgraph.json .

RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

CMD uvicorn src.api.main:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WEB_CONCURRENCY:-1} \
    --timeout-keep-alive 120 \
    --log-level info

    

