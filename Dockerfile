# Dockerfile for FrEDie Agent API — Google Cloud Run
#
# Build:  docker build -t fredie-agent .
# Test:   docker run -p 8080:8080 --env-file .env fredie-agent
# Deploy: gcloud run deploy (see DEPLOY_CLOUDRUN.md)

FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api_server.py .

# Cloud Run injects PORT env (default 8080)
ENV PORT=8080

# --timeout-keep-alive 300 keeps SSE connections alive
CMD exec uvicorn api_server:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 300
