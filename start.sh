#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# ThyraX CDSS — Production Startup Script
# ─────────────────────────────────────────────────────────────
# Starts all infrastructure services (Redis, RabbitMQ, Celery)
# then launches the FastAPI app via Gunicorn with UvicornWorkers.
#
# Environment Variables (all optional, sensible defaults):
#   GUNICORN_WORKERS          — Number of worker processes (default: 4)
#   GUNICORN_TIMEOUT          — Worker timeout in seconds  (default: 120)
#   GUNICORN_BIND             — Host:port to bind          (default: 0.0.0.0:7860)
#   GUNICORN_LOG_LEVEL        — Log verbosity              (default: info)
#   GUNICORN_MAX_REQUESTS     — Requests before restart    (default: 1000)
#   GUNICORN_GRACEFUL_TIMEOUT — Graceful shutdown timeout   (default: 30)
# ─────────────────────────────────────────────────────────────

echo "============================================="
echo "  ThyraX CDSS — Starting Services"
echo "============================================="

# ── 1. Start Redis ──
echo "[1/4] Starting Redis..."
redis-server --daemonize yes
echo "       Redis started ✓"

# ── 2. Start RabbitMQ ──
echo "[2/4] Starting RabbitMQ..."
rabbitmq-server -detached
echo "       Waiting for RabbitMQ to initialize..."
sleep 5
echo "       RabbitMQ started ✓"

# ── 3. Start Celery Workers ──
echo "[3/4] Starting Celery Worker + Beat..."
celery -A app.core.celery_app worker --loglevel=info &
celery -A app.core.celery_app beat --loglevel=info &
echo "       Celery started ✓"

# ── 4. Start FastAPI via Gunicorn ──
echo "[4/4] Starting Gunicorn with UvicornWorkers..."
echo "       Config: gunicorn.conf.py"
echo "       Workers: ${GUNICORN_WORKERS:-4}"
echo "       Timeout: ${GUNICORN_TIMEOUT:-120}s"
echo "       Bind:    ${GUNICORN_BIND:-0.0.0.0:7860}"
echo "============================================="

# exec replaces the shell process with gunicorn, ensuring
# proper signal forwarding (SIGTERM, SIGINT) for graceful shutdown.
exec uv run gunicorn main:app -c gunicorn.conf.py
