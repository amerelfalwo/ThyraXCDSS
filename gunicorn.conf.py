"""
Gunicorn Configuration — Production Process Manager for ThyraX CDSS.

This file is loaded automatically by gunicorn when invoked with:
    gunicorn -c gunicorn.conf.py main:app

Architecture:
    gunicorn (master process)
      ├── UvicornWorker 1  ← async event loop + threadpool for ONNX
      ├── UvicornWorker 2
      ├── UvicornWorker 3
      └── UvicornWorker 4

Why Gunicorn + Uvicorn Workers?
    - Gunicorn handles process lifecycle (fork, supervise, restart on crash).
    - Each UvicornWorker runs its own asyncio event loop.
    - CPU-bound ONNX inference runs in per-worker threadpools via
      starlette.concurrency.run_in_threadpool (see app/services/inference.py).
    - If one worker crashes (OOM, segfault in ONNX), gunicorn restarts
      it without affecting the other workers.

Tuning Notes:
    - Workers = 2 × CPU + 1 is the general formula, but for heavy CV
      inference we cap at 4 to avoid RAM exhaustion (each worker loads
      its own copy of ONNX models into memory).
    - Timeout is set high (120s) because U-Net segmentation +
      classification + LLM explanation can take 30-60s on CPU.
    - Graceful timeout gives workers 30s to finish in-flight requests
      before being forcefully killed on shutdown.
    - Preload is DISABLED because ONNX models use lru_cache and
      forking after model load can cause issues with numpy/cv2.
"""

import multiprocessing
import os

# ═══════════════════════════════════════════════════════════════
# Server Socket
# ═══════════════════════════════════════════════════════════════

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:7860")

# ═══════════════════════════════════════════════════════════════
# Worker Configuration
# ═══════════════════════════════════════════════════════════════

# Use UvicornWorker for async FastAPI support
worker_class = "uvicorn.workers.UvicornWorker"

# Number of worker processes
# Formula: min(2 * CPU_COUNT + 1, MAX_WORKERS)
# Capped at 4 to prevent RAM exhaustion from duplicated ONNX models
_cpu_count = multiprocessing.cpu_count()
_max_workers = int(os.getenv("GUNICORN_WORKERS", "4"))
workers = min(2 * _cpu_count + 1, _max_workers)

# ═══════════════════════════════════════════════════════════════
# Timeouts
# ═══════════════════════════════════════════════════════════════

# Worker timeout (seconds) — kill worker if it doesn't respond.
# Set high for ONNX inference + LLM explanation chains.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# Graceful shutdown timeout — time to finish in-flight requests
# before the worker is forcefully terminated.
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# Keep-alive connections timeout (seconds)
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# ═══════════════════════════════════════════════════════════════
# Process Naming & Management
# ═══════════════════════════════════════════════════════════════

# Process title (visible in `ps aux`)
proc_name = "thyrax-cdss"

# Do NOT preload the app — each worker loads models independently.
# This avoids issues with numpy/cv2/ONNX after fork().
preload_app = False

# Max requests before worker restart (prevents memory leaks from
# long-running ONNX sessions). Jitter prevents thundering herd.
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════

# Access log format (JSON-like for structured logging)
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# Log format: timestamp, method, path, status, response time
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sμs'
)

# ═══════════════════════════════════════════════════════════════
# Security
# ═══════════════════════════════════════════════════════════════

# Limit request line size (prevent DoS via huge URLs)
limit_request_line = 8190

# Limit total header size
limit_request_fields = 100
limit_request_field_size = 8190

# ═══════════════════════════════════════════════════════════════
# Server Hooks (lifecycle callbacks)
# ═══════════════════════════════════════════════════════════════

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info(
        f"ThyraX CDSS starting — workers={workers}, "
        f"timeout={timeout}s, bind={bind}"
    )


def post_fork(server, worker):
    """Called after a worker has been forked."""
    server.log.info(f"Worker spawned: pid={worker.pid}")


def worker_exit(server, worker):
    """Called when a worker exits (crash or graceful shutdown)."""
    server.log.info(f"Worker exited: pid={worker.pid}")
