FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system dependencies including Redis and RabbitMQ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    redis-server \
    rabbitmq-server \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
ENV PATH="/usr/local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code
COPY . /app

# Ensure start.sh is executable
RUN chmod +x /app/start.sh

# Install project itself
RUN uv sync --frozen --no-dev

# Expose the API port
EXPOSE 7860

# Start everything via the shell script
CMD ["/app/start.sh"]