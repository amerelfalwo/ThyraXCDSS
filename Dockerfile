FROM python:3.12-slim

# Install system dependencies required for OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast package manager written in Rust)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation and virtual environment tracking for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy



# Copy the dependency files first to maximize Docker caching
COPY pyproject.toml uv.lock* ./

# Sync dependencies into a virtual environment
RUN uv sync --frozen --no-install-project --no-dev || uv sync --no-install-project --no-dev

# Add the virtual environment to PATH so we don't need to manually activate it
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"


# Copy the rest of the application code
COPY . .

# Install the project itself
RUN uv sync --no-dev

# إعطاء صلاحيات القراءة والكتابة للمستخدم العادي (مهم جداً لـ Hugging Face)
RUN chmod -R 777 /app

# تغيير البورت لـ 7860
EXPOSE 7860

# تشغيل uvicorn على بورت 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1"]