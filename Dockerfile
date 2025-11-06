# Multi-stage build to keep the runtime image slim
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.1 /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/
COPY models/ ./models/

# Install dependencies and project
RUN uv sync --frozen --no-dev

###############################
# Runtime image
###############################
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=false

WORKDIR /app

# Install runtime system libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy uv from builder
COPY --from=ghcr.io/astral-sh/uv:0.9.1 /uv /usr/local/bin/uv

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy project code
COPY src/ ./src/
COPY models/ ./models/

# Prepare writable directories expected by the app
RUN mkdir -p data/raw data/processed data/external logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Health check endpoint is `/health`
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Launch the Flask application using the venv Python
CMD ["/app/.venv/bin/python", "-m", "src.app.routes"]
