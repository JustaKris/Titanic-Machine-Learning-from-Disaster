# ====================
# Stage 1: Builder
# ====================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files for layer caching
COPY pyproject.toml uv.lock /app/

# Create a venv for the app dependencies
RUN uv venv /opt/venv

# Export production deps and install into the venv
RUN uv export --frozen --no-hashes --no-emit-project \
        --no-group dev --no-group docs --no-group notebooks \
        > requirements.txt && \
    uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt

# ====================
# Stage 2: Runtime
# ====================
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    PORT=5000

# Minimal runtime deps (libgomp1 required by XGBoost/CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy app code and model artifacts
COPY src/ ./src/
COPY saved_models/ ./saved_models/

# Prepare writable directories expected by the app
RUN mkdir -p data/raw data/processed data/external logs

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["python", "-m", "titanic_ml.app.routes"]
