# ============================================================
# Indian Telecom Churn Prediction – Production Dockerfile
# Multi-stage build: smaller final image
# ============================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Install Python dependencies first (better layer caching)
# ----------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ----------------------------------------------------------
# Copy application code
# ----------------------------------------------------------
COPY app/ ./app/
COPY train_model.py .
COPY streamlit_app.py .
COPY saved_models/ ./saved_models/
COPY data/ ./data/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default port (FastAPI)
EXPOSE 8000

# Healthcheck for the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command = FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]