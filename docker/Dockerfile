# Multi-stage build for COVENANT.AI
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production image
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 covenant && \
    mkdir -p /app /app/data /app/logs && \
    chown -R covenant:covenant /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=covenant:covenant . /app/

# Install the package
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER covenant

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "covenant.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

