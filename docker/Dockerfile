# =============================================================================
# Code Intelligence Platform — Dockerfile
# =============================================================================
#
# Your setup (from .env):
#   - Supabase Cloud  (no local Postgres needed)
#   - Neo4j Aura      (no local Neo4j needed)
#   - Qdrant Cloud    (no local Qdrant needed)
#   - Gemini API      (Google AI, via API key)
#   - GitLab repos    (cloned at runtime)
#
# This image only contains the Python API server + PDF tooling.
# All databases are external cloud services reached via env vars.
#
# Build:
#   docker build -f docker/Dockerfile -t code-intel-api .
#
# Run:
#   docker run --env-file .env -p 8000:8000 code-intel-api
#
# =============================================================================


# -----------------------------------------------------------------------------
# Stage 1: BUILDER — install Python packages (thrown away after build)
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Don't write .pyc files, don't buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# We need build-essential + libffi-dev so pip can compile C extensions
# like cffi (needed by WeasyPrint) and grpcio (needed by Google Cloud).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (Python dependency manager)
RUN pip install poetry==2.1.3

WORKDIR /app

# Copy dependency manifests first (Docker layer caching: deps only rebuild
# when pyproject.toml or poetry.lock change, not when code changes).
COPY pyproject.toml poetry.lock* ./

# Tell Poetry to install into the system Python, not a virtualenv.
# --no-root  = don't install our own project as a package
# --only main = skip dev dependencies (pytest, black, ruff, etc.)
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only main --no-interaction --no-ansi


# -----------------------------------------------------------------------------
# Stage 2: RUNTIME — lean image with only what's needed to run
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE=1  — don't create .pyc files (smaller image)
# PYTHONUNBUFFERED=1         — print logs immediately, don't buffer
# PYTHONPATH=/app            — so "from src.xxx import" works
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── System libraries needed at RUNTIME ──
#
# curl & git:
#   curl  = health check endpoint probe
#   git   = clone GitLab repos at runtime (ingestion)
#
# WeasyPrint needs these C libraries to render HTML → PDF:
#   libcairo2          = 2D graphics engine (draws everything)
#   libpango-1.0-0     = text layout engine (handles fonts/wrapping)
#   libpangocairo-1.0-0= connects Pango text layout to Cairo renderer
#   libgdk-pixbuf2.0-0 = image loading (PNG/JPEG in PDFs)
#   shared-mime-info    = MIME type database (WeasyPrint needs it)
#
# Mermaid diagram rendering needs a headless browser:
#   chromium               = headless browser that renders diagrams
#   fonts-liberation       = standard fonts so diagrams have readable text
#   fonts-noto-color-emoji = emoji support in diagrams
#
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    chromium \
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js + Mermaid CLI ──
# Mermaid CLI (mmdc) converts Mermaid text → SVG/PNG diagrams.
# It needs Node.js to run and Chromium to render.
# PUPPETEER_SKIP_CHROMIUM_DOWNLOAD = use the system Chromium we installed above
# PUPPETEER_EXECUTABLE_PATH        = tell Puppeteer where Chromium lives
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @mermaid-js/mermaid-cli && \
    npm cache clean --force && \
    rm -rf /var/lib/apt/lists/*

# ── Copy Python packages from the builder stage ──
# This is the key multi-stage trick: we compiled everything in "builder"
# (which had build-essential, gcc, etc.) and now we just copy the
# resulting site-packages into this lean runtime image.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Security: run as non-root user ──
# Never run containers as root in production. We create a dedicated user
# so even if the app is compromised, the attacker has limited permissions.
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Copy application code ──
# Only copy src/ and config/ — NOT .env, data/, node_modules/, etc.
# The --chown flag makes appuser own these files (not root).
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup config/ ./config/

# ── Create writable directories ──
# data/repos    = where cloned GitLab repos are stored
# generated_pdfs = where PRD/QA PDF output files go
RUN mkdir -p /app/data/repos /app/generated_pdfs && \
    chown -R appuser:appgroup /app/data /app/generated_pdfs

# ── Puppeteer config for Docker ──
# Chromium inside Docker needs --no-sandbox because we run as non-root.
# --disable-dev-shm-usage avoids /dev/shm size issues in containers.
# This file is auto-detected by pdf_service.py via the env var.
ENV PUPPETEER_CONFIG_PATH=/app/puppeteer-config.json
RUN echo '{"executablePath":"/usr/bin/chromium","args":["--no-sandbox","--disable-setuid-sandbox","--disable-gpu","--disable-dev-shm-usage"]}' \
    > /app/puppeteer-config.json && \
    chown appuser:appgroup /app/puppeteer-config.json

# ── Switch to non-root user ──
USER appuser

# ── Expose the API port ──
EXPOSE 8000

# ── Health check ──
# Docker / Docker Desktop uses this to show container health status.
# Every 30s it calls the /live endpoint. If it fails 3 times, container
# is marked unhealthy. start-period gives the app 60s to boot up first.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/admin/live || exit 1

# ── Start the server ──
# uvicorn = ASGI server that runs our FastAPI app
# --host 0.0.0.0 = listen on all interfaces (required inside Docker)
# --port 8000    = match the EXPOSE above
# --workers 2    = 2 worker processes for handling concurrent requests
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
