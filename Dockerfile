# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ── Builder stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv "$HOME/.local/bin/uv"  /usr/local/bin/uv && \
    mv "$HOME/.local/bin/uvx" /usr/local/bin/uvx

# Copy dependency manifest first for layer-cache efficiency
COPY pyproject.toml .

ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies only (project not yet copied — maximises cache hits)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

# Copy full project, then install the project itself
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# Generate synthetic audio data inside the container
RUN uv run python generate_data.py

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# libsndfile1 is a C library required by soundfile at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
