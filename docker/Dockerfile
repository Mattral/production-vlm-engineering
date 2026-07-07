# Lightweight CPU image: runs all five example pipelines in their
# honest CPU-fallback mode (no GPU, no torch/transformers required).
# For real GPU fine-tuning/export, see docker/Dockerfile.gpu.
FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib (chart rendering) and pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
COPY benchmarks ./benchmarks

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev,cli]"

ENV PYTHONUNBUFFERED=1

# Default: run the fastest example end-to-end as a container smoke test.
CMD ["python", "-m", "production_vlm.cli", "run-example", "embedding_drift_active_learning"]
