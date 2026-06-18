# syntax=docker/dockerfile:1.7
#
# Self-contained image for the Adaptive Booth API.
#   /app/main.py     FastAPI app (3 endpoints; /hall_with_booth_predict upgraded)
#   /app/pipeline    baked production detector package (detectors/, utils/) +
#                    support modules (logging_setup, trail_merger, image_hash_checker)
#   /app/adaptive    the adaptive engine (pipeline.py, config.py, tiling.py, ...)
#
# The only thing NOT in the image is the runtime .env (Roboflow + auth keys),
# supplied at `docker compose up` time.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    BOOTH_DETECTOR_ROOT=/app/pipeline \
    PYTHONPATH=/app/adaptive:/app/pipeline:/app \
    ONNXRUNTIME_EXECUTION_PROVIDERS=CPUExecutionProvider \
    OMP_NUM_THREADS=4 \
    LOG_DIR=/data/logs \
    HASH_DB_PATH=/data/hash_db.json \
    SAM2_BUILD_CUDA=0

# System libraries:
#   libgl1 / libglib2.0-0 / libgomp1 -> OpenCV runtime
#   poppler-utils                    -> PDF tooling
#   curl                             -> compose healthcheck
#   git / build-essential            -> source builds (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
        libgl1 libglib2.0-0 libgomp1 \
        poppler-utils curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch from PyTorch's dedicated index.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# 1) Build helpers + CPU torch first so later steps see them satisfied.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip setuptools wheel Cython ninja && \
    python -m pip install --extra-index-url ${TORCH_INDEX} \
        torch==2.5.1+cpu torchvision==0.20.1+cpu

# 2) Production stack MINUS the optional SAM-2 git line (single source of truth).
#    Copied alone so editing app code later does not bust this slow layer.
COPY requirements.txt /tmp/requirements.prod.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -ivE '^[[:space:]]*SAM-2[[:space:]]*@' /tmp/requirements.prod.txt \
        > /tmp/requirements.core.txt && \
    python -m pip install --extra-index-url ${TORCH_INDEX} -r /tmp/requirements.core.txt

# 3) EasyOCR for the raster per-booth OCR fallback (deps already satisfied).
COPY requirements.extra.txt /tmp/requirements.extra.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements.extra.txt

# 4) Application code (baked in -> self-contained image).
COPY app/ /app/

# Default data dirs (overlaid by the ./data bind mount at runtime).
RUN mkdir -p /data/in /data/out /data/logs

# Sanity: detector package + adaptive engine import cleanly at build time.
# (main.py is NOT imported here because it requires runtime auth/Roboflow env.)
RUN python -c "import detectors.ensemble_detector, detectors.opencv_detector, detectors.color_detector; print('detectors import OK')" && \
    python -c "import pipeline, config, tiling, labeling, input_profile, _detectors; print('adaptive engine import OK')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
