# syntax=docker/dockerfile:1
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    OMP_NUM_THREADS=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY docker/constraints.txt ./docker/constraints.txt
# Install the matched CUDA wheels first; constraints prevent replacement later.
RUN python -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install -r requirements.txt -c docker/constraints.txt \
    && python -m pip check

COPY head_mvt_classification_LIA.py ./
COPY LIA_encoder/ ./LIA_encoder/
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY tests/ ./tests/
RUN mkdir -p /app/output /app/LIA_encoder/checkpoints \
    && python head_mvt_classification_LIA.py --help \
    && python -m unittest discover -s tests -v

CMD ["python", "head_mvt_classification_LIA.py", "--help"]
