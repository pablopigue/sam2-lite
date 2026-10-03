# CPU image of the sam2-lite API. Build from the repo root:
#   docker build -f api/Dockerfile -t sam2-lite-api .
# Run with the local weights mounted (or -e HF_TOKEN=... to download the private model):
#   docker run --rm -p 8000:8000 -v $PWD/checkpoints:/app/checkpoints:ro sam2-lite-api
FROM python:3.11-slim

# ffmpeg: H.264 output the browser can play (D-043); git: sam-2 is installed from GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAM2_BUILD_CUDA=0

WORKDIR /app
# Dependencies first: this layer is cached and only rebuilt when requirements.txt changes.
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt \
    && pip install --no-build-isolation \
       "sam-2 @ git+https://github.com/facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4"

COPY src/sam2lite src/sam2lite
COPY configs/app.yaml configs/app.yaml
COPY api api
ENV PYTHONPATH=/app/src

# Do not run as root inside the container.
RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
