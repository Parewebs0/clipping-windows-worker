# CPU fallback worker (mini PC). The Windows GPU worker keeps running natively.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf XDG_CACHE_HOME=/models/cache
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates \
      libgl1 libglib2.0-0 libegl1 libgles2 \
 && rm -rf /var/lib/apt/lists/*
# CPU-only torch: the cpu index wins (x.y.z+cpu sorts above x.y.z), so
# whisperx does not pull the CUDA wheels.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple -r requirements.txt \
 && python -c "import torch; assert '+cpu' in torch.__version__, torch.__version__"

# Face detector model for reframe face mode (T1). Baked outside /models, which
# compose mounts as a volume. mediapipe needs libgl1/libglib2.0-0 (opencv) and
# libegl1/libgles2 (loaded even on CPU).
ARG FACE_MODEL_URL=https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite
ARG FACE_MODEL_SHA256=3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b
ENV REFRAME_FACE_MODEL=/opt/reframe-models/blaze_face_full_range.tflite
RUN mkdir -p /opt/reframe-models \
 && python -c "import urllib.request as u; u.urlretrieve('${FACE_MODEL_URL}', '${REFRAME_FACE_MODEL}')" \
 && echo "${FACE_MODEL_SHA256}  ${REFRAME_FACE_MODEL}" | sha256sum -c - \
 && python -c "from mediapipe.tasks.python.vision import FaceDetector; print('mediapipe ok')"

# gog CLI for Drive downloads; config/keyring are bind-mounted (see compose).
ARG GOG_VERSION=0.40.0
ARG GOG_SHA256_AMD64=5f73815950f30de4165b7b767103ca45c4950e84a1da601eda5294e9ff94f767
ARG GOG_SHA256_ARM64=21ca9757f67a573115b517854184561cef6b3b73c21e0f60c72229522c7198ac
ARG TARGETARCH
COPY docker/fetch_gog.py /tmp/fetch_gog.py
RUN set -eu; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$arch" in amd64) sha="$GOG_SHA256_AMD64" ;; arm64) sha="$GOG_SHA256_ARM64" ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; esac; \
    python /tmp/fetch_gog.py \
      "https://github.com/openclaw/gogcli/releases/download/v${GOG_VERSION}/gogcli_${GOG_VERSION}_linux_${arch}.tar.gz" \
      "$sha" /usr/local/bin/gog; \
    chmod 0755 /usr/local/bin/gog; rm /tmp/fetch_gog.py; gog --version

COPY . .
CMD ["python", "run.py"]
