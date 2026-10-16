# CPU fallback worker (mini PC). The Windows GPU worker keeps running natively.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf XDG_CACHE_HOME=/models/cache
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# CPU-only torch first so whisperx does not pull the CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
COPY requirements.txt .
RUN pip install -r requirements.txt

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
