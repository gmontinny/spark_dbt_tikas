FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

RUN --mount=type=cache,target=/var/cache/apt \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      curl make bash git ca-certificates procps \
      openjdk-21-jre-headless netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-hashes --no-editable -o /tmp/req.txt && \
    uv pip install --system -r /tmp/req.txt && \
    pip install openai -q

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKA_SERVER_JAR=http://tika-server:9998
