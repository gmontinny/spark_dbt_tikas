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

# uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      pyspark>=4.0.0 \
      minio>=7.1.0 \
      kafka-python>=2.0.2 \
      psycopg2-binary>=2.9.0 \
      requests>=2.28.0 \
      pandas>=1.5.0 \
      pyarrow>=10.0.0 \
      streamlit>=1.49.1 \
      confluent-kafka>=2.11.1 \
      python-dotenv>=1.1.1 \
      grpcio>=1.74.0 \
      grpcio-status>=1.74.0 \
      zstandard>=0.25.0 \
      trino>=0.328.0 \
      dbt-trino>=1.8.0 \
      dbt-starrocks>=1.3.0 \
      tika>=2.6.0 \
      sentence-transformers>=2.2.0 \
      torch>=2.0.0 \
      transformers>=4.30.0 \
      langchain>=0.1.0 \
      langchain-community>=0.0.20 \
      numpy>=1.21.0 \
      scikit-learn>=1.0.0

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKA_SERVER_JAR=http://tika-server:9998
