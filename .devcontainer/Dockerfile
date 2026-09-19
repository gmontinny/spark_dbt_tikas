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
      pyspark==4.0.1 \
      minio==7.2.15 \
      kafka-python==2.0.2 \
      psycopg2-binary==2.9.10 \
      requests==2.32.3 \
      pandas==2.2.3 \
      pyarrow==18.1.0 \
      streamlit==1.45.1 \
      confluent-kafka==2.11.0 \
      python-dotenv==1.1.0 \
      grpcio==1.74.0 \
      grpcio-status==1.74.0 \
      zstandard==0.23.0 \
      trino==0.328.0 \
      dbt-trino==1.9.2 \
      dbt-starrocks==1.3.0 \
      tika==2.6.0 \
      sentence-transformers==3.4.1 \
      torch==2.5.1 \
      transformers==4.51.3 \
      langchain==0.3.25 \
      langchain-community==0.3.24 \
      numpy==1.26.4 \
      scikit-learn==1.6.1 \
      mysql-connector-python==9.3.0

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKA_SERVER_JAR=http://tika-server:9998
