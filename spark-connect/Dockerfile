FROM apache/spark:4.0.1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip openjdk-21-jre-headless curl && \
    pip3 install --no-cache-dir \
      pyspark==4.0.1 \
      pandas>=2.0.0 \
      pyarrow \
      grpcio>=1.74.0 \
      grpcio-status>=1.74.0 \
      zstandard>=0.25.0 \
      requests \
      tika>=2.6.0 \
      sentence-transformers>=2.2.0 \
      torch>=2.0.0 \
      transformers>=4.30.0 \
      numpy \
      mysql-connector-python && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3
ENV TIKA_SERVER_JAR=http://tika-server:9998

COPY ./conf/spark-defaults.conf "$SPARK_HOME/conf/spark-defaults.conf"
COPY ./conf/core-site.xml "$SPARK_HOME/conf/core-site.xml"
