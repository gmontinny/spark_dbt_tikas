"""
Streaming Inference — Inferência Contínua com Structured Streaming
===================================================================
Consome novos documentos de um tópico Kafka (após extração Tika),
aplica classificação zero-shot em tempo real via LLM e grava
os resultados enriquecidos no StarRocks continuamente.

Fluxo:
    Kafka (tópico: tika.documents) → Spark Structured Streaming
    → LLM zero-shot classification → StarRocks (gold.documents_stream)

Execução:
    python jobs/streaming_inference.py
"""

import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, udf
from pyspark.sql.types import StringType, StructField, StructType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:9093")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_DOCUMENTS", "tika.documents")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")
CHECKPOINT_PATH = "s3a://warehouse/checkpoints/streaming_inference"

STARROCKS_HOST = os.getenv("STARROCKS_HOST", "starrocks-fe-0")
STARROCKS_PORT = os.getenv("STARROCKS_FE_QUERY_PORT", "9030")
STARROCKS_USER = os.getenv("STARROCKS_USER", "root")
STARROCKS_PASSWORD = os.getenv("STARROCKS_PASSWORD", "")
STARROCKS_DB = os.getenv("STARROCKS_DB", "gold")
STARROCKS_JDBC = f"jdbc:mysql://{STARROCKS_HOST}:{STARROCKS_PORT}/{STARROCKS_DB}?useSSL=false"

TOPICS = ["economia", "imóveis", "inflação", "mercado financeiro", "política", "tecnologia"]

# Schema da mensagem Kafka (JSON publicado pelo bronze_tika_ingest)
MESSAGE_SCHEMA = StructType([
    StructField("file_name", StringType()),
    StructField("text_content", StringType()),
    StructField("content_type", StringType()),
    StructField("title", StringType()),
])


def _classify(text: str) -> str:
    if not text:
        return "desconhecido"
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)
    result = classifier(text[:512], candidate_labels=TOPICS)
    return result["labels"][0]


def _write_to_starrocks(batch_df, batch_id: int) -> None:
    """foreachBatch: upsert de cada micro-batch no StarRocks."""
    if batch_df.isEmpty():
        return
    import mysql.connector
    conn = mysql.connector.connect(
        host=STARROCKS_HOST, port=int(STARROCKS_PORT),
        user=STARROCKS_USER, password=STARROCKS_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute(f"USE {STARROCKS_DB}")
    sql = """
        INSERT INTO documents_stream
            (file_name, content_type, title, topic, classified_at)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            topic         = VALUES(topic),
            classified_at = VALUES(classified_at)
    """
    rows = [
        (r["file_name"], r["content_type"], r["title"], r["topic"], r["classified_at"])
        for r in batch_df.toPandas().to_dict("records")
    ]
    cur.executemany(sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    log.info("Batch %d: %d registros → StarRocks (upsert)", batch_id, len(rows))


def _ensure_stream_table() -> None:
    import mysql.connector
    conn = mysql.connector.connect(
        host=STARROCKS_HOST, port=int(STARROCKS_PORT),
        user=STARROCKS_USER, password=STARROCKS_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {STARROCKS_DB}")
    cur.execute(f"USE {STARROCKS_DB}")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents_stream (
            file_name    VARCHAR(512) NOT NULL,
            content_type VARCHAR(128),
            title        VARCHAR(512),
            topic        VARCHAR(128),
            classified_at DATETIME
        )
        ENGINE = OLAP
        PRIMARY KEY(file_name)
        DISTRIBUTED BY HASH(file_name) BUCKETS 4
        PROPERTIES ("replication_num" = "1")
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info("✅ StarRocks: tabela gold.documents_stream pronta")


def run() -> None:
    _ensure_stream_table()

    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

    # UDF registrada APÓS a criação da sessão
    classify_udf = udf(_classify, StringType())

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.select(from_json(col("value").cast("string"), MESSAGE_SCHEMA).alias("data"))
        .select("data.*")
        .filter(col("text_content").isNotNull())
    )

    enriched = (
        parsed
        .withColumn("topic", classify_udf(col("text_content")))
        .withColumn("classified_at", current_timestamp())
        .drop("text_content")
    )

    query = (
        enriched.writeStream
        .foreachBatch(_write_to_starrocks)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="30 seconds")
        .start()
    )

    log.info("🚀 Streaming inference iniciado — aguardando mensagens em '%s'...", KAFKA_TOPIC)
    query.awaitTermination()


if __name__ == "__main__":
    run()
