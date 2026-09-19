"""
Gold Layer — Inferência Distribuída com LLM
============================================
Lê a camada Silver, executa inferência em lote com LLM (sumarização
e classificação de tópicos) distribuída nos executores Spark,
e persiste os resultados enriquecidos na camada Gold para consumo
pelo dbt e Trino.

Execução:
    python jobs/gold_llm_inference.py
"""

import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, udf
from pyspark.sql.types import StringType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
SILVER_PATH = "s3a://warehouse/silver/documents_features"
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")

# StarRocks — escrita via JDBC (MySQL-compatible)
STARROCKS_HOST = os.getenv("STARROCKS_HOST", "starrocks-fe-0")
STARROCKS_PORT = os.getenv("STARROCKS_FE_QUERY_PORT", "9030")
STARROCKS_USER = os.getenv("STARROCKS_USER", "root")
STARROCKS_PASSWORD = os.getenv("STARROCKS_PASSWORD", "")
STARROCKS_DB = os.getenv("STARROCKS_DB", "gold")
STARROCKS_JDBC = f"jdbc:mysql://{STARROCKS_HOST}:{STARROCKS_PORT}/{STARROCKS_DB}?useSSL=false"

TOPICS = ["economia", "imóveis", "inflação", "mercado financeiro", "política", "tecnologia"]


def _summarize(text: str) -> str:
    """Sumariza texto — executado em executor Spark."""
    if not text or len(text) < 100:
        return text or ""
    from transformers import pipeline
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, max_length=130, min_length=30)
    result = summarizer(text[:1024], truncation=True)
    return result[0]["summary_text"]


def _classify_topic(text: str) -> str:
    """Zero-shot classification — executado em executor Spark."""
    if not text:
        return "desconhecido"
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)
    result = classifier(text[:512], candidate_labels=TOPICS)
    return result["labels"][0]


summarize_udf = udf(_summarize, StringType())
classify_udf = udf(_classify_topic, StringType())


def _ensure_starrocks_table() -> None:
    """Cria banco e tabela no StarRocks se não existirem."""
    import mysql.connector

    conn = mysql.connector.connect(
        host=STARROCKS_HOST, port=int(STARROCKS_PORT),
        user=STARROCKS_USER, password=STARROCKS_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {STARROCKS_DB}")
    cur.execute(f"USE {STARROCKS_DB}")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents_enriched (
            file_name       VARCHAR(512)  NOT NULL,
            content_type    VARCHAR(128),
            title           VARCHAR(512),
            author          VARCHAR(256),
            language        VARCHAR(32),
            num_pages       VARCHAR(16),
            text_length     INT,
            topic           VARCHAR(128),
            summary         STRING,
            ingested_at     DATETIME,
            processed_at    DATETIME,
            enriched_at     DATETIME
        )
        ENGINE = OLAP
        DUPLICATE KEY(file_name)
        DISTRIBUTED BY HASH(file_name) BUCKETS 4
        PROPERTIES ("replication_num" = "1")
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info("✅ StarRocks: tabela gold.documents_enriched pronta")


def run() -> None:
    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

    df = spark.read.parquet(SILVER_PATH).select(
        "file_name", "content_type", "title", "author",
        "language", "num_pages", "text_clean", "text_length",
        "ingested_at", "processed_at",
    )

    df_enriched = (
        df
        .withColumn("summary", summarize_udf(col("text_clean")))
        .withColumn("topic", classify_udf(col("text_clean")))
        .withColumn("enriched_at", current_timestamp())
        .drop("text_clean")
    )

    _ensure_starrocks_table()

    # Escrita no StarRocks via JDBC (Spark → StarRocks FE)
    (
        df_enriched.write.format("jdbc")
        .option("url", STARROCKS_JDBC)
        .option("dbtable", "documents_enriched")
        .option("user", STARROCKS_USER)
        .option("password", STARROCKS_PASSWORD)
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .mode("overwrite")
        .save()
    )
    log.info("✅ Gold: documentos enriquecidos gravados no StarRocks (%s)", STARROCKS_DB)
    spark.stop()


if __name__ == "__main__":
    run()
