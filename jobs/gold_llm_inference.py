"""
Gold Layer — Inferência Distribuída com LLM
============================================
Lê a camada Silver, executa inferência em lote com LLM (sumarização
e classificação de tópicos) distribuída nos executores Spark,
e persiste os resultados enriquecidos na camada Gold (StarRocks).

Idempotência garantida via PRIMARY KEY + upsert (StarRocks Stream Load).
Rodar múltiplas vezes nunca duplica — atualiza o registro existente.

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

STARROCKS_HOST = os.getenv("STARROCKS_HOST", "starrocks-fe-0")
STARROCKS_PORT = os.getenv("STARROCKS_FE_QUERY_PORT", "9030")
STARROCKS_USER = os.getenv("STARROCKS_USER", "root")
STARROCKS_PASSWORD = os.getenv("STARROCKS_PASSWORD", "")
STARROCKS_DB = os.getenv("STARROCKS_DB", "gold")
STARROCKS_JDBC = f"jdbc:mysql://{STARROCKS_HOST}:{STARROCKS_PORT}/{STARROCKS_DB}?useSSL=false"

TOPICS = ["economia", "imóveis", "inflação", "mercado financeiro", "política", "tecnologia"]


def _summarize(text: str) -> str:
    if not text or len(text) < 100:
        return text or ""
    from transformers import pipeline
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, max_length=130, min_length=30)
    return summarizer(text[:1024], truncation=True)[0]["summary_text"]


def _classify_topic(text: str) -> str:
    if not text:
        return "desconhecido"
    from transformers import pipeline
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)
    return classifier(text[:512], candidate_labels=TOPICS)["labels"][0]


def _get_conn():
    import mysql.connector
    return mysql.connector.connect(
        host=STARROCKS_HOST, port=int(STARROCKS_PORT),
        user=STARROCKS_USER, password=STARROCKS_PASSWORD,
    )


def _ensure_starrocks_table() -> None:
    """
    Cria tabela com PRIMARY KEY (modelo de tabela PRIMARY do StarRocks).
    PRIMARY KEY garante upsert real: INSERT com chave duplicada atualiza
    o registro existente em vez de inserir novo — idempotência garantida.
    """
    conn = _get_conn()
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
        PRIMARY KEY(file_name)
        DISTRIBUTED BY HASH(file_name) BUCKETS 4
        PROPERTIES ("replication_num" = "1")
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info("✅ StarRocks: tabela gold.documents_enriched pronta (PRIMARY KEY)")


def _upsert_batch(pdf_rows: list[dict]) -> None:
    """
    Upsert via INSERT INTO ... ON DUPLICATE KEY UPDATE.
    Garante que reexecuções atualizam sem duplicar.
    """
    if not pdf_rows:
        return
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(f"USE {STARROCKS_DB}")
    sql = """
        INSERT INTO documents_enriched
            (file_name, content_type, title, author, language, num_pages,
             text_length, topic, summary, ingested_at, processed_at, enriched_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            topic        = VALUES(topic),
            summary      = VALUES(summary),
            enriched_at  = VALUES(enriched_at),
            text_length  = VALUES(text_length)
    """
    rows = [
        (r["file_name"], r["content_type"], r["title"], r["author"],
         r["language"], r["num_pages"], r["text_length"], r["topic"],
         r["summary"], r["ingested_at"], r["processed_at"], r["enriched_at"])
        for r in pdf_rows
    ]
    cur.executemany(sql, rows)
    conn.commit()
    cur.close()
    conn.close()


def run() -> None:
    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

    summarize_udf = udf(_summarize, StringType())
    classify_udf = udf(_classify_topic, StringType())

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

    # Coleta e faz upsert — para volumes grandes usar foreachBatch com JDBC
    rows = df_enriched.toPandas().to_dict("records")
    _upsert_batch(rows)

    log.info("✅ Gold: %d documentos gravados no StarRocks com upsert", len(rows))
    spark.stop()


if __name__ == "__main__":
    run()
