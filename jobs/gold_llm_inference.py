"""
Gold Layer — Inferência com LLM
================================
Lê a camada Silver, executa sumarização e classificação zero-shot
no driver (após collect) e persiste os resultados no StarRocks.

Idempotência garantida via PRIMARY KEY + ON DUPLICATE KEY UPDATE.
Rodar múltiplas vezes nunca duplica — atualiza o registro existente.

Execução:
    python jobs/gold_llm_inference.py
"""

import logging
import os
from datetime import datetime

from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
SILVER_PATH = "s3a://warehouse/silver/documents_features"
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")

STARROCKS_HOST = os.getenv("STARROCKS_HOST", "starrocks-fe-0")
STARROCKS_PORT = os.getenv("STARROCKS_FE_QUERY_PORT", "9030")
STARROCKS_USER = os.getenv("STARROCKS_USER", "root")
STARROCKS_DB = os.getenv("STARROCKS_DB", "gold")

TOPICS = ["economia", "imóveis", "inflação", "mercado financeiro", "política", "tecnologia"]


def _get_conn():
    import mysql.connector
    return mysql.connector.connect(
        host=STARROCKS_HOST, port=int(STARROCKS_PORT),
        user=STARROCKS_USER,
        password=os.getenv("STARROCKS_PASSWORD", ""),
    )


def _ensure_starrocks_table() -> None:
    """
    Cria tabela com PRIMARY KEY (modelo PRIMARY do StarRocks).
    Garante upsert real: INSERT com chave duplicada atualiza o registro
    existente em vez de inserir novo — idempotência garantida.
    """
    with _get_conn() as conn:
        cur = conn.cursor()
        try:
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
        finally:
            cur.close()
    log.info("✅ StarRocks: tabela gold.documents_enriched pronta (PRIMARY KEY)")


def _to_str(v) -> str | None:
    """Converte pandas.Timestamp e outros tipos para str compatível com MySQL."""
    if v is None:
        return None
    import pandas as pd
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return str(v) if not isinstance(v, str) else v


def _upsert_batch(rows: list[tuple]) -> None:
    """Upsert via INSERT ... ON DUPLICATE KEY UPDATE — idempotente."""
    if not rows:
        return
    sql = """
        INSERT INTO documents_enriched
            (file_name, content_type, title, author, language, num_pages,
             text_length, topic, summary, ingested_at, processed_at, enriched_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    with _get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(f"USE {STARROCKS_DB}")
            for row in rows:
                cur.execute(sql, row)
            conn.commit()
        finally:
            cur.close()


def _build_pipelines():
    """Carrega os modelos uma única vez no driver."""
    from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

    tokenizer = BartTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = BartForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL)
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)
    return (tokenizer, model), classifier


def _summarize(summarizer, text: str) -> str:
    if not text or len(text) < 100:
        return text or ""
    tokenizer, model = summarizer
    inputs = tokenizer(
        text[:1024], return_tensors="pt", truncation=True, max_length=1024
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=130,
        min_new_tokens=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def _classify(classifier, text: str) -> str:
    if not text:
        return "desconhecido"
    return classifier(text[:512], candidate_labels=TOPICS, truncation=True)["labels"][0]


def run() -> None:
    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

    df = spark.read.parquet(SILVER_PATH).select(
        "file_name", "content_type", "title", "author",
        "language", "num_pages", "text_clean", "text_length",
        "ingested_at", "processed_at",
    )

    # Coleta no driver — inferência LLM é feita localmente (modelos grandes)
    records = df.toPandas().to_dict("records")
    spark.stop()

    _ensure_starrocks_table()

    log.info("🤖 Carregando modelos LLM no driver...")
    summarizer, classifier = _build_pipelines()

    enriched_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for r in records:
        text = r.get("text_clean") or ""
        summary = _summarize(summarizer, text)
        topic = _classify(classifier, text)
        log.info("  [%s] tópico=%s", r["file_name"], topic)
        rows.append((
            r["file_name"], r.get("content_type"), r.get("title"), r.get("author"),
            r.get("language"), r.get("num_pages"), r.get("text_length"),
            topic, summary,
            _to_str(r.get("ingested_at")), _to_str(r.get("processed_at")), enriched_at,
        ))

    _upsert_batch(rows)
    log.info("✅ Gold: %d documentos gravados no StarRocks com upsert", len(rows))


if __name__ == "__main__":
    run()
