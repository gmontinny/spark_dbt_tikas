"""
Bronze Layer — Extração de Documentos com Apache Tika
======================================================
Lê arquivos PDF/DOCX da pasta datas/, extrai texto e metadados via Tika
e persiste como Delta/Parquet no MinIO (camada Bronze).

Execução:
    python jobs/bronze_tika_ingest.py
"""

import logging
import os
from pathlib import Path

import tika
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, lit
from pyspark.sql.types import StringType, StructField, StructType
from tika import parser as tika_parser

tika.TikaClientOnly = True
tika.ServerEndpoint = os.getenv("TIKA_SERVER_JAR", "http://tika-server:9998")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
DATAS_DIR = Path(os.getenv("DATAS_DIR", "/opt/spark/work-dir/datas"))
BRONZE_PATH = "s3a://warehouse/bronze/documents"
DATABASE = "bronze"

SCHEMA = StructType([
    StructField("file_name", StringType(), False),
    StructField("content_type", StringType(), True),
    StructField("text_content", StringType(), True),
    StructField("author", StringType(), True),
    StructField("title", StringType(), True),
    StructField("created", StringType(), True),
    StructField("language", StringType(), True),
    StructField("num_pages", StringType(), True),
])


def extract_with_tika(file_path: Path) -> dict:
    """Extrai texto e metadados de um arquivo via Tika REST."""
    parsed = tika_parser.from_file(str(file_path), serverEndpoint=tika.ServerEndpoint)
    meta = parsed.get("metadata", {}) or {}
    return {
        "file_name": file_path.name,
        "content_type": meta.get("Content-Type", ""),
        "text_content": (parsed.get("content") or "").strip(),
        "author": meta.get("Author", meta.get("dc:creator", "")),
        "title": meta.get("title", meta.get("dc:title", "")),
        "created": meta.get("Creation-Date", meta.get("dcterms:created", "")),
        "language": meta.get("language", ""),
        "num_pages": str(meta.get("xmpTPg:NPages", meta.get("Page-Count", ""))),
    }


def run() -> None:
    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

    supported = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".txt", ".html"}
    files = [f for f in DATAS_DIR.iterdir() if f.suffix.lower() in supported]

    if not files:
        log.warning("Nenhum arquivo encontrado em %s", DATAS_DIR)
        return

    log.info("Extraindo %d arquivo(s) via Tika...", len(files))
    rows = [extract_with_tika(f) for f in files]

    df = spark.createDataFrame(
        [[r["file_name"], r["content_type"], r["text_content"],
          r["author"], r["title"], r["created"], r["language"], r["num_pages"]]
         for r in rows],
        schema=SCHEMA,
    ).withColumn("ingested_at", current_timestamp())

    (
        df.write.format("parquet")
        .option("path", BRONZE_PATH)
        .mode("overwrite")
        .saveAsTable(f"{DATABASE}.documents")
    )
    log.info("✅ Bronze: %d documentos gravados em %s", len(rows), BRONZE_PATH)
    spark.stop()


if __name__ == "__main__":
    run()
