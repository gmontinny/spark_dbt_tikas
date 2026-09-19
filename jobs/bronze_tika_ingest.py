"""
Bronze Layer — Extração de Documentos com Apache Tika
======================================================
Lê arquivos PDF/DOCX da pasta datas/, extrai texto e metadados via
tika-python (modo cliente REST) e persiste como Parquet no MinIO.

Nota: tika-python 3.x requer service="text" para retornar o conteúdo
corretamente via endpoint /tika do servidor REST.

Execução:
    python jobs/bronze_tika_ingest.py
"""

import logging
import os
from pathlib import Path

import tika
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import StringType, StructField, StructType
from tika import parser as tika_parser

# Modo cliente: usa o servidor REST em vez de baixar o JAR localmente
os.environ["TIKA_CLIENT_ONLY"] = "True"
tika.TikaClientOnly = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
DATAS_DIR = Path(os.getenv("DATAS_DIR", "/opt/spark/work-dir/datas"))
TIKA_ENDPOINT = os.getenv("TIKA_SERVER_JAR", "http://tika-server:9998")
BRONZE_PATH = "s3a://warehouse/bronze/documents"
DATABASE = "bronze"
SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".txt", ".html"}

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


def _str(v) -> str:
    """Normaliza valores que podem ser lista ou None."""
    if isinstance(v, list):
        return v[0] if v else ""
    return str(v) if v else ""


def extract_with_tika(file_path: Path) -> dict:
    """
    Extrai texto e metadados via tika-python em modo cliente REST.
    service="text" usa o endpoint /tika que retorna conteúdo corretamente
    na versão 3.x da biblioteca.
    """
    parsed = tika_parser.from_file(
        str(file_path),
        serverEndpoint=TIKA_ENDPOINT,
        service="text",
    )
    meta = parsed.get("metadata", {}) or {}
    return {
        "file_name": file_path.name,
        "content_type": _str(meta.get("Content-Type", "")),
        "text_content": (parsed.get("content") or "").strip(),
        "author": _str(meta.get("Author", meta.get("dc:creator", ""))),
        "title": _str(meta.get("title", meta.get("dc:title", ""))),
        "created": _str(meta.get("Creation-Date", meta.get("dcterms:created", ""))),
        "language": _str(meta.get("language", "")),
        "num_pages": _str(meta.get("xmpTPg:NPages", meta.get("Page-Count", ""))),
    }


def run() -> None:
    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

    files = [f for f in DATAS_DIR.iterdir() if f.suffix.lower() in SUPPORTED]
    if not files:
        log.warning("Nenhum arquivo encontrado em %s", DATAS_DIR)
        return

    log.info("Extraindo %d arquivo(s) via tika-python...", len(files))
    rows = []
    for f in files:
        log.info("  → %s", f.name)
        row = extract_with_tika(f)
        log.info("    texto: %d chars", len(row["text_content"]))
        rows.append(row)

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
