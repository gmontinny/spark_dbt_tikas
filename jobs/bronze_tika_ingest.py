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
DATAS_DIR = Path(os.getenv("DATAS_DIR", "/workspace/datas"))
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
    Extrai texto via endpoint /tika (service='text') e metadados via
    endpoint /meta (service='meta'). Duas chamadas são necessárias pois
    service='text' não retorna metadados e service='all' não retorna conteúdo.
    """
    text_result = tika_parser.from_file(
        str(file_path), serverEndpoint=TIKA_ENDPOINT, service="text",
    )
    meta_result = tika_parser.from_file(
        str(file_path), serverEndpoint=TIKA_ENDPOINT, service="meta",
    )
    meta = meta_result.get("metadata") or {}
    return {
        "file_name": file_path.name,
        "content_type": _str(meta.get("Content-Type", "")),
        "text_content": (text_result.get("content") or "").strip(),
        "author": _str(meta.get("dc:creator", meta.get("pdf:docinfo:creator", meta.get("Author", "")))),
        "title": _str(meta.get("dc:title", meta.get("title", ""))),
        "created": _str(meta.get("dcterms:created", meta.get("Creation-Date", ""))),
        "language": _str(meta.get("dc:language", meta.get("language", ""))),
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
        log.info("    texto: %d chars | autor: %s | páginas: %s | idioma: %s",
                 len(row["text_content"]), row["author"] or "(vazio)",
                 row["num_pages"] or "(vazio)", row["language"] or "(vazio)")
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
