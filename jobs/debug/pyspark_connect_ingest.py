"""
Exemplo 1 — PySpark Connect (cliente remoto)
============================================
Conecta ao Spark Connect Server via gRPC (sc://spark-connect:15002),
lê os CSVs do dataset Olist e grava tabelas Parquet no Hive Metastore
(MinIO), tornando-as consultáveis pelo Trino via catálogo `hive`.

Execução (dentro do devcontainer ou host com pyspark instalado):
    python jobs/pyspark_connect_ingest.py
"""

import logging
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_timestamp, trim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
SPARK_CONNECT_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
# No PySpark Connect o servidor (container) lê os arquivos — usar sempre o path do container.
# Para rodar no host, os dados ficam montados em /opt/spark/work-dir/data via docker-compose.
DATA_DIR = os.getenv("DATA_DIR", "/opt/spark/work-dir/data")
WAREHOUSE = "s3a://warehouse"
DATABASE = "raw_olist"

DATASETS: dict[str, dict] = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "timestamp_cols": [],
    },
    "orders": {
        "file": "olist_orders_dataset.csv",
        "timestamp_cols": [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "timestamp_cols": ["shipping_limit_date"],
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "timestamp_cols": [],
    },
    "order_reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "timestamp_cols": ["review_creation_date", "review_answer_timestamp"],
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "timestamp_cols": [],
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "timestamp_cols": [],
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "timestamp_cols": [],
    },
    "product_category_name_translation": {
        "file": "product_category_name_translation.csv",
        "timestamp_cols": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_spark() -> SparkSession:
    """Cria sessão remota via Spark Connect."""
    log.info("Conectando ao Spark Connect em %s", SPARK_CONNECT_URL)
    return SparkSession.builder.remote(SPARK_CONNECT_URL).getOrCreate()


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(path)
    )


def clean(df: DataFrame, timestamp_cols: list[str]) -> DataFrame:
    """Trim em strings e converte colunas de timestamp."""
    string_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()"]
    for c in string_cols:
        df = df.withColumn(c, trim(col(c)))
    for c in timestamp_cols:
        if c in df.columns:
            df = df.withColumn(c, to_timestamp(col(c)))
    return df


def write_hive(df: DataFrame, table: str) -> None:
    """Grava como Parquet no MinIO e registra no Hive Metastore."""
    full_table = f"{DATABASE}.{table}"
    location = f"{WAREHOUSE}/{DATABASE}.db/{table}"
    log.info("Gravando %s → %s", full_table, location)
    (
        df.write.format("parquet")
        .option("path", location)
        .mode("overwrite")
        .saveAsTable(full_table)
    )
    log.info("✓ %s gravada (%d linhas)", full_table, df.count())


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run() -> None:
    spark = build_spark()

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

    for table, cfg in DATASETS.items():
        path = f"{DATA_DIR}/{cfg['file']}"
        log.info("Lendo %s", path)
        df = read_csv(spark, path)
        df = clean(df, cfg["timestamp_cols"])
        write_hive(df, table)

    log.info("✅ Ingestão via PySpark Connect concluída.")
    spark.stop()


if __name__ == "__main__":
    run()
