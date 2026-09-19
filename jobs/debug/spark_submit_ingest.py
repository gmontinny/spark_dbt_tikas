"""
Exemplo 2 — spark-submit
========================
Script self-contained submetido via `spark-submit` diretamente ao
cluster Spark. Não depende de conexão gRPC — o driver roda dentro do
container com acesso direto ao Hive Metastore e ao MinIO.

Execução via docker-compose (ver Makefile):
    docker compose exec spark-connect \
        spark-submit \
        --master local[*] \
        --packages org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
        /opt/spark/work-dir/jobs/spark_submit_ingest.py

Diferenças em relação ao PySpark Connect:
  - SparkSession criada localmente (builder.getOrCreate), não remota
  - Configurações S3/Hive passadas diretamente na sessão ou via spark-defaults.conf
  - Ideal para pipelines agendados (Airflow, cron) sem servidor Connect ativo
"""

import logging
import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_timestamp, trim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
_default_data_dir = Path(__file__).parent.parent / "data"
DATA_DIR = os.getenv("DATA_DIR", _default_data_dir.as_uri())
WAREHOUSE = "s3a://warehouse"
DATABASE = "raw_olist_submit"

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
HIVE_METASTORE_URI = os.getenv("HIVE_METASTORE_URI", "thrift://hive-metastore:9083")

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
    """
    Cria SparkSession local com suporte a S3A (MinIO) e Hive Metastore.
    As configs de S3 e Hive já estão em spark-defaults.conf; aqui ficam
    apenas as que precisam de valores de runtime (env vars).
    """
    log.info("Iniciando SparkSession local (spark-submit)")
    return (
        SparkSession.builder.appName("spark-submit-olist-ingest")
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.hadoop.hive.metastore.uris", HIVE_METASTORE_URI)
        .config("spark.sql.warehouse.dir", WAREHOUSE)
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .enableHiveSupport()
        .getOrCreate()
    )


def read_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(path)
    )


def clean(df: DataFrame, timestamp_cols: list[str]) -> DataFrame:
    string_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()"]
    for c in string_cols:
        df = df.withColumn(c, trim(col(c)))
    for c in timestamp_cols:
        if c in df.columns:
            df = df.withColumn(c, to_timestamp(col(c)))
    return df


def write_hive(df: DataFrame, spark: SparkSession, table: str) -> None:
    """Grava Parquet no MinIO e registra no Hive Metastore."""
    full_table = f"{DATABASE}.{table}"
    location = f"{WAREHOUSE}/{DATABASE}.db/{table}"
    log.info("Gravando %s → %s", full_table, location)
    (
        df.write.format("parquet")
        .option("path", location)
        .mode("overwrite")
        .saveAsTable(full_table)
    )
    count = spark.table(full_table).count()
    log.info("✓ %s gravada (%d linhas)", full_table, count)


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
        write_hive(df, spark, table)

    log.info("✅ Ingestão via spark-submit concluída.")
    spark.stop()


if __name__ == "__main__":
    run()
