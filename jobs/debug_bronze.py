"""
Diagnóstico — verifica conteúdo do Bronze antes do Silver
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
BRONZE_PATH = "s3a://warehouse/bronze/documents"

spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

df = spark.read.parquet(BRONZE_PATH)
print("=== Schema ===")
df.printSchema()

print(f"\n=== Total de registros: {df.count()} ===")

print("\n=== Amostra (file_name, content_type, text_content length) ===")
df.select(
    col("file_name"),
    col("content_type"),
    length(col("text_content")).alias("text_len"),
    col("text_content").substr(1, 200).alias("text_preview"),
).show(10, truncate=False)

print("\n=== Registros com text_content nulo ou vazio ===")
df.filter(col("text_content").isNull() | (length(col("text_content")) == 0)).select("file_name").show()

spark.stop()
