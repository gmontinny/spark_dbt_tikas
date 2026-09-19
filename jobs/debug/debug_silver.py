import os
from pyspark.sql import SparkSession

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://spark-connect:15002")
SILVER_PATH = "s3a://warehouse/silver/documents_features"

spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()
df = spark.read.parquet(SILVER_PATH)

print("=== SCHEMA ===")
df.printSchema()

print("\n=== CONTAGEM ===")
print("Rows:", df.count())

pdf = df.select("file_name", "text_length", "text_clean", "embedding").toPandas()
for _, r in pdf.iterrows():
    emb = r["embedding"]
    print(f"\nFILE : {r['file_name']}")
    print(f"LEN  : {r['text_length']}")
    emb_len = len(emb) if emb is not None else 'None'
    print(f"EMB  : dims={emb_len}")
    print(f"TEXT : {str(r['text_clean'])[:400]}")

spark.stop()
