from pyspark.sql import SparkSession
spark = SparkSession.builder.remote("sc://spark-connect:15002").getOrCreate()
df = spark.read.parquet("s3a://warehouse/silver/documents_features").toPandas()
spark.stop()
for _, r in df.iterrows():
    if "ABRAINC" in r["file_name"]:
        print("embedding dims:", len(r["embedding"]))
