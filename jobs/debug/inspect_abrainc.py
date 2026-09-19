import warnings
warnings.filterwarnings("ignore")
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://spark-connect:15002").getOrCreate()
df = spark.read.parquet("s3a://warehouse/silver/documents_features").toPandas()
spark.stop()

for _, r in df.iterrows():
    if "ABRAINC" in r["file_name"]:
        text = r["text_clean"]
        print("FILE:", r["file_name"])
        print("LEN:", len(text))
        idx = text.lower().find("pib")
        print("PIB idx:", idx)
        if idx >= 0:
            print("CONTEXTO PIB:")
            print(text[max(0, idx - 100):idx + 600])
        print("---INICIO---")
        print(text[:1000])
        print("---MEIO (3000-6000)---")
        print(text[3000:6000])
