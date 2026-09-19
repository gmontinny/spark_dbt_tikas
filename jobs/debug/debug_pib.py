import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(os.getenv("SPARK_CONNECT_URL", "sc://spark-connect:15002")).getOrCreate()
pdf = spark.read.parquet("s3a://warehouse/silver/documents_features") \
    .filter("file_name = 'Boletim_Economico_ABRAINC_2tri25.pdf'") \
    .select("text_clean", "text_length").toPandas()
spark.stop()

text = pdf.iloc[0]["text_clean"] or ""
print(f"Tamanho total do texto: {len(text)} chars")
print(f"text_length coluna: {pdf.iloc[0]['text_length']}")

# Busca por PIB
idx = text.lower().find("pib")
if idx == -1:
    print("\n❌ 'pib' NÃO encontrado no text_clean")
else:
    print(f"\n✅ 'pib' encontrado na posição {idx}")
    print("Contexto:")
    print(text[max(0, idx-100):idx+500])

# Busca no Bronze para comparar
spark2 = SparkSession.builder.remote(os.getenv("SPARK_CONNECT_URL", "sc://spark-connect:15002")).getOrCreate()
pdf_bronze = spark2.read.parquet("s3a://warehouse/bronze/documents") \
    .filter("file_name = 'Boletim_Economico_ABRAINC_2tri25.pdf'") \
    .select("text_content").toPandas()
spark2.stop()

raw = pdf_bronze.iloc[0]["text_content"] or ""
print(f"\nBronze text_content tamanho: {len(raw)} chars")
idx2 = raw.lower().find("pib")
if idx2 == -1:
    print("❌ 'pib' NÃO encontrado no Bronze text_content")
else:
    print(f"✅ 'pib' encontrado no Bronze na posição {idx2}")
    print("Contexto Bronze:")
    print(raw[max(0, idx2-100):idx2+500])
