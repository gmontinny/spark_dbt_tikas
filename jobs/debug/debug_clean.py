import os, re
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(os.getenv("SPARK_CONNECT_URL", "sc://spark-connect:15002")).getOrCreate()
raw = spark.read.parquet("s3a://warehouse/bronze/documents") \
    .filter("file_name = 'Boletim_Economico_ABRAINC_2tri25.pdf'") \
    .select("text_content").toPandas().iloc[0]["text_content"] or ""
spark.stop()

# Simula a limpeza atual
def clean_current(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s*32;\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.,;:!?\-àáâãéêíóôõúüçÀÁÂÃÉÊÍÓÔÕÚÜÇ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

cleaned = clean_current(raw)
print(f"Bronze : {len(raw)} chars")
print(f"Silver : {len(cleaned)} chars")
print(f"Perdido: {len(raw) - len(cleaned)} chars ({(len(raw)-len(cleaned))/len(raw)*100:.1f}%)")

# Mostra trecho com PIB no Bronze
idx = raw.lower().find("pib construção")
if idx == -1:
    idx = raw.lower().find("pib da construção")
if idx == -1:
    idx = raw.lower().find("pib construcao")

print(f"\nBusca 'pib construção' no Bronze: posição {idx}")
if idx >= 0:
    print(raw[max(0,idx-200):idx+600])

# Mostra o mesmo trecho no Silver
idx2 = cleaned.lower().find("pib constru")
print(f"\nBusca 'pib constru' no Silver: posição {idx2}")
if idx2 >= 0:
    print(cleaned[max(0,idx2-200):idx2+600])

# Mostra caracteres removidos pela regex
removed = set(raw) - set(cleaned)
print(f"\nCaracteres removidos pela limpeza: {sorted(removed)}")
