"""
Silver Layer — Feature Engineering e Embeddings
================================================
Lê a camada Bronze, realiza limpeza/normalização de texto,
gera TF-IDF via Spark MLlib e embeddings semânticos no driver
(torch disponível apenas no container dev, não no worker).

Execução:
    python jobs/silver_features_embeddings.py
"""

import logging
import os
import re

import torch
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, length, lower, trim, udf
from pyspark.sql.types import ArrayType, FloatType, StringType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL       = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
BRONZE_PATH     = "s3a://warehouse/bronze/documents"
SILVER_PATH     = "s3a://warehouse/silver/documents_features"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder")
DATABASE        = "silver"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s*32;\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.,;:!?\-àáâãéêíóôõúüçÀÁÂÃÉÊÍÓÔÕÚÜÇ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("🖥️  Dispositivo de inferência: %s", device)

    log.info("Carregando modelo de embeddings: %s", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

    clean_udf = udf(_clean_text, StringType())

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

    df = spark.read.parquet(BRONZE_PATH)

    df_clean = (
        df.withColumn("text_clean", clean_udf(col("text_content")))
        .withColumn("text_clean", lower(trim(col("text_clean"))))
        .withColumn("text_length", length(col("text_clean")))
        .filter(col("text_length") > 50)
    )

    # TF-IDF via MLlib — roda nos workers do Spark (sem torch)
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer

    tokenizer  = Tokenizer(inputCol="text_clean", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=512)
    idf        = IDF(inputCol="raw_features", outputCol="tfidf_features")

    df_words  = tokenizer.transform(df_clean)
    df_tf     = hashing_tf.transform(df_words)
    idf_model = idf.fit(df_tf)
    df_tfidf  = idf_model.transform(df_tf).drop("words", "raw_features")

    # Coleta no driver para gerar embeddings (torch só existe no dev)
    # tfidf_features é SparseVector — não serializável pelo PyArrow, descartado aqui
    log.info("Coletando textos para gerar embeddings no driver...")
    pdf = df_tfidf.drop("text_content", "tfidf_features").toPandas()

    texts = pdf["text_clean"].fillna("").tolist()
    log.info("Gerando embeddings para %d documentos...", len(texts))
    embeddings = embed_model.encode(
        [t[:2048] for t in texts],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()
    pdf["embedding"] = embeddings
    pdf["processed_at"] = __import__("datetime").datetime.utcnow()

    df_result = spark.createDataFrame(pdf)

    (
        df_result
        .withColumn("processed_at", current_timestamp())
        .write.format("parquet")
        .option("path", SILVER_PATH)
        .mode("overwrite")
        .saveAsTable(f"{DATABASE}.documents_features")
    )
    log.info("✅ Silver: embeddings gravados em %s", SILVER_PATH)
    spark.stop()


if __name__ == "__main__":
    run()
