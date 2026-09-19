"""
Silver Layer — Feature Engineering e Embeddings
================================================
Lê a camada Bronze, realiza limpeza/normalização de texto,
gera embeddings semânticos com SentenceTransformers
e persiste na camada Silver.

Execução:
    python jobs/silver_features_embeddings.py
"""

import logging
import os
import re

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


def _embed(text: str) -> list[float]:
    import torch
    from functools import lru_cache
    from sentence_transformers import SentenceTransformer

    @lru_cache(maxsize=1)
    def _get_model(name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(name, device=device)

    return _get_model(EMBEDDING_MODEL).encode(text[:2048], normalize_embeddings=True).tolist()


def run() -> None:
    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

    clean_udf = udf(_clean_text, StringType())
    embed_udf = udf(_embed, ArrayType(FloatType()))

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

    df = spark.read.parquet(BRONZE_PATH)

    df_clean = (
        df.withColumn("text_clean", clean_udf(col("text_content")))
        .withColumn("text_clean", lower(trim(col("text_clean"))))
        .withColumn("text_length", length(col("text_clean")))
        .filter(col("text_length") > 50)
    )

    # TF-IDF via MLlib
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer

    tokenizer  = Tokenizer(inputCol="text_clean", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=512)
    idf        = IDF(inputCol="raw_features", outputCol="tfidf_features")

    df_words  = tokenizer.transform(df_clean)
    df_tf     = hashing_tf.transform(df_words)
    idf_model = idf.fit(df_tf)
    df_tfidf  = idf_model.transform(df_tf).drop("words", "raw_features")

    df_embedded = df_tfidf.withColumn("embedding", embed_udf(col("text_clean")))

    (
        df_embedded
        .drop("text_content")
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
