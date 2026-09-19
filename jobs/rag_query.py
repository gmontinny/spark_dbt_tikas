"""
RAG — Retrieval-Augmented Generation
======================================
Etapa 1 — Retrieval: busca os chunks mais relevantes nos embeddings
           gerados na camada Silver (cosine similarity).
Etapa 2 — Augmented Generation: monta prompt com o contexto recuperado
           e envia para um LLM (Mistral-7B via HuggingFace ou OpenAI)
           que gera uma resposta fundamentada nos documentos.

Execução standalone:
    python jobs/rag_query.py --question "Qual foi a variação do IPCA em dezembro?"

Execução via Spark Connect (busca distribuída em grandes volumes):
    python jobs/rag_query.py --question "..." --spark
"""

import argparse
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
SILVER_PATH = os.getenv("SILVER_PATH", "s3a://warehouse/silver/documents_features")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _embed_query(question: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(question, normalize_embeddings=True)


def _load_embeddings_local() -> list[dict]:
    """Carrega embeddings do Silver via pandas (modo standalone)."""
    import pandas as pd
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()
    df = spark.read.parquet(SILVER_PATH).select(
        "file_name", "text_clean", "embedding"
    ).toPandas()
    spark.stop()
    return df.to_dict("records")


def retrieve(question: str, docs: list[dict]) -> list[dict]:
    """Retorna os TOP_K chunks mais similares à pergunta."""
    q_vec = _embed_query(question)
    scored = []
    for doc in docs:
        emb = np.array(doc["embedding"], dtype=np.float32)
        score = _cosine_similarity(q_vec, emb)
        scored.append({**doc, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:TOP_K]


def generate(question: str, chunks: list[dict]) -> str:
    """Gera resposta com LLM usando os chunks como contexto (RAG)."""
    context = "\n\n---\n\n".join(
        f"[{c['file_name']}]\n{c['text_clean'][:800]}" for c in chunks
    )
    prompt = (
        "Você é um assistente especializado em análise de documentos.\n"
        "Use APENAS o contexto abaixo para responder. Se não souber, diga que não encontrou nos documentos.\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PERGUNTA: {question}\n\n"
        "RESPOSTA:"
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    log.info("Carregando LLM %s...", LLM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def run(question: str) -> str:
    log.info("🔍 Recuperando chunks relevantes para: '%s'", question)
    docs = _load_embeddings_local()
    chunks = retrieve(question, docs)

    for i, c in enumerate(chunks, 1):
        log.info("  [%d] %s (score=%.4f)", i, c["file_name"], c["score"])

    log.info("🤖 Gerando resposta com LLM...")
    answer = generate(question, chunks)
    log.info("✅ Resposta gerada")
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Pergunta sobre os documentos")
    args = parser.parse_args()

    answer = run(args.question)
    print("\n" + "=" * 60)
    print("RESPOSTA:")
    print("=" * 60)
    print(answer)
