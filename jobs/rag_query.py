"""
RAG — Retrieval-Augmented Generation
======================================
Suporta múltiplos providers de LLM via LLM_PROVIDER:
  - openai   → OpenAI API (gpt-4o-mini por padrão)
  - gemini   → Google Gemini API (gemini-1.5-flash por padrão)
  - ollama   → Ollama local no host (llama3.1 por padrão)

Execução standalone:
    python jobs/rag_query.py --question "Qual foi a variação do IPCA em dezembro?"
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
SILVER_PATH = os.getenv("SILVER_PATH", "s3a://warehouse/silver/documents_features")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _embed_query(question: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(question, normalize_embeddings=True)


def _load_embeddings_local() -> list[dict]:
    """Carrega embeddings do Silver via Spark Connect."""
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
    scored = [
        {**doc, "score": _cosine_similarity(q_vec, np.array(doc["embedding"], dtype=np.float32))}
        for doc in docs
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:TOP_K]


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['file_name']}]\n{c['text_clean'][:3000]}" for c in chunks
    )
    return (
        "Você é um assistente especializado em análise de documentos em português brasileiro.\n"
        "Com base APENAS nos trechos abaixo, responda à pergunta de forma clara e objetiva.\n"
        "Se a informação não estiver nos trechos, diga exatamente: "
        "'Não encontrei essa informação nos documentos.'\n\n"
        f"Trechos:\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Resposta:"
    )


def _generate_ollama(prompt: str) -> str:
    log.info("Chamando Ollama (%s) em %s...", LLM_MODEL, OLLAMA_URL)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _generate_openai(prompt: str) -> str:
    from openai import OpenAI
    log.info("Chamando OpenAI (%s)...", OPENAI_MODEL)
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _generate_gemini(prompt: str) -> str:
    log.info("Chamando Gemini (%s)...", GEMINI_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate(question: str, chunks: list[dict]) -> str:
    """Roteia para o provider configurado em LLM_PROVIDER."""
    prompt = _build_prompt(question, chunks)
    if LLM_PROVIDER == "openai":
        return _generate_openai(prompt)
    if LLM_PROVIDER == "gemini":
        return _generate_gemini(prompt)
    return _generate_ollama(prompt)


def run(question: str) -> str:
    log.info("🔍 Recuperando chunks para: '%s'", question)
    docs = _load_embeddings_local()
    chunks = retrieve(question, docs)
    for i, c in enumerate(chunks, 1):
        log.info("  [%d] %s (score=%.4f)", i, c["file_name"], c["score"])
    log.info("🤖 Gerando resposta via %s...", LLM_PROVIDER)
    answer = generate(question, chunks)
    log.info("✅ Resposta gerada")
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    answer = run(args.question)
    print("\n" + "=" * 60)
    print("RESPOSTA:")
    print("=" * 60)
    print(answer)
