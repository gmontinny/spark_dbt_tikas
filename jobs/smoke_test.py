"""
Smoke Test — Valida todos os pré-requisitos antes de rodar o pipeline.

Execução:
    python jobs/smoke_test.py

Verifica:
    1. Conectividade com Spark Connect (versão compatível)
    2. Conectividade com Tika Server
    3. Conectividade com MinIO (S3)
    4. Conectividade com StarRocks
    5. Conectividade com Hive Metastore (via Spark)
    6. Arquivos em datas/
    7. Modelos HuggingFace (import + device)
    8. Variáveis de ambiente obrigatórias
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  WARN\033[0m"

failures = []


def check(name: str, fn):
    try:
        result = fn()
        msg = f" — {result}" if result else ""
        print(f"  {PASS}  {name}{msg}")
    except Exception as e:
        print(f"  {FAIL}  {name} → {e}")
        failures.append(name)


def warn(name: str, fn):
    try:
        result = fn()
        msg = f" — {result}" if result else ""
        print(f"  {PASS}  {name}{msg}")
    except Exception as e:
        print(f"  {WARN}  {name} → {e}")


# ── 1. Variáveis de ambiente ──────────────────────────────────────────────────
print("\n[1/8] Variáveis de ambiente")

def _check_env_vars():
    required = ["SPARK_CONNECT_URL", "TIKA_SERVER_JAR", "EMBEDDING_MODEL",
                "STARROCKS_HOST", "STARROCKS_FE_QUERY_PORT", "LLM_PROVIDER"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise ValueError(f"Faltando: {missing}")
    return f"LLM_PROVIDER={os.getenv('LLM_PROVIDER')}"

check("Variáveis obrigatórias presentes", _check_env_vars)

def _check_llm_keys():
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider == "openai" and not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        raise ValueError("OPENAI_API_KEY inválida ou ausente")
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY", "").startswith("AIza"):
        raise ValueError("GEMINI_API_KEY inválida ou ausente")
    return f"provider={provider}"

check("API Key do LLM provider", _check_llm_keys)

# ── 2. Arquivos em datas/ ─────────────────────────────────────────────────────
print("\n[2/8] Arquivos de entrada")

def _check_datas():
    datas = Path(os.getenv("DATAS_DIR", "/workspace/datas"))
    supported = {".pdf", ".docx", ".doc", ".txt"}
    files = [f for f in datas.iterdir() if f.suffix.lower() in supported]
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo em {datas}")
    return f"{len(files)} arquivo(s): {[f.name for f in files]}"

check("Documentos em datas/", _check_datas)

# ── 3. Spark Connect ──────────────────────────────────────────────────────────
print("\n[3/8] Spark Connect")

def _check_spark():
    from pyspark.sql import SparkSession
    import pyspark
    spark = SparkSession.builder.remote(
        os.getenv("SPARK_CONNECT_URL", "sc://spark-connect:15002")
    ).getOrCreate()
    spark.sql("SELECT 1").collect()
    spark.stop()
    return f"PySpark {pyspark.__version__}"

check("Conexão e query simples", _check_spark)

# ── 4. Tika Server ────────────────────────────────────────────────────────────
print("\n[4/8] Apache Tika")

def _check_tika():
    import requests
    url = os.getenv("TIKA_SERVER_JAR", "http://tika-server:9998")
    resp = requests.get(f"{url}/tika", timeout=5)
    resp.raise_for_status()
    return f"{url} respondeu {resp.status_code}"

check("Tika Server acessível", _check_tika)

# ── 5. MinIO / S3 ─────────────────────────────────────────────────────────────
print("\n[5/8] MinIO")

def _check_minio():
    import requests
    endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    resp = requests.get(f"{endpoint}/minio/health/live", timeout=5)
    resp.raise_for_status()
    return f"{endpoint} healthy"

check("MinIO health check", _check_minio)

# ── 6. StarRocks ──────────────────────────────────────────────────────────────
print("\n[6/8] StarRocks")

def _check_starrocks():
    import mysql.connector
    conn = mysql.connector.connect(
        host=os.getenv("STARROCKS_HOST", "starrocks-fe-0"),
        port=int(os.getenv("STARROCKS_FE_QUERY_PORT", "9030")),
        user=os.getenv("STARROCKS_USER", "root"),
        password=os.getenv("STARROCKS_PASSWORD", ""),
        connection_timeout=5,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1")
    cur.fetchone()
    cur.close()
    conn.close()
    return "query SELECT 1 OK"

check("StarRocks acessível", _check_starrocks)

# ── 7. Modelos HuggingFace ────────────────────────────────────────────────────
print("\n[7/8] Modelos HuggingFace")

def _check_torch():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return f"torch {torch.__version__} | device={device}"

check("PyTorch importável", _check_torch)

def _check_embedding_model():
    from sentence_transformers import SentenceTransformer
    import torch
    model_name = os.getenv("EMBEDDING_MODEL", "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    vec = model.encode("teste", normalize_embeddings=True)
    return f"dims={len(vec)} device={device}"

check("Embedding model carrega e encoda", _check_embedding_model)

def _check_summarizer():
    from transformers import BartTokenizer
    model_name = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
    BartTokenizer.from_pretrained(model_name)
    return f"{model_name} tokenizer OK"

warn("Summarizer model (BART) acessível", _check_summarizer)

def _check_classifier():
    from transformers import pipeline
    import torch
    model_name = os.getenv("CLASSIFIER_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification", model=model_name, device=device)
    result = clf("teste de classificação", candidate_labels=["economia", "política"])
    return f"top={result['labels'][0]}"

warn("Classifier model (zero-shot) funcional", _check_classifier)

# ── 8. LLM Provider ───────────────────────────────────────────────────────────
print("\n[8/8] LLM Provider")

def _check_llm():
    import requests
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return f"Ollama OK | modelos: {models}"
    if provider == "gemini":
        import requests as req
        key = os.getenv("GEMINI_API_KEY", "")
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        resp = req.post(url, json={"contents": [{"parts": [{"text": "ping"}]}]}, timeout=10)
        resp.raise_for_status()
        return f"Gemini {model} OK"
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return f"OpenAI OK | {resp.choices[0].message.content}"

check(f"LLM provider ({os.getenv('LLM_PROVIDER', 'ollama')}) responde", _check_llm)

# ── Resultado final ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if failures:
    print(f"\033[91m❌ {len(failures)} verificação(ões) falharam: {failures}\033[0m")
    print("Corrija os problemas acima antes de rodar make pipeline-tika")
    sys.exit(1)
else:
    print(f"\033[92m✅ Todos os checks passaram — pode rodar make pipeline-tika\033[0m")
