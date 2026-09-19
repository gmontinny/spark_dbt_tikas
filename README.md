# Spark + Tika + dbt + StarRocks + IA Generativa

Pipeline de ingestão, enriquecimento e consulta inteligente de documentos
com arquitetura de medalhas (Bronze → Silver → Gold) e IA Generativa (RAG + LLMs).

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FONTES DE DADOS                                 │
│                    datas/ (PDF, DOCX, TXT, ...)                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAMADA BRONZE                                   │
│                                                                         │
│  Apache Tika (REST :9998)                                               │
│  ├── Extração de texto completo                                         │
│  ├── Extração de metadados (autor, título, páginas, idioma)             │
│  └── Suporte a PDF, DOCX, PPTX, XLSX, HTML, TXT                        │
│                                                                         │
│  Spark 4.x Connect → Parquet → MinIO (s3a://warehouse/bronze)          │
│  Catálogo: hive.bronze.documents  (consultável via Trino)               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAMADA SILVER                                   │
│                                                                         │
│  Spark 4.x — Feature Engineering e Deep Learning                       │
│  ├── Limpeza e normalização de texto (UDF distribuída)                  │
│  ├── TF-IDF via Spark MLlib (HashingTF + IDF)                          │
│  └── Embeddings semânticos distribuídos                                 │
│      └── SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)  │
│          executado como UDF em cada executor Spark                      │
│                                                                         │
│  Spark 4.x Connect → Parquet → MinIO (s3a://warehouse/silver)          │
│  Catálogo: hive.silver.documents_features  (consultável via Trino)     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CAMADA GOLD                                    │
│                                                                         │
│  Spark 4.x — Inferência Distribuída com LLMs                           │
│  ├── Sumarização (facebook/bart-large-cnn)                              │
│  │   └── Gera resumo de cada documento via UDF Spark                   │
│  └── Classificação zero-shot (cross-encoder/nli-MiniLM2-L6-H768)       │
│      └── Classifica tópico sem treinamento supervisionado               │
│                                                                         │
│  Escrita via JDBC → StarRocks OLAP (:9030)                             │
│  └── gold.documents_enriched                                            │
│                                                                         │
│  dbt (adapter: starrocks) — Modelagem Analítica                        │
│  ├── stg_documents_enriched      (staging / view)                      │
│  ├── mart_documents_by_topic     (tabela OLAP por tópico)              │
│  └── mart_document_summaries     (tabela para RAG/busca semântica)     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
┌───────────────────────────┐  ┌────────────────────────────────────────┐
│      IA GENERATIVA        │  │           STREAMING                    │
│         (RAG)             │  │                                        │
│                           │  │  Kafka → Structured Streaming          │
│  1. Pergunta do usuário   │  │  ├── Consome tópico tika.documents     │
│  2. Embed da pergunta     │  │  ├── Classificação zero-shot em        │
│     (SentenceTransformers)│  │  │   tempo real (micro-batch 30s)      │
│  3. Cosine similarity     │  │  └── Grava em StarRocks                │
│     nos embeddings Silver │  │      gold.documents_stream             │
│  4. TOP-K chunks          │  └────────────────────────────────────────┘
│     mais relevantes       │
│  5. Prompt + contexto     │  ┌────────────────────────────────────────┐
│     → Mistral-7B          │  │         FINE-TUNING                    │
│  6. Resposta gerada       │  │                                        │
│     fundamentada nos docs │  │  TorchDistributor (Spark 4.x)         │
│                           │  │  ├── Barrier execution distribuído     │
│  Interface: Streamlit     │  │  ├── DistilBERT multilingual           │
│  http://localhost:8501    │  │  └── Modelo salvo em MinIO             │
└───────────────────────────┘  │      s3a://warehouse/models/           │
                               └────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          VISUALIZAÇÃO                                   │
│                                                                         │
│  Superset (:8088)   — Dashboards sobre marts Gold (StarRocks)          │
│  Trino UI (:8080)   — Exploração ad-hoc Bronze e Silver                │
│  Streamlit (:8501)  — Interface RAG (perguntas sobre documentos)       │
│  MinIO UI (:9001)   — Exploração dos arquivos Parquet                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Serviços

| Serviço          | Porta   | Descrição                                          |
|------------------|---------|----------------------------------------------------|
| tika-server      | 9998    | Apache Tika REST — extração de texto e metadados   |
| spark-connect    | 15002   | Spark 4.x Connect (gRPC) — processamento distribuído |
| starrocks-fe-0   | 9030    | StarRocks FE — query engine OLAP (camada Gold)     |
| starrocks-be-0   | 8040    | StarRocks BE — storage e execução                  |
| trino            | 8080    | Query engine ad-hoc (camadas Bronze e Silver)      |
| hive-metastore   | 9083    | Catálogo de tabelas Bronze e Silver                |
| minio            | 9000/01 | Object store — Bronze, Silver e modelos            |
| kafka            | 9092    | Broker para inferência em streaming                |
| superset         | 8088    | Dashboards analíticos                              |
| postgres         | 5432    | Backend do Hive Metastore                          |

## Execução

```bash
# 1. Subir todos os serviços
make up

# 2. Pipeline completo (Bronze → Silver → Gold)
make pipeline-tika

# 3. Transformações dbt (Gold → marts analíticos)
make dbt-tika

# 4. Interface RAG (perguntas sobre os documentos)
make app          # abre em http://localhost:8501

# Ou via CLI:
docker compose exec dev python /workspace/jobs/rag_query.py --question "Qual foi a variação do IPCA?"

# Inferência contínua (Kafka → StarRocks)
make streaming

# Fine-tuning distribuído (TorchDistributor)
make finetune

# Etapas individuais do pipeline:
make bronze   # Extração Tika
make silver   # Features + Embeddings
make gold     # Inferência LLM → StarRocks
```

## Jobs

| Arquivo                              | Camada    | Descrição                                              |
|--------------------------------------|-----------|--------------------------------------------------------|
| `jobs/bronze_tika_ingest.py`         | Bronze    | Extrai texto e metadados via Apache Tika REST          |
| `jobs/silver_features_embeddings.py` | Silver    | TF-IDF (MLlib) + embeddings semânticos distribuídos    |
| `jobs/gold_llm_inference.py`         | Gold      | Sumarização + classificação zero-shot → StarRocks      |
| `jobs/rag_query.py`                  | IA Gen.   | RAG: cosine similarity + Mistral-7B gera resposta      |
| `jobs/streaming_inference.py`        | Streaming | Inferência contínua Kafka → StarRocks (Structured Streaming) |
| `jobs/finetune_distributed.py`       | IA Gen.   | Fine-tuning distribuído com TorchDistributor (Spark 4.x) |
| `jobs/pipeline_run.py`               | All       | Orquestrador Bronze → Silver → Gold                    |

## IA Generativa — Detalhes

### RAG (Retrieval-Augmented Generation)
O `rag_query.py` implementa o ciclo completo:

```
Pergunta → Embedding (SentenceTransformers)
         → Cosine Similarity nos embeddings do Silver
         → TOP-K chunks mais relevantes
         → Prompt com contexto
         → Mistral-7B gera resposta fundamentada nos documentos
```

Configurável via `.env`:
```env
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
RAG_TOP_K=3
```

### Streaming (Inferência Contínua)
O `streaming_inference.py` usa Structured Streaming do Spark 4.x:
- Consome mensagens JSON do tópico Kafka `tika.documents`
- Aplica classificação zero-shot a cada micro-batch (30s)
- Grava resultados em `gold.documents_stream` no StarRocks

### Fine-Tuning Distribuído
O `finetune_distributed.py` usa o **TorchDistributor** do Spark 4.x:
- Barrier execution garante sincronização entre workers
- Ajusta DistilBERT multilingual nos documentos da camada Silver
- Modelo salvo em `s3a://warehouse/models/topic_classifier`

## dbt (camada Gold — StarRocks)

```bash
cd dbt_project
dbt run --profiles-dir .    # cria marts analíticos no StarRocks
dbt test --profiles-dir .   # valida qualidade dos dados
```

### Modelos dbt

| Modelo                      | Tipo   | Descrição                                        |
|-----------------------------|--------|--------------------------------------------------|
| `stg_documents_enriched`    | view   | Staging normalizado dos documentos enriquecidos  |
| `mart_documents_by_topic`   | table  | Métricas analíticas por tópico (para Superset)   |
| `mart_document_summaries`   | table  | Sumarizações prontas para RAG e busca semântica  |

## Variáveis de Ambiente (.env)

```env
# Tika
TIKA_SERVER_JAR=http://tika-server:9998
DATAS_DIR=/opt/spark/work-dir/datas

# StarRocks
STARROCKS_HOST=starrocks-fe-0
STARROCKS_FE_QUERY_PORT=9030
STARROCKS_USER=root
STARROCKS_PASSWORD=
STARROCKS_DB=gold

# Modelos de IA (HuggingFace)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
SUMMARIZER_MODEL=facebook/bart-large-cnn
CLASSIFIER_MODEL=cross-encoder/nli-MiniLM2-L6-H768
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
RAG_TOP_K=3
```

## Acesso via DataGrip (ou qualquer SQL client)

### Trino — camadas Bronze e Silver
| Campo    | Valor                                      |
|----------|--------------------------------------------|
| Driver   | Trino                                      |
| Host     | `localhost`                                |
| Port     | `8080`                                     |
| User     | `admin` (qualquer string, sem autenticação)|
| Password | *(vazio)*                                  |
| Catalog  | `hive`                                     |

```sql
-- Schemas disponíveis
SHOW SCHEMAS FROM hive;

-- Bronze: documentos extraídos pelo Tika
SELECT file_name, content_type, language, num_pages, ingested_at
FROM hive.bronze.documents;

-- Silver: features e embeddings gerados pelo Spark
SELECT file_name, text_length, processed_at
FROM hive.silver.documents_features;
```

### StarRocks — camada Gold
| Campo    | Valor                              |
|----------|------------------------------------|
| Driver   | MySQL (StarRocks é MySQL-compatible)|
| Host     | `localhost`                        |
| Port     | `9030`                             |
| User     | `root`                             |
| Password | *(vazio)*                          |
| Database | `gold`                             |

```sql
-- Documentos enriquecidos pelo LLM
SELECT file_name, topic, summary, enriched_at
FROM gold.documents_enriched
ORDER BY enriched_at DESC;

-- Mart por tópico (gerado pelo dbt)
SELECT topic, topic_total_docs, topic_avg_text_length
FROM gold.mart_documents_by_topic
ORDER BY topic_total_docs DESC;

-- Sumarizações para RAG
SELECT document_title, topic, summary
FROM gold.mart_document_summaries;

-- Inferência em streaming (tempo real)
SELECT file_name, topic, classified_at
FROM gold.documents_stream
ORDER BY classified_at DESC
LIMIT 20;
```

### Outros serviços
| Serviço    | URL                   | Credenciais              |
|------------|-----------------------|--------------------------|
| MinIO      | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Superset   | http://localhost:8088 | `admin` / `admin`        |
| Streamlit  | http://localhost:8501 | sem autenticação         |
| Spark UI   | http://localhost:4040 | sem autenticação         |
| Trino UI   | http://localhost:8080 | sem autenticação         |
