# Spark + Tika + dbt + Trino + IA Generativa

Pipeline de ingestão e enriquecimento de documentos com arquitetura de medalhas (Bronze → Silver → Gold).

## Arquitetura

```
datas/ (PDF, DOCX)
    │
    ▼
[Bronze] Apache Tika → extração de texto e metadados → MinIO (s3a://warehouse/bronze)
    │
    ▼
[Silver] Spark 4.x → limpeza + TF-IDF + Embeddings (SentenceTransformers) → MinIO (s3a://warehouse/silver)
    │
    ▼
[Gold]   Spark 4.x → LLM Inference (sumarização + classificação zero-shot) → StarRocks (OLAP)
    │
    ▼
[dbt]    StarRocks → modelos analíticos (mart_documents_by_topic, mart_document_summaries)
    │
    ▼
[Superset] Dashboards e exploração
```

## Serviços

| Serviço          | Porta  | Descrição                              |
|------------------|--------|----------------------------------------|
| starrocks-fe-0   | 9030   | StarRocks FE — query engine OLAP para camada Gold |
| starrocks-be-0   | 8040   | StarRocks BE — storage e execução                 |
| tika-server      | 9998   | Apache Tika REST para extração de docs |
| spark-connect    | 15002  | Spark 4.x Connect (gRPC)               |
| trino            | 8080   | Query engine para camadas Bronze/Silver |
| hive-metastore   | 9083   | Catálogo de tabelas (Bronze/Silver)    |
| minio            | 9000/1 | Object store (Bronze/Silver)           |
| superset         | 8088   | Dashboards                             |
| postgres         | 5432   | Backend do Hive Metastore              |

## Execução

```bash
# 1. Subir todos os serviços
make up

# 2. Pipeline completo (Bronze → Silver → Gold)
make pipeline-tika

# 3. Transformações dbt (Gold → marts analíticos)
make dbt-tika

# Ou etapas individuais:
make bronze   # Extração Tika
make silver   # Features + Embeddings
make gold     # Inferência LLM
```

## Jobs

| Arquivo                          | Camada  | Descrição                                      |
|----------------------------------|---------|------------------------------------------------|
| `jobs/bronze_tika_ingest.py`     | Bronze  | Extrai texto/metadados via Tika REST           |
| `jobs/silver_features_embeddings.py` | Silver | TF-IDF + embeddings semânticos (distribuído) |
| `jobs/gold_llm_inference.py`     | Gold    | Sumarização + classificação zero-shot (LLM)    |
| `jobs/pipeline_run.py`           | All     | Orquestrador Bronze → Silver → Gold            |

## dbt (camada Gold)

```bash
cd dbt_project
dbt run --profiles-dir .    # cria marts analíticos no Trino
dbt test --profiles-dir .   # valida qualidade dos dados
```

### Modelos dbt

- `stg_documents_enriched` — staging dos documentos enriquecidos
- `mart_documents_by_topic` — documentos com métricas por tópico
- `mart_document_summaries` — sumarizações para RAG/busca semântica

## Variáveis de Ambiente (.env)

```env
TIKA_SERVER_JAR=http://tika-server:9998
DATAS_DIR=/opt/spark/work-dir/datas
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
SUMMARIZER_MODEL=facebook/bart-large-cnn
CLASSIFIER_MODEL=cross-encoder/nli-MiniLM2-L6-H768
```

## Consultas Trino (exemplos)

```sql
-- Documentos por tópico
SELECT topic, total_documents, avg_text_length
FROM hive.gold.mart_documents_by_topic
ORDER BY total_documents DESC;

-- Sumarizações
SELECT document_title, topic, summary
FROM hive.gold.mart_document_summaries;
```
