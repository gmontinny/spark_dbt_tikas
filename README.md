# Spark + Tika + dbt + StarRocks + IA Generativa

> Pipeline completo de ingestão, enriquecimento e consulta inteligente de documentos com arquitetura de medalhas (Bronze → Silver → Gold), IA Generativa (RAG) e suporte a múltiplos provedores de LLM.

---

## Visão Geral

Este projeto demonstra uma arquitetura moderna de dados aplicada a documentos não estruturados (PDF, DOCX, TXT). O pipeline extrai texto e metadados com **Apache Tika**, processa e gera embeddings semânticos com **Apache Spark 4.x**, aplica modelos de linguagem para sumarização e classificação, e disponibiliza uma interface de perguntas e respostas via **RAG (Retrieval-Augmented Generation)**.

O usuário pode fazer perguntas em linguagem natural sobre os documentos e receber respostas fundamentadas no conteúdo real — usando **OpenAI**, **Google Gemini** ou **Ollama** (local, sem custo) como LLM.

Os embeddings são gerados com o modelo **PORTULAN/serafim-100m-portuguese-pt-sentence-encoder**, especializado em português, garantindo alta precisão na recuperação semântica de documentos em língua portuguesa.

---

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
│  ├── Extração de texto completo  (endpoint /tika)                       │
│  └── Extração de metadados       (endpoint /meta)                       │
│      autor, título, páginas, idioma, data de criação                    │
│                                                                         │
│  Spark 4.x Connect → Parquet → MinIO (s3a://warehouse/bronze)          │
│  Catálogo: hive.bronze.documents  (consultável via Trino)               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAMADA SILVER                                   │
│                                                                         │
│  Spark 4.x — Feature Engineering                                       │
│  ├── Limpeza e normalização de texto (UDF distribuída)                  │
│  │   └── Remove artefatos de PDF, espaços, caracteres especiais        │
│  ├── TF-IDF via Spark MLlib (HashingTF + IDF)                          │
│  └── Embeddings semânticos em português (GPU prioritário)               │
│      └── PORTULAN/serafim-100m-portuguese-pt-sentence-encoder          │
│          (modelo especializado em português — CUDA → CPU fallback)      │
│                                                                         │
│  Spark 4.x Connect → Parquet → MinIO (s3a://warehouse/silver)          │
│  Catálogo: hive.silver.documents_features  (consultável via Trino)     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CAMADA GOLD                                    │
│                                                                         │
│  Spark 4.x — Inferência com LLMs (GPU prioritário)                     │
│  ├── Sumarização (facebook/bart-large-cnn — CUDA → CPU fallback)       │
│  └── Classificação zero-shot (cross-encoder/nli-MiniLM2-L6-H768)       │
│                                                                         │
│  Escrita via mysql-connector-python → StarRocks OLAP (:9030)           │
│  └── gold.documents_enriched  (PRIMARY KEY — upsert nativo)            │
│                                                                         │
│  dbt (adapter: starrocks) — Modelagem Analítica                        │
│  ├── stg_documents_enriched      (staging / view)                      │
│  ├── mart_documents_by_topic     (métricas por tópico)                 │
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
│     (PORTULAN serafim)    │  │  │   tempo real (micro-batch 30s)      │
│  3. Cosine similarity     │  │  └── Grava em StarRocks                │
│     nos embeddings Silver │  │      gold.documents_stream             │
│  4. TOP-K documentos      │  └────────────────────────────────────────┘
│     mais relevantes       │
│  5. Prompt + contexto     │  ┌────────────────────────────────────────┐
│     → LLM (OpenAI /       │  │         FINE-TUNING                    │
│       Gemini / Ollama)    │  │                                        │
│  6. Resposta gerada       │  │  TorchDistributor (Spark 4.x)         │
│     fundamentada nos docs │  │  ├── Barrier execution distribuído     │
│                           │  │  ├── DistilBERT multilingual           │
│  Interface: Streamlit     │  │  └── Modelo salvo em MinIO             │
│  http://localhost:8501    │  │      s3a://warehouse/models/           │
└───────────────────────────┘  └────────────────────────────────────────┘
```

---

## Stack de Tecnologias

| Categoria | Tecnologia |
|---|---|
| Processamento distribuído | Apache Spark 4.0.1 (Connect mode) |
| Extração de documentos | Apache Tika 2.x (REST) |
| Object store | MinIO (S3-compatible) |
| Catálogo de metadados | Hive Metastore + PostgreSQL |
| Query engine ad-hoc | Trino |
| OLAP / Data Warehouse | StarRocks |
| Modelagem analítica | dbt (adapter starrocks) |
| Streaming | Apache Kafka + Spark Structured Streaming |
| Embeddings (português) | PORTULAN serafim-100m (HuggingFace) |
| Inferência GPU/CPU | PyTorch + CUDA (GPU prioritário, CPU fallback) |
| LLM — geração de texto | OpenAI GPT-4o-mini / Google Gemini / Ollama |
| Interface RAG | Streamlit |
| Dashboards | Apache Superset |
| Linguagem | Python 3.12 |
| Gerenciador de pacotes | uv |

---

## Pré-requisitos

### Obrigatórios
- [Docker](https://docs.docker.com/get-docker/) 24+
- [Docker Compose](https://docs.docker.com/compose/install/) v2+
- 16 GB de RAM disponível (recomendado: 32 GB)
- 20 GB de espaço em disco

### GPU (opcional, mas recomendado)
O projeto detecta automaticamente a GPU via CUDA em todos os estágios de inferência. Se disponível, embeddings (Silver), sumarização e classificação (Gold), e o embedding do RAG são executados na GPU — significativamente mais rápido. O `docker-compose.yml` já está configurado com `deploy.resources` para passar a GPU NVIDIA ao container.

A ordem de prioridade é sempre: **GPU (CUDA) → CPU**. Na camada Gold, o log confirma o dispositivo em uso:
```
🖥️  Dispositivo de inferência: cuda:0
```

Requisito: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) instalado no host.

### Para o LLM (escolha uma opção)

| Opção | Custo | Requisito |
|---|---|---|
| **Ollama** (padrão) | Gratuito | [Instalar Ollama](https://ollama.com) + `ollama pull llama3.1` |
| **OpenAI** | Pago por uso | [API Key](https://platform.openai.com/api-keys) |
| **Google Gemini** | Gratuito com limites | [API Key](https://aistudio.google.com/app/apikey) |

---

## Início Rápido

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/spark-tika-dbt-starrocks.git
cd spark-tika-dbt-starrocks
```

### 2. Configure o ambiente

```bash
cp .env.example .env
```

Abra o `.env` e configure o provider de LLM desejado:

```env
# Escolha: openai | gemini | ollama
LLM_PROVIDER=ollama

# Temperatura: 0.0 = preciso/determinístico | 1.0 = criativo
# Para RAG recomenda-se entre 0.0 e 0.2
LLM_TEMPERATURE=0.1
```

### 3. Suba os serviços

```bash
make up
```

> Na primeira execução o Docker irá baixar e construir as imagens (~5–10 min dependendo da conexão).

### 4. Execute o pipeline completo

```bash
# Bronze → Silver → Gold (extração, embeddings GPU, inferência LLM GPU)
make pipeline-tika

# Transformações dbt (marts analíticos no StarRocks)
make dbt-tika
```

### 5. Abra a interface RAG

```bash
make app
```

Acesse **http://localhost:8501** e faça perguntas sobre os documentos.

---

## Adicionando seus próprios documentos

Coloque arquivos PDF, DOCX ou TXT na pasta `datas/` e reexecute o pipeline:

```bash
make pipeline-tika
make dbt-tika
```

O pipeline detecta automaticamente os novos arquivos.

---

## Comandos disponíveis

```bash
make help              # Lista todos os comandos disponíveis
make up                # Sobe todos os serviços
make down              # Para todos os serviços
make rebuild           # Rebuild completo das imagens e reinicia
make logs              # Logs de todos os containers (follow)
make pipeline-tika     # Pipeline completo: Bronze → Silver → Gold
make bronze            # Apenas extração Tika → MinIO
make silver            # Apenas features + embeddings (GPU)
make gold              # Apenas sumarização + classificação (GPU)
make dbt-tika          # Transformações dbt + testes de qualidade
make app               # Interface RAG (Streamlit)
make streaming         # Inferência contínua via Kafka
make finetune          # Fine-tuning distribuído (TorchDistributor)
make rag Q="pergunta"  # Pergunta via CLI (sem interface web)
```

---

## Serviços e Portas

| Serviço | URL | Credenciais |
|---|---|---|
| Streamlit (RAG) | http://localhost:8501 | — |
| Trino UI | http://localhost:8080 | — |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Superset | http://localhost:8088 | `admin` / `admin` |
| Spark UI | http://localhost:4040 | — |
| StarRocks FE | `localhost:9030` | `root` / *(vazio)* |
| Apache Tika | http://localhost:9998 | — |

---

## Estrutura do Projeto

```
.
├── datas/                          # Documentos de entrada (PDF, DOCX, TXT)
├── jobs/
│   ├── bronze_tika_ingest.py       # Extração Tika → MinIO (Bronze)
│   ├── silver_features_embeddings.py # TF-IDF + embeddings PORTULAN (Silver)
│   ├── gold_llm_inference.py       # Sumarização + classificação (Gold)
│   ├── rag_query.py                # RAG: busca semântica + geração LLM
│   ├── streaming_inference.py      # Inferência contínua Kafka → StarRocks
│   ├── finetune_distributed.py     # Fine-tuning com TorchDistributor
│   ├── pipeline_run.py             # Orquestrador Bronze → Silver → Gold
│   └── debug/                      # Scripts de diagnóstico (não produção)
├── app/
│   └── streamlit_rag.py            # Interface web RAG
├── dbt_project/
│   ├── models/
│   │   ├── staging/                # stg_documents_enriched (view)
│   │   └── gold/                   # mart_documents_by_topic, mart_document_summaries
│   ├── dbt_project.yml
│   └── profiles.yml
├── .devcontainer/
│   └── Dockerfile                  # Imagem do container dev (Python 3.12 + uv)
├── spark-connect/                  # Configuração do Spark Connect
├── hive-metastore/                 # Configuração do Hive Metastore
├── trino/                          # Catálogo Trino (hive)
├── docker-compose.yml
├── Makefile
├── pyproject.toml                  # Dependências Python (gerenciado por uv)
├── .env.example                    # Template de configuração
└── README.md
```

---

## Consultando os dados

### Trino — camadas Bronze e Silver

| Campo | Valor |
|---|---|
| Driver | Trino |
| Host | `localhost` |
| Port | `8080` |
| User | `admin` |
| Catalog | `hive` |

```sql
-- Documentos extraídos pelo Tika
SELECT file_name, content_type, author, language, num_pages, ingested_at
FROM hive.bronze.documents;

-- Features e embeddings gerados pelo Spark
SELECT file_name, text_length, processed_at
FROM hive.silver.documents_features;
```

### StarRocks — camada Gold

| Campo | Valor |
|---|---|
| Driver | MySQL |
| Host | `localhost` |
| Port | `9030` |
| User | `root` |
| Database | `gold` |

```sql
-- Documentos enriquecidos pelo LLM
SELECT file_name, topic, summary, enriched_at
FROM gold.documents_enriched
ORDER BY enriched_at DESC;

-- Mart por tópico (gerado pelo dbt)
SELECT topic, topic_total_docs, topic_avg_text_length
FROM gold.mart_documents_by_topic
ORDER BY topic_total_docs DESC;
```

---

## Configuração do LLM

O provider é configurado pela variável `LLM_PROVIDER` no `.env`. **Não é necessário recriar o container** — a alteração é lida a cada execução.

### Trocar de provider

Edite apenas uma linha no `.env`:

```env
LLM_PROVIDER=openai    # ou: gemini | ollama
```

### Temperatura

Controla o comportamento do modelo na geração de respostas:

```env
LLM_TEMPERATURE=0.1   # 0.0 = determinístico/preciso | 1.0 = criativo
```

Para RAG recomenda-se entre `0.0` e `0.2` — o modelo deve extrair informações dos documentos, não inventar.

### Ollama (gratuito, local)

```bash
# 1. Instale o Ollama: https://ollama.com
# 2. Baixe o modelo
ollama pull llama3.1

# 3. Configure o .env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OLLAMA_URL=http://host.docker.internal:11434
```

Outros modelos disponíveis: `gemma3`, `phi3`, `mistral`, `deepseek-coder-v2`.

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Google Gemini

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-3.5-flash
```

Outros modelos Gemini disponíveis: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.1-flash-lite`.

---

## Como funciona o RAG

```
Pergunta do usuário
      │
      ▼
Embedding da pergunta
(PORTULAN/serafim-100m — modelo especializado em português)
      │
      ▼
Cosine Similarity contra embeddings da camada Silver
      │
      ▼
TOP-K documentos mais relevantes (padrão: 3)
      │
      ▼
Prompt estruturado em português + contexto dos documentos
      │
      ▼
LLM escolhido (OpenAI / Gemini / Ollama) gera a resposta
      │
      ▼
Resposta fundamentada nos documentos
```

O modelo responde com base no conteúdo dos documentos. Se a informação não estiver disponível, informa explicitamente.

---

## Modelos dbt

| Modelo | Tipo | Descrição |
|---|---|---|
| `stg_documents_enriched` | view | Staging normalizado dos documentos enriquecidos |
| `mart_documents_by_topic` | table | Métricas analíticas por tópico (para Superset) |
| `mart_document_summaries` | table | Sumarizações prontas para RAG e busca semântica |

```bash
cd dbt_project
dbt run --profiles-dir .
dbt test --profiles-dir .
```

---

## Requisitos de Hardware

| Configuração | RAM | Disco | Observação |
|---|---|---|---|
| Mínima | 16 GB | 20 GB | Sem Ollama local, sem GPU — inferência na CPU |
| Recomendada | 32 GB | 40 GB | Com Ollama local — inferência na CPU |
| Ideal | 32 GB + GPU NVIDIA | 60 GB | GPU acelera embeddings, sumarização e classificação (RTX 3060+) |

---

## Solução de Problemas

**Erro `SparkNoSuchElementException: SQL_CONF_NOT_FOUND` no Bronze/Silver/Gold**
Incompatibilidade de versão entre o cliente PySpark (container `dev`) e o servidor Spark Connect. O projeto fixa `pyspark==4.0.1` no `pyproject.toml` para garantir compatibilidade. Se ocorrer após uma limpeza do Docker, rebuilde o container:
```bash
docker compose up -d --build dev
```

**Containers não sobem / ficam em `unhealthy`**
```bash
make logs
make down && make up
```

**Erro de memória no Spark**
Edite `spark-connect/conf/spark-defaults.conf` e reduza `spark.executor.memory`.

**GPU não detectada no container**
Verifique se o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) está instalado e reinicie o Docker Desktop. Para confirmar que a GPU está sendo usada, observe o log da camada Gold:
```
🖥️  Dispositivo de inferência: cuda:0   ← GPU ativa
🖥️  Dispositivo de inferência: cpu      ← GPU não detectada
```

**Ollama não responde**
Verifique se o Ollama está rodando no host: `ollama list`. O container acessa via `host.docker.internal:11434`.

**Modelo HuggingFace não baixa**
Os modelos são baixados automaticamente na primeira execução. Verifique a conexão com a internet e o espaço em disco.

**Erro `shapes (384,) and (768,) not aligned` no RAG**
O Silver foi gerado com um modelo de embeddings diferente do atual. Reprocesse:
```bash
make silver
```
Isso acontece ao trocar `EMBEDDING_MODEL` no `.env`. O modelo padrão é `PORTULAN/serafim-100m-portuguese-pt-sentence-encoder` (768 dims).

**Gemini retorna erro de conflito de protobuf**
O SDK `google-generativeai` conflita com o PySpark 4.x. O projeto já usa Gemini via HTTP direto — certifique-se de que `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` está no `.env`.

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
