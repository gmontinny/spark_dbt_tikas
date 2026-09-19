# Pipeline de Documentos com IA Generativa: Apache Spark, Tika, dbt e RAG em Português

## Resumo

Este artigo descreve a construção de um pipeline completo de ingestão, enriquecimento e consulta inteligente de documentos não estruturados utilizando uma arquitetura de medalhas (Bronze → Silver → Gold). O sistema combina Apache Spark 4.x, Apache Tika, embeddings semânticos especializados em português, modelos de linguagem de grande escala (LLMs) e a técnica RAG (Retrieval-Augmented Generation) para permitir que usuários façam perguntas em linguagem natural sobre documentos PDF e recebam respostas fundamentadas no conteúdo real. Todo o ambiente é containerizado com Docker e orquestrado via Makefile, tornando a reprodução do projeto acessível em qualquer máquina com Docker instalado.

---

## 1. Introdução

A quantidade de documentos não estruturados produzidos diariamente — relatórios econômicos, boletins técnicos, contratos, laudos — cresce de forma exponencial. Extrair informações precisas desses documentos de forma manual é inviável em escala. Ao mesmo tempo, os modelos de linguagem modernos abriram uma nova possibilidade: perguntar ao documento em linguagem natural e receber uma resposta precisa, fundamentada no texto original.

O desafio técnico, porém, não é trivial. É necessário:

1. Extrair texto de formatos heterogêneos (PDF, DOCX, PPTX) de forma confiável
2. Normalizar e enriquecer esse texto com features semânticas
3. Gerar representações vetoriais (embeddings) que capturem o significado do conteúdo
4. Armazenar e consultar esses vetores de forma eficiente
5. Integrar um LLM que gere respostas coerentes com base nos trechos recuperados

Este projeto resolve cada um desses desafios com ferramentas de código aberto, priorizando o suporte ao português brasileiro — uma lacuna frequente em projetos de NLP que focam predominantemente no inglês.

---

## 2. Arquitetura Geral

O projeto adota a **arquitetura de medalhas** (Medallion Architecture), amplamente utilizada em plataformas de dados modernas como o Databricks Lakehouse. Cada camada tem responsabilidade bem definida:

```
Documentos (PDF/DOCX/TXT)
        │
        ▼
┌─────────────────────┐
│   BRONZE            │  Extração bruta via Apache Tika
│   MinIO / Parquet   │  texto + metadados → hive.bronze.documents
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   SILVER            │  Limpeza + TF-IDF + Embeddings semânticos
│   MinIO / Parquet   │  PORTULAN serafim-100m → hive.silver.documents_features
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   GOLD              │  Sumarização BART + Classificação zero-shot
│   StarRocks OLAP    │  gold.documents_enriched + marts dbt
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   RAG               │  Cosine similarity + LLM (OpenAI/Gemini/Ollama)
│   Streamlit         │  Respostas fundamentadas nos documentos
└─────────────────────┘
```

### 2.1 Stack de Tecnologias

| Categoria | Tecnologia | Versão |
|---|---|---|
| Processamento distribuído | Apache Spark Connect | 4.0.1 |
| Extração de documentos | Apache Tika (REST) | 2.x |
| Object store | MinIO (S3-compatible) | latest |
| Catálogo de metadados | Hive Metastore + PostgreSQL | — |
| Query engine ad-hoc | Trino | latest |
| OLAP / Data Warehouse | StarRocks | latest |
| Modelagem analítica | dbt (adapter starrocks) | 1.12.x |
| Streaming | Apache Kafka + Spark Structured Streaming | — |
| Embeddings em português | PORTULAN serafim-100m | HuggingFace |
| Inferência GPU/CPU | PyTorch + CUDA | 2.14.0 |
| LLM | OpenAI / Google Gemini / Ollama | — |
| Interface RAG | Streamlit | 1.59.x |
| Linguagem | Python | 3.10 |
| Gerenciador de pacotes | uv | latest |

---

## 3. Camada Bronze — Extração com Apache Tika

### 3.1 O papel do Apache Tika

O Apache Tika é um toolkit de análise de conteúdo capaz de extrair texto e metadados de mais de 1.000 formatos de arquivo diferentes. No projeto, ele é executado como servidor REST na porta 9998, e o cliente Python (`tika-python`) faz chamadas HTTP para os endpoints `/tika` (texto) e `/meta` (metadados).

Uma descoberta importante durante o desenvolvimento: o `tika-python` na versão 3.x exige o parâmetro `service="text"` para retornar o conteúdo corretamente. Sem esse parâmetro, o endpoint retorna apenas metadados. Além disso, uma única chamada com `service="all"` não retorna o conteúdo de texto — são necessárias duas chamadas separadas.

```python
# Extração de texto
text_result = tika_parser.from_file(
    str(file_path), serverEndpoint=TIKA_ENDPOINT, service="text"
)
# Extração de metadados
meta_result = tika_parser.from_file(
    str(file_path), serverEndpoint=TIKA_ENDPOINT, service="meta"
)
```

### 3.2 Metadados extraídos

Para cada documento, o pipeline extrai:

- `file_name` — nome do arquivo
- `content_type` — tipo MIME (ex: `application/pdf`)
- `text_content` — texto completo extraído
- `author` — autor do documento (campos `dc:creator`, `pdf:docinfo:creator` ou `Author`)
- `title` — título
- `created` — data de criação
- `language` — idioma detectado (ex: `pt`, `pt-BR`)
- `num_pages` — número de páginas

### 3.3 Persistência no MinIO

O resultado é gravado como Parquet no MinIO via Spark Connect, com registro no catálogo Hive:

```
s3a://warehouse/bronze/documents
hive.bronze.documents
```

O uso de Parquet garante compressão eficiente e leitura colunar nas camadas seguintes.

---

## 4. Camada Silver — Feature Engineering e Embeddings

### 4.1 Limpeza e normalização de texto

O texto extraído pelo Tika contém artefatos típicos de PDFs: espaços não-quebráveis (`\xa0`), sequências de escape HTML (`&#32;`), quebras de linha excessivas e caracteres especiais. A função de limpeza aplica uma série de substituições via expressões regulares:

```python
def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s*32;\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.,;:!?\-àáâãéêíóôõúüçÀÁÂÃÉÊÍÓÔÕÚÜÇ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
```

Essa UDF é executada nos workers do Spark via `udf()`, aproveitando o processamento distribuído para limpeza em escala.

### 4.2 TF-IDF via Spark MLlib

O pipeline calcula features TF-IDF usando a pipeline nativa do Spark MLlib:

```python
tokenizer  = Tokenizer(inputCol="text_clean", outputCol="words")
hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=512)
idf        = IDF(inputCol="raw_features", outputCol="tfidf_features")
```

O TF-IDF captura a importância estatística de cada termo no corpus, complementando os embeddings semânticos.

### 4.3 Embeddings semânticos em português

A escolha do modelo de embeddings foi uma das decisões mais importantes do projeto. A maioria dos modelos disponíveis no HuggingFace é treinada predominantemente em inglês, o que resulta em representações vetoriais de baixa qualidade para textos em português.

O modelo escolhido foi o **PORTULAN/serafim-100m-portuguese-pt-sentence-encoder**, desenvolvido pelo laboratório PORTULAN CLARIN da Universidade de Lisboa. Com 100 milhões de parâmetros e treinamento específico em português europeu e brasileiro, o modelo gera embeddings de 768 dimensões com alta precisão semântica para o idioma.

```python
embed_model = SentenceTransformer(
    "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
embeddings = embed_model.encode(texts, normalize_embeddings=True)
```

Os embeddings são normalizados (norma L2 = 1), o que permite usar o produto escalar diretamente como medida de similaridade — equivalente ao cosine similarity para vetores normalizados.

### 4.4 Decisão arquitetural: embeddings no driver

Uma limitação importante surgiu durante o desenvolvimento: o worker do Spark Connect roda em um container separado (`spark-connect`) que não possui PyTorch instalado. Instalar PyTorch no worker adicionaria ~3GB à imagem e aumentaria significativamente o tempo de build.

A solução adotada foi executar os embeddings no driver (container `dev`), após um `collect()` dos textos limpos:

```python
# TF-IDF nos workers (sem torch)
df_tfidf = idf_model.transform(df_tf)

# Collect no driver → embeddings com GPU
pdf = df_tfidf.drop("text_content", "tfidf_features").toPandas()
embeddings = embed_model.encode(texts, normalize_embeddings=True).tolist()
```

Para o volume atual (documentos de análise), essa abordagem é perfeitamente adequada. Em escala de produção com milhões de documentos, a solução seria instalar PyTorch nos workers ou usar um cluster Spark com workers GPU.

---

## 5. Camada Gold — Inferência com LLMs

### 5.1 Sumarização com BART

O modelo `facebook/bart-large-cnn` é utilizado para gerar resumos automáticos de cada documento. O BART (Bidirectional and Auto-Regressive Transformers) foi pré-treinado para tarefas de geração de texto e fine-tuned no dataset CNN/DailyMail para sumarização.

```python
summary_ids = model.generate(
    inputs["input_ids"],
    max_new_tokens=130,
    min_new_tokens=30,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True,
)
```

### 5.2 Classificação zero-shot

A classificação de tópicos utiliza o modelo `cross-encoder/nli-MiniLM2-L6-H768` com a técnica de zero-shot classification. Sem necessidade de fine-tuning, o modelo classifica cada documento em uma das categorias predefinidas:

```python
TOPICS = ["economia", "imóveis", "inflação", "mercado financeiro", "política", "tecnologia"]
classifier(text[:512], candidate_labels=TOPICS, truncation=True)
```

A técnica zero-shot funciona reformulando a classificação como um problema de inferência de linguagem natural (NLI): para cada label, o modelo avalia se o texto "implica" aquela categoria.

### 5.3 Persistência no StarRocks

Os resultados são gravados no StarRocks via `mysql-connector-python`, usando a tabela com `PRIMARY KEY` para garantir idempotência:

```sql
CREATE TABLE documents_enriched (
    file_name    VARCHAR(512) NOT NULL,
    topic        VARCHAR(128),
    summary      STRING,
    enriched_at  DATETIME
)
ENGINE = OLAP
PRIMARY KEY(file_name)
DISTRIBUTED BY HASH(file_name) BUCKETS 4
```

O modelo `PRIMARY KEY` do StarRocks garante upsert nativo: executar o pipeline múltiplas vezes atualiza os registros existentes sem duplicação.

### 5.4 Modelagem analítica com dbt

O dbt (data build tool) transforma os dados brutos do Gold em marts analíticos prontos para consumo:

| Modelo | Tipo | Descrição |
|---|---|---|
| `stg_documents_enriched` | view | Staging normalizado |
| `mart_documents_by_topic` | table | Métricas por tópico |
| `mart_document_summaries` | table | Sumarizações para RAG |

---

## 6. RAG — Retrieval-Augmented Generation

### 6.1 O problema que o RAG resolve

LLMs como GPT-4 e Gemini têm conhecimento limitado ao seu período de treinamento e não têm acesso ao conteúdo de documentos privados ou específicos. O RAG resolve isso em duas etapas:

1. **Recuperação**: encontrar os trechos mais relevantes para a pergunta
2. **Geração**: usar esses trechos como contexto para o LLM gerar a resposta

### 6.2 Fluxo de execução

```
Pergunta do usuário
      │
      ▼
Embedding da pergunta (PORTULAN serafim-100m, GPU)
      │
      ▼
Cosine Similarity contra todos os embeddings do Silver
      │
      ▼
TOP-K documentos mais relevantes (padrão: K=3)
      │
      ▼
Prompt estruturado em português + contexto dos documentos
      │
      ▼
LLM (OpenAI / Gemini / Ollama) gera a resposta
      │
      ▼
Resposta fundamentada nos documentos
```

### 6.3 Cosine Similarity

A similaridade entre a pergunta e cada documento é calculada via produto escalar (equivalente ao cosine similarity para vetores normalizados):

```python
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
```

### 6.4 Suporte a múltiplos providers de LLM

O sistema suporta três providers configuráveis via variável de ambiente `LLM_PROVIDER`:

- **Ollama** — execução local, gratuita, sem envio de dados para a nuvem
- **OpenAI** — GPT-4o-mini, alta qualidade, pago por uso
- **Google Gemini** — gemini-3.5-flash, gratuito com limites de requisição

A troca de provider não requer rebuild do container — basta alterar o `.env`.

### 6.5 Exemplo de resultado

Pergunta: *"Qual foi o crescimento do PIB Construção no primeiro semestre de 2025?"*

Chunks recuperados:
- `Boletim_Economico_ABRAINC_2tri25.pdf` (score=0.6738)
- `ipop-dezembro-2025.pdf` (score=0.6329)
- `fipezap-202608-residencial-venda.pdf` (score=0.5832)

Resposta gerada pelo Gemini:
> *"O PIB da construção cresceu 1,8% no primeiro semestre de 2025 em relação ao mesmo período de 2024."*

---

## 7. Infraestrutura e Containerização

### 7.1 Separação de responsabilidades entre containers

O projeto utiliza dois containers principais para o pipeline:

- **`spark-connect`** — servidor Spark 4.0.1, workers JVM, sem PyTorch
- **`dev`** — driver Python 3.10, PyTorch 2.14, todos os jobs do pipeline

Essa separação foi uma decisão deliberada: manter o `spark-connect` leve (imagem base `apache/spark:4.0.1`) e concentrar as dependências de ML no `dev`.

### 7.2 Compatibilidade de versões

Um dos desafios mais recorrentes foi garantir compatibilidade entre versões:

- PySpark no driver (`dev`) e no servidor (`spark-connect`) devem ser **exatamente iguais** — qualquer diferença de versão minor causa `PYTHON_VERSION_MISMATCH`
- O projeto fixa `pyspark==4.0.1` no `pyproject.toml` para evitar que o `uv` resolva para versões mais recentes incompatíveis

### 7.3 GPU no container

O `docker-compose.yml` configura o acesso à GPU NVIDIA via NVIDIA Container Toolkit:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

O pipeline detecta automaticamente a GPU em todos os estágios:
```
🖥️  Dispositivo de inferência: cuda:0   ← GPU ativa
🖥️  Dispositivo de inferência: cpu      ← fallback CPU
```

---

## 8. Dificuldades Encontradas

### 8.1 Incompatibilidade Python 3.10 vs 3.12

O container `dev` foi inicialmente configurado com `python:3.12-slim`. O servidor `spark-connect`, baseado em Ubuntu 22.04, possui Python 3.10 nativo. O Spark Connect exige que driver e worker usem a **mesma versão minor** do Python. A tentativa de instalar Python 3.12 via PPA no Ubuntu 22.04 falhou por dependências ausentes (`python3.12-distutils`). Solução: migrar o `dev` para `python:3.10-slim`.

### 8.2 UDFs do Spark e serialização de dependências

A primeira abordagem para gerar embeddings foi criar uma UDF Spark que carregasse o modelo internamente. O problema: o Spark serializa a UDF via `cloudpickle` e a envia para o worker. O worker tenta deserializar a função e importar suas dependências — que não existem no `spark-connect`. O erro `ModuleNotFoundError: No module named 'torch'` foi o sintoma.

Tentativas intermediárias com `lru_cache` no nível de módulo e variáveis `global` também falharam porque o worker roda em um processo Python separado, sem acesso ao namespace do driver.

A solução definitiva foi abandonar a UDF para embeddings e executar o `encode()` no driver após `collect()`.

### 8.3 SparseVector não serializável pelo PyArrow

Após mover os embeddings para o driver, surgiu um novo problema: a coluna `tfidf_features` do Spark MLlib é do tipo `SparseVector` — um tipo nativo do PySpark que o PyArrow não consegue serializar durante o `toPandas()`. A solução foi descartar a coluna antes do collect:

```python
pdf = df_tfidf.drop("text_content", "tfidf_features").toPandas()
```

### 8.4 mysql.connector não é context manager

O `mysql.connector.connect()` retorna um objeto que **não implementa o protocolo de context manager** (`__enter__`/`__exit__`). Usar `with _get_conn() as conn:` lançava `AttributeError`. A correção foi usar `try/finally` explícito com `conn.close()`.

### 8.5 Conflito de protobuf com SDK do Gemini

O SDK oficial `google-generativeai` usa protobuf de forma incompatível com o PySpark 4.x. A solução foi não usar o SDK e chamar a API do Gemini diretamente via HTTP com `requests`, eliminando completamente a dependência conflitante.

### 8.6 Qualidade do RAG e formulação de perguntas

Um comportamento inicialmente confuso: o RAG respondia "Não encontrei essa informação nos documentos" para perguntas cujas respostas estavam claramente nos PDFs. A investigação revelou que o problema era a formulação da pergunta, não o pipeline.

O modelo de embeddings PORTULAN é sensível ao vocabulário. A pergunta "Qual é o PIB Construção do ano 2025?" gerava um embedding semanticamente distante do trecho "PIB CONSTRUÇÃO: SETOR CRESCE NO 1º SEMESTRE". Reformulando para "Qual foi o crescimento do PIB Construção no primeiro semestre de 2025?", o score de similaridade subiu de ~0.45 para 0.67 e a resposta foi correta.

---

## 9. Capacidades Adicionais

### 9.1 Streaming com Kafka

O projeto inclui um job de inferência contínua (`streaming_inference.py`) que consome mensagens de um tópico Kafka (`tika.documents`) e aplica classificação zero-shot em tempo real via Spark Structured Streaming, gravando os resultados em `gold.documents_stream` no StarRocks com micro-batches de 30 segundos.

### 9.2 Fine-tuning distribuído

O job `finetune_distributed.py` demonstra o uso do `TorchDistributor` do Spark 4.x para fine-tuning distribuído de um modelo DistilBERT multilingual. O `TorchDistributor` usa barrier execution para sincronizar os workers durante o treinamento, e o modelo resultante é salvo no MinIO em `s3a://warehouse/models/`.

### 9.3 Dashboards com Apache Superset

O StarRocks expõe os marts analíticos gerados pelo dbt para visualização no Apache Superset, permitindo análises como distribuição de documentos por tópico, evolução temporal de ingestões e comparação de métricas entre documentos.

---

## 10. Conclusão

Este projeto demonstra que é possível construir um pipeline completo de inteligência documental com ferramentas de código aberto, sem depender de plataformas proprietárias. A arquitetura de medalhas provou ser uma escolha acertada: cada camada tem responsabilidade clara, é testável independentemente e pode ser reprocessada sem afetar as demais.

A escolha do modelo PORTULAN para embeddings foi determinante para a qualidade do RAG em português. Modelos multilíngues genéricos produzem representações vetoriais de qualidade inferior para o idioma, resultando em recuperação semântica imprecisa.

As principais lições aprendidas:

1. **Compatibilidade de versões em ambientes distribuídos é crítica** — driver e worker devem ter exatamente a mesma versão do Python e do PySpark
2. **UDFs Spark têm limitações sérias para dependências pesadas** — PyTorch não pode ser serializado e enviado para workers via cloudpickle
3. **A qualidade do RAG depende tanto do modelo quanto da formulação da pergunta** — embeddings semânticos são sensíveis ao vocabulário usado
4. **Idempotência deve ser projetada desde o início** — o uso de `PRIMARY KEY` no StarRocks e `mode("overwrite")` no Parquet garante que o pipeline possa ser reexecutado sem efeitos colaterais

Como próximos passos naturais, o projeto pode evoluir para:

- Chunking de documentos longos (dividir cada PDF em múltiplos chunks para recuperação mais precisa)
- Indexação vetorial com FAISS ou pgvector para escala com milhões de documentos
- Reranking dos chunks recuperados com um modelo cross-encoder antes de enviar ao LLM
- Avaliação automática da qualidade do RAG com métricas como RAGAS

---

## Referências

1. **Apache Spark 4.0** — Spark Connect: nova arquitetura cliente-servidor para PySpark. https://spark.apache.org/docs/latest/spark-connect-overview.html

2. **Apache Tika** — A content analysis toolkit. https://tika.apache.org/

3. **PORTULAN/serafim-100m-portuguese-pt-sentence-encoder** — Modelo de embeddings especializado em português. https://huggingface.co/PORTULAN/serafim-100m-portuguese-pt-sentence-encoder

4. **Lewis, P. et al. (2020)** — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *arXiv:2005.11401*. https://arxiv.org/abs/2005.11401

5. **Lewis, M. et al. (2019)** — BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *arXiv:1910.13461*. https://arxiv.org/abs/1910.13461

6. **Yin, W. et al. (2019)** — Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. *arXiv:1909.00161*. https://arxiv.org/abs/1909.00161

7. **StarRocks** — A high-performance analytical database. https://www.starrocks.io/

8. **dbt (data build tool)** — Analytics engineering framework. https://docs.getdbt.com/

9. **MinIO** — High-performance S3-compatible object storage. https://min.io/

10. **Databricks** — Medallion Architecture. https://www.databricks.com/glossary/medallion-architecture

11. **Reimers, N. & Gurevych, I. (2019)** — Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *arXiv:1908.10084*. https://arxiv.org/abs/1908.10084

12. **uv** — An extremely fast Python package manager. https://docs.astral.sh/uv/
