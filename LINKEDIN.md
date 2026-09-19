# LinkedIn Post

---

🚀 **Construí um pipeline completo de IA Generativa para documentos em português — e não foi fácil.**

Depois de muitas horas depurando erros, incompatibilidades e decisões arquiteturais difíceis, finalmente tenho um sistema funcionando de ponta a ponta.

O projeto combina **Apache Spark 4.x**, **Apache Tika**, **embeddings semânticos em português**, **StarRocks**, **dbt** e **RAG** para responder perguntas em linguagem natural sobre documentos PDF.

---

**Como funciona na prática?**

Você joga um PDF na pasta. O pipeline extrai o texto, gera embeddings semânticos, sumariza, classifica o tópico e armazena tudo. Depois é só perguntar:

> *"Qual foi o crescimento do PIB Construção no primeiro semestre de 2025?"*

E o sistema responde:

> *"O PIB da construção cresceu 1,8% no primeiro semestre de 2025 em relação ao mesmo período de 2024."*

Fundamentado no documento. Sem inventar.

---

**O que mais deu trabalho — e mais valeu a pena:**

🔴 **Compatibilidade Python entre containers**
O driver rodava Python 3.12. O worker do Spark rodava Python 3.10. O Spark Connect exige versões idênticas. Tentei instalar Python 3.12 via PPA no Ubuntu 22.04 — não funcionou. Solução: migrar o container dev para Python 3.10. Simples no papel, horas na prática.

🔴 **UDFs do Spark com PyTorch**
A ideia original era gerar embeddings como UDF distribuída no Spark. O problema: o worker não tem PyTorch. O Spark serializa a função via cloudpickle e tenta importar as dependências no worker — que não existem lá. Resultado: `ModuleNotFoundError: No module named 'torch'`. Solução: mover os embeddings para o driver após um collect(). Arquitetura diferente do planejado, resultado idêntico.

🔴 **SparseVector do MLlib quebrando o PyArrow**
Depois de resolver os embeddings, o `toPandas()` quebrava com erro de serialização. Causa: a coluna `tfidf_features` é um `SparseVector` — tipo nativo do PySpark que o PyArrow não consegue converter. Uma linha de código resolveu. Mas encontrar essa linha levou tempo.

🔴 **O RAG "funcionava" mas não respondia**
O pipeline rodava sem erros. Os chunks eram recuperados. O LLM era chamado. Mas a resposta era sempre: *"Não encontrei essa informação nos documentos."* O texto estava lá. O problema era a pergunta. Embeddings semânticos são sensíveis ao vocabulário — "PIB Construção do ano 2025" gerava um vetor distante do trecho "PIB CONSTRUÇÃO: SETOR CRESCE NO 1º SEMESTRE". Reformular a pergunta resolveu. Lição valiosa sobre como RAG realmente funciona.

---

**O que ficou de aprendizado:**

✅ Arquitetura de medalhas (Bronze → Silver → Gold) é poderosa para pipelines de documentos
✅ Para português, o modelo **PORTULAN serafim-100m** faz diferença real na qualidade do RAG
✅ Idempotência não é opcional — PRIMARY KEY no StarRocks e overwrite no Parquet salvam horas de debugging
✅ Em ambientes distribuídos, versão de dependência não é detalhe — é fundação

---

**Stack completa:**
Apache Spark 4.0.1 · Apache Tika · MinIO · Hive Metastore · Trino · StarRocks · dbt · Kafka · PyTorch + CUDA · PORTULAN serafim-100m · OpenAI / Gemini / Ollama · Streamlit · Docker · Python 3.10 · uv

---

O projeto é open source. Se você trabalha com documentos não estruturados, análise de relatórios ou quer entender como RAG funciona na prática — especialmente em português — vale dar uma olhada.

💬 Fico à disposição para trocar ideias sobre arquitetura de dados, LLMs e engenharia de ML.

#DataEngineering #MachineLearning #RAG #ApacheSpark #IA #Python #LLM #NLP #OpenSource #DataScience #PortuguesNLP
