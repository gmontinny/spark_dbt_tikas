# The name of the Docker Compose file
COMPOSE_FILE = docker-compose.yml

.PHONY: help up down rebuild logs bronze silver gold pipeline-tika dbt-tika rag streaming finetune app connect-ingest submit-ingest

help: ## Lista todos os comandos disponíveis
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Sobe todos os serviços
	docker compose -f $(COMPOSE_FILE) up -d

down: ## Para todos os serviços
	docker compose -f $(COMPOSE_FILE) down

rebuild: ## Rebuild completo das imagens e reinicia
	docker compose -f $(COMPOSE_FILE) up -d --build

logs: ## Logs de todos os containers
	docker compose -f $(COMPOSE_FILE) logs -f

# ── Pipeline Tika + IA (Arquitetura de Medalhas) ─────────────────────────────
bronze: ## [Bronze] Extrai PDFs/DOCX com Apache Tika → MinIO
	docker compose exec dev python /workspace/jobs/bronze_tika_ingest.py

silver: ## [Silver] Feature engineering + embeddings semânticos (GPU)
	docker compose exec dev python /workspace/jobs/silver_features_embeddings.py

gold: ## [Gold] Sumarização + classificação zero-shot (GPU)
	docker compose exec dev python /workspace/jobs/gold_llm_inference.py

pipeline-tika: bronze silver gold ## Pipeline completo: Bronze → Silver → Gold

dbt-tika: ## dbt: transforma resultados LLM em marts analíticos
	docker compose exec dev bash -c "cd /workspace/dbt_project && dbt run --profiles-dir . && dbt test --profiles-dir ."

rag: ## RAG: pergunta sobre os documentos (ex: make rag Q="sua pergunta")
	@docker compose exec dev python /workspace/jobs/rag_query.py --question "$(Q)"

streaming: ## Inferência contínua via Structured Streaming (Kafka → StarRocks)
	docker compose exec dev python /workspace/jobs/streaming_inference.py

finetune: ## Fine-tuning distribuído com TorchDistributor (Spark 4.x)
	docker compose exec dev python /workspace/jobs/finetune_distributed.py

app: ## Interface RAG Streamlit (http://localhost:8501)
	docker compose exec dev streamlit run /workspace/app/streamlit_rag.py --server.port 8501 --server.address 0.0.0.0

# ── Exemplos PySpark Connect vs spark-submit ────────────────────────────────
connect-ingest: ## [Exemplo] Ingestão via PySpark Connect (gRPC)
	docker compose exec spark-connect python /opt/spark/work-dir/jobs/debug/pyspark_connect_ingest.py

submit-ingest: ## [Exemplo] Ingestão via spark-submit (driver local)
	docker compose exec spark-connect \
		spark-submit \
		--master local[*] \
		--packages org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
		--conf spark.sql.shuffle.partitions=4 \
		/opt/spark/work-dir/jobs/debug/spark_submit_ingest.py
