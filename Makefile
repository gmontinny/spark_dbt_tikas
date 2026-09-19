# The name of the Docker Compose file
COMPOSE_FILE = docker-compose.yml

.PHONY: help up down rebuild logs browse ml-demo analise otimizacao ml-all ingest dbt-run dbt-test connect-ingest submit-ingest bronze silver gold pipeline-tika dbt-tika

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Run the containers in the background
up: ## Start development environment
	docker compose -f $(COMPOSE_FILE) up -d

# Stop the containers
down: ## Stop containers
	docker compose -f $(COMPOSE_FILE) down

# Rebuild the Docker image and restart the containers
rebuild: down up ## Rebuild and restart containers

# Show logs
logs: ## Show container logs
	docker compose -f $(COMPOSE_FILE) logs

# Open the browser
browse: ## Open Spark UI in browser
	open http://localhost:4040

# Machine Learning targets
ml-demo: ## Execute complete ML pipeline demo
	docker compose exec dev python src/ml_pipeline_demo.py

analise: ## Run exploratory data analysis
	docker compose exec dev python src/analise_exploratoria.py

otimizacao: ## Run model optimization and comparison
	docker compose exec dev python src/otimizacao_modelo.py

ml-all: ## Run all ML scripts in sequence
	make analise
	make ml-demo
	make otimizacao

teste: ## Quick environment test
	docker compose exec dev python src/teste_rapido.py

# Data Pipeline: Spark ingest + dbt transform
ingest: ## Ingest CSV files into raw tables via Spark
	docker compose exec dev python src/migrate_csv_to_trino.py

dbt-run: ## Run dbt models (staging + marts)
	docker compose exec dev bash -c "cd dbt_olist && dbt run --profiles-dir ."

dbt-test: ## Run dbt tests
	docker compose exec dev bash -c "cd dbt_olist && dbt test --profiles-dir ."

pipeline: ingest dbt-run ## Full pipeline: ingest raw + dbt transform

# ── Exemplos PySpark Connect vs spark-submit ────────────────────────────────
connect-ingest: ## [Exemplo 1] Ingestão via PySpark Connect (cliente remoto gRPC)
	docker compose exec spark-connect python /opt/spark/work-dir/jobs/debug/pyspark_connect_ingest.py

submit-ingest: ## [Exemplo 2] Ingestão via spark-submit (driver local no container)
	docker compose exec spark-connect \
		spark-submit \
		--master local[*] \
		--packages org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.262 \
		--conf spark.sql.shuffle.partitions=4 \
		/opt/spark/work-dir/jobs/debug/spark_submit_ingest.py

# ── Pipeline Tika + IA (Arquitetura de Medalhas) ─────────────────────────────
bronze: ## [Bronze] Extrai PDFs/DOCX com Apache Tika → MinIO
	docker compose exec spark-connect python /opt/spark/work-dir/jobs/bronze_tika_ingest.py

silver: ## [Silver] Feature engineering + embeddings semânticos
	docker compose exec spark-connect python /opt/spark/work-dir/jobs/silver_features_embeddings.py

gold: ## [Gold] Inferência LLM distribuída (sumarização + classificação)
	docker compose exec spark-connect python /opt/spark/work-dir/jobs/gold_llm_inference.py

pipeline-tika: bronze silver gold ## Pipeline completo: Bronze → Silver → Gold

dbt-tika: ## dbt: transforma resultados LLM em marts analíticos
	docker compose exec dev bash -c "cd dbt_project && dbt run --profiles-dir . && dbt test --profiles-dir ."

rag: ## RAG: pergunta sobre os documentos via LLM
	@read -p "Pergunta: " q; docker compose exec dev python /workspace/jobs/rag_query.py --question "$$q"

streaming: ## Inferência contínua via Structured Streaming (Kafka → StarRocks)
	docker compose exec spark-connect python /opt/spark/work-dir/jobs/streaming_inference.py

finetune: ## Fine-tuning distribuído com TorchDistributor (Spark 4.x)
	docker compose exec spark-connect python /opt/spark/work-dir/jobs/finetune_distributed.py

app: ## Interface RAG Streamlit
	docker compose exec dev streamlit run /workspace/app/streamlit_rag.py --server.port 8501 --server.address 0.0.0.0
