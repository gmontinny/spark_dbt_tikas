#!/bin/bash
# Reset completo do ambiente e execução do pipeline
# Uso: ./reset.sh

set -e

echo "🔄 Parando containers e removendo volumes..."
docker-compose down -v

echo "🗑️  Removendo dados locais do MinIO..."
rm -rf minio_data

echo "🚀 Subindo containers..."
docker-compose up -d

echo "⏳ Aguardando serviços ficarem healthy..."
until docker-compose ps | grep -q "hive-metastore.*healthy"; do
  sleep 5
  echo "   Aguardando hive-metastore..."
done

echo "⏳ Aguardando Trino..."
until docker-compose ps | grep -q "trino.*healthy"; do
  sleep 5
  echo "   Aguardando trino..."
done

echo "⏳ Aguardando Spark Connect..."
sleep 10

echo "📦 Executando pipeline (ingest + dbt)..."
make pipeline

echo "✅ Pipeline concluído com sucesso!"
