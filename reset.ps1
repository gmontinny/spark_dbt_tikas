# Reset completo do ambiente e execucao do pipeline
# Uso: .\reset.ps1

$ErrorActionPreference = "Stop"

Write-Host "[RESET] Parando containers e removendo volumes..." -ForegroundColor Cyan
docker-compose down -v

Write-Host "[CLEAN] Removendo dados locais do MinIO..." -ForegroundColor Cyan
if (Test-Path "minio_data") {
    Remove-Item -Recurse -Force "minio_data"
}

Write-Host "[START] Subindo containers..." -ForegroundColor Cyan
docker-compose up -d

Write-Host "[WAIT] Aguardando servicos ficarem healthy..." -ForegroundColor Yellow
do {
    Start-Sleep -Seconds 5
    $status = docker-compose ps 2>&1 | Out-String
    Write-Host "   Aguardando hive-metastore..." -ForegroundColor Gray
} until ($status -match "hive-metastore.*healthy")

Write-Host "[WAIT] Aguardando Trino..." -ForegroundColor Yellow
do {
    Start-Sleep -Seconds 5
    $status = docker-compose ps 2>&1 | Out-String
    Write-Host "   Aguardando trino..." -ForegroundColor Gray
} until ($status -match "trino.*healthy")

Write-Host "[WAIT] Aguardando Spark Connect..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "[RUN] Executando pipeline (ingest + dbt)..." -ForegroundColor Cyan
make pipeline

Write-Host "[OK] Pipeline concluido com sucesso!" -ForegroundColor Green
