"""
Fine-Tuning Distribuído — TorchDistributor (Spark 4.x)
=======================================================
Usa o TorchDistributor do Spark 4.x para ajustar finamente um modelo
de classificação de tópicos (DistilBERT) nos documentos da camada Silver,
distribuindo o treinamento pelos executores Spark (barrier execution).

Execução:
    python jobs/finetune_distributed.py
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPARK_URL = os.getenv("SPARK_CONNECT_URL", "sc://localhost:15002")
SILVER_PATH = os.getenv("SILVER_PATH", "s3a://warehouse/silver/documents_features")
MODEL_OUTPUT = os.getenv("FINETUNE_OUTPUT", "s3a://warehouse/models/topic_classifier")
BASE_MODEL = os.getenv("FINETUNE_BASE_MODEL", "distilbert-base-multilingual-cased")
NUM_WORKERS = int(os.getenv("FINETUNE_NUM_WORKERS", "1"))
EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "3"))

TOPICS = ["economia", "imóveis", "inflação", "mercado financeiro", "política", "tecnologia"]
LABEL2ID = {t: i for i, t in enumerate(TOPICS)}
ID2LABEL = {i: t for i, t in enumerate(TOPICS)}


def _train_fn(data: list[dict], epochs: int, base_model: str, output_path: str) -> None:
    """
    Função de treinamento executada em cada worker pelo TorchDistributor.
    Recebe os dados já coletados e treina localmente com PyTorch + HuggingFace.
    """
    import random
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

    # Garante reprodutibilidade
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    class DocDataset(Dataset):
        def __init__(self, records, tokenizer):
            self.records = records
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            enc = self.tokenizer(
                r["text_clean"][:512], truncation=True, padding="max_length",
                max_length=128, return_tensors="pt",
            )
            label = LABEL2ID.get(r.get("topic", "economia"), 0)
            return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(label)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(TOPICS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = DocDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_enc, labels in loader:
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**batch_enc, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, total_loss / len(loader))

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    log.info("✅ Modelo salvo em %s", output_path)


def run() -> None:
    from pyspark.sql import SparkSession
    from pyspark.ml.torch.distributor import TorchDistributor

    spark = SparkSession.builder.remote(SPARK_URL).getOrCreate()

    # Coleta dados do Silver para distribuir entre workers
    df = spark.read.parquet(SILVER_PATH).select("text_clean", "topic").filter(
        "topic IS NOT NULL AND text_clean IS NOT NULL"
    )
    data = df.toPandas().to_dict("records")
    log.info("📦 %d documentos carregados para fine-tuning", len(data))

    # TorchDistributor — executa _train_fn distribuído nos workers Spark
    # Nota: spark.stop() NÃO pode ser chamado antes do distributor.run()
    distributor = TorchDistributor(
        num_processes=NUM_WORKERS,
        local_mode=NUM_WORKERS == 1,
        use_gpu=False,
    )
    distributor.run(_train_fn, data, EPOCHS, BASE_MODEL, MODEL_OUTPUT)
    log.info("✅ Fine-tuning distribuído concluído")
    spark.stop()


if __name__ == "__main__":
    run()
