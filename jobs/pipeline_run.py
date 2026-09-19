"""
Pipeline Orquestrador — Bronze → Silver → Gold
===============================================
Executa as três etapas da arquitetura de medalhas em sequência.

Execução:
    python jobs/pipeline_run.py
    python jobs/pipeline_run.py --layer bronze   # apenas Bronze
    python jobs/pipeline_run.py --layer silver   # apenas Silver
    python jobs/pipeline_run.py --layer gold     # apenas Gold
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_bronze():
    from bronze_tika_ingest import run
    log.info("▶ Iniciando camada Bronze (Tika extraction)...")
    run()


def run_silver():
    from silver_features_embeddings import run
    log.info("▶ Iniciando camada Silver (features + embeddings)...")
    run()


def run_gold():
    from gold_llm_inference import run
    log.info("▶ Iniciando camada Gold (LLM inference)...")
    run()


LAYERS = {"bronze": run_bronze, "silver": run_silver, "gold": run_gold}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=list(LAYERS), default=None)
    args = parser.parse_args()

    if args.layer:
        LAYERS[args.layer]()
    else:
        for name, fn in LAYERS.items():
            log.info("=" * 50)
            fn()
        log.info("✅ Pipeline completo: Bronze → Silver → Gold")
