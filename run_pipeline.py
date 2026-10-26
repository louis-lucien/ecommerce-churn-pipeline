"""
Point d'entrée principal — exécute tout le pipeline en une commande.

    python run_pipeline.py

Enchaîne : ingestion -> nettoyage -> features -> entraînement -> évaluation.
Version sans orchestrateur (aucune dépendance lourde). Pour la version
orchestrée avec suivi visuel, voir pipelines/flow.py (Prefect).
"""
from __future__ import annotations

import time

from src.cleaning import clean
from src.config import get_logger, load_config
from src.eda import run_eda
from src.evaluate import evaluate
from src.features import build_features
from src.ingestion import extract
from src.train import train

log = get_logger("pipeline")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("=== DÉBUT DU PIPELINE : %s v%s ===",
             cfg["project"]["name"], cfg["project"]["version"])

    raw = extract(cfg)                                   # 1. Extract
    clean_df = clean(raw, cfg)                           # 2. Transform
    build_features(clean_df, cfg)                        # 3. Features + label
    run_eda(cfg)                                         # 4. Analyse exploratoire
    model, X_te, y_te, bp, cv_auc, baseline, champ = train(cfg)  # 5. Baseline + champion
    metrics = evaluate(model, X_te, y_te, bp, cv_auc,    # 6. Évaluation
                       baseline, champ, cfg)

    log.info("=== FIN DU PIPELINE en %.1fs — ROC-AUC test = %.3f "
             "(baseline %.3f) | champion : %s ===",
             time.time() - t0, metrics["roc_auc_test"],
             metrics["baseline_logistic_regression"]["roc_auc_test"],
             metrics["champion_deployed"])


if __name__ == "__main__":
    main()
