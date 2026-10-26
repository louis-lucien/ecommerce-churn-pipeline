"""
Orchestration du pipeline avec Prefect.

Chaque étape devient une @task ; le @flow les enchaîne dans l'ordre et
fournit un suivi d'exécution (logs, durées, reprises sur erreur).

Lancer :  python -m pipelines.flow
Ou via l'UI Prefect :  prefect server start   puis   python -m pipelines.flow
"""
from __future__ import annotations

from prefect import flow, task

from src.cleaning import clean
from src.config import load_config
from src.eda import run_eda
from src.evaluate import evaluate
from src.features import build_features
from src.ingestion import extract
from src.train import train


@task(name="extract", retries=2, retry_delay_seconds=10)
def t_extract(cfg):
    # retries=2 : utile si le téléchargement kagglehub échoue ponctuellement
    return extract(cfg)


@task(name="clean")
def t_clean(df, cfg):
    return clean(df, cfg)


@task(name="features")
def t_features(df, cfg):
    return build_features(df, cfg)


@task(name="eda")
def t_eda(cfg):
    return run_eda(cfg)


@task(name="train")
def t_train(cfg):
    return train(cfg)


@task(name="evaluate")
def t_evaluate(model, X_te, y_te, bp, cv_auc, baseline, champ, cfg):
    return evaluate(model, X_te, y_te, bp, cv_auc, baseline, champ, cfg)


@flow(name="ecommerce-churn-pipeline", log_prints=True)
def run_pipeline():
    cfg = load_config()
    raw = t_extract(cfg)
    clean_df = t_clean(raw, cfg)
    t_features(clean_df, cfg)
    t_eda(cfg)
    model, X_te, y_te, bp, cv_auc, baseline, champ = t_train(cfg)
    metrics = t_evaluate(model, X_te, y_te, bp, cv_auc, baseline, champ, cfg)
    print("Pipeline terminé. ROC-AUC test =", metrics["roc_auc_test"],
          "| champion :", metrics["champion_deployed"])
    return metrics


if __name__ == "__main__":
    run_pipeline()
