"""
ÉTAPE Modélisation — Entraînement du modèle de churn.

Un Pipeline scikit-learn encapsule TOUT le préprocessing + le modèle, de
sorte que rien ne fuit du test vers le train, et que le même objet peut
servir en production (une seule étape : predict).

Choix méthodologiques :
  - séparation train/test STRATIFIÉE (préserve le taux de churn)
  - validation croisée k-fold pour l'ajustement des hyperparamètres
  - métrique d'optimisation : ROC-AUC (robuste au déséquilibre)
  - gestion du déséquilibre : scale_pos_weight (XGB) / class_weight (RF)
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import get_logger, load_config, resolve

log = get_logger("train")

NUMERIC = ["recency", "frequency", "monetary", "avg_basket", "n_products",
           "total_qty", "tenure", "avg_interpurchase"]
CATEGORICAL = ["country"]


def _build_estimator(cfg: dict):
    """Renvoie (estimateur, grille) selon l'algo choisi en config."""
    algo = cfg["model"]["algorithm"]
    rs = cfg["project"]["random_state"]

    if algo == "xgboost":
        base = XGBClassifier(
            random_state=rs, eval_metric="logloss",
            tree_method="hist", n_jobs=-1,
        )
        grid = {f"model__{k}": v for k, v in cfg["model"]["xgboost"].items()}
    elif algo == "random_forest":
        base = RandomForestClassifier(
            random_state=rs, class_weight="balanced", n_jobs=-1,
        )
        grid = {f"model__{k}": v for k, v in cfg["model"]["random_forest"].items()}
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")
    return base, grid


def train(cfg: dict | None = None):
    cfg = cfg or load_config()
    rs = cfg["project"]["random_state"]

    df = pd.read_parquet(resolve(cfg["paths"]["features"]))
    X = df[NUMERIC + CATEGORICAL]
    y = df["churn"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["model"]["test_size"],
        stratify=y, random_state=rs,
    )
    log.info("Train : %d | Test : %d | churn(train)=%.1f%%",
             len(X_tr), len(X_te), 100 * y_tr.mean())

    # --- Préprocessing : scaling numérique + one-hot pays ---
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    # --- BASELINE : régression logistique -----------------------------------
    # Règle d'or : toujours comparer le modèle « avancé » à un modèle simple.
    # Si XGBoost ne bat pas nettement la baseline, la complexité ne se
    # justifie pas. La baseline sert de plancher de performance.
    baseline = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=rs)),
    ])
    baseline_auc = cross_val_score(
        baseline, X_tr, y_tr, scoring="roc_auc",
        cv=cfg["model"]["cv_folds"], n_jobs=-1).mean()
    log.info("BASELINE  LogisticRegression — ROC-AUC (CV) : %.3f", baseline_auc)
    baseline.fit(X_tr, y_tr)
    baseline_auc_test = roc_auc_score(y_te, baseline.predict_proba(X_te)[:, 1])
    log.info("BASELINE  LogisticRegression — ROC-AUC (test) : %.3f", baseline_auc_test)

    base, grid = _build_estimator(cfg)

    # Déséquilibre pour XGBoost : poids de la classe positive
    if cfg["model"]["algorithm"] == "xgboost":
        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        base.set_params(scale_pos_weight=pos_weight)

    pipe = Pipeline([("pre", pre), ("model", base)])

    search = GridSearchCV(
        pipe, grid, scoring="roc_auc",
        cv=cfg["model"]["cv_folds"], n_jobs=-1, refit=True,
    )
    log.info("Recherche d'hyperparamètres (%d folds)...", cfg["model"]["cv_folds"])
    search.fit(X_tr, y_tr)
    log.info("Meilleurs paramètres : %s", search.best_params_)
    log.info("ROC-AUC (CV) : %.3f", search.best_score_)

    best = search.best_estimator_

    gain = search.best_score_ - baseline_auc
    log.info("Gain %s vs baseline (CV) : %+.3f AUC",
             cfg["model"]["algorithm"], gain)

    # --- SÉLECTION DU CHAMPION (en CV, décision prise AVANT de voir le test) --
    # Si le modèle complexe ne bat pas la baseline en validation croisée, la
    # complexité ne se justifie pas : on déploie le modèle simple (plus léger,
    # plus interprétable). La complexité doit se mériter.
    if search.best_score_ > baseline_auc:
        champion, champ_name, champ_cv = best, cfg["model"]["algorithm"], search.best_score_
    else:
        champion, champ_name, champ_cv = baseline, "logistic_regression", baseline_auc
    log.info("CHAMPION déployé : %s (ROC-AUC CV = %.3f)", champ_name, champ_cv)

    joblib.dump(champion, resolve(cfg["paths"]["model"]))
    log.info("Modèle sauvegardé : %s", resolve(cfg["paths"]["model"]))

    baseline_scores = {"roc_auc_cv": round(float(baseline_auc), 4),
                       "roc_auc_test": round(float(baseline_auc_test), 4)}
    best_params = search.best_params_ if champ_name != "logistic_regression" else {}
    # On renvoie tout ce qu'il faut à l'évaluation
    return champion, X_te, y_te, best_params, champ_cv, baseline_scores, champ_name


if __name__ == "__main__":
    train()
