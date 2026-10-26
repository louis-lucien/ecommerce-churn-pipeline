"""
ÉTAPE Inférence — Scorer un client à partir de ses features RFM.

Utilisé par l'API et le dashboard. Charge le modèle une seule fois.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src.config import get_logger, load_config, resolve
from src.train import CATEGORICAL, NUMERIC

log = get_logger("predict")


@lru_cache(maxsize=1)
def _load_model():
    cfg = load_config()
    path = resolve(cfg["paths"]["model"])
    log.info("Chargement du modèle : %s", path)
    return joblib.load(path)


def predict_one(features: dict, cfg: dict | None = None) -> dict:
    """features = dict avec les clés NUMERIC + 'country'."""
    cfg = cfg or load_config()
    model = _load_model()
    row = {k: features.get(k) for k in NUMERIC + CATEGORICAL}
    X = pd.DataFrame([row])
    proba = float(model.predict_proba(X)[:, 1][0])
    thr = cfg["business"]["alert_threshold"]
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": int(proba >= thr),
        "risk_level": _risk(proba),
        "threshold": thr,
    }


def _risk(p: float) -> str:
    if p >= 0.70:
        return "ÉLEVÉ"
    if p >= 0.40:
        return "MODÉRÉ"
    return "FAIBLE"


if __name__ == "__main__":
    demo = {"recency": 120, "frequency": 2, "monetary": 90.0,
            "avg_basket": 45.0, "n_products": 3, "total_qty": 40,
            "tenure": 200, "avg_interpurchase": 100.0, "country": "United Kingdom"}
    print(predict_one(demo))
