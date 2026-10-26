"""
API REST d'inférence (FastAPI).

Expose le modèle de churn en temps réel.

Lancer :  uvicorn api.main:app --reload
Docs interactives :  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_one

app = FastAPI(
    title="E-commerce Churn API",
    description="Prédiction du risque de désabonnement client à partir des indicateurs RFM.",
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    """Indicateurs comportementaux d'un client (fenêtre d'observation)."""
    recency: float = Field(..., example=45, description="Jours depuis le dernier achat")
    frequency: int = Field(..., example=6, description="Nombre de commandes")
    monetary: float = Field(..., example=520.0, description="CA total généré")
    avg_basket: float = Field(..., example=86.7, description="Panier moyen")
    n_products: int = Field(..., example=14, description="Nombre de produits distincts")
    total_qty: int = Field(..., example=210, description="Quantité totale achetée")
    tenure: int = Field(..., example=300, description="Ancienneté en jours")
    avg_interpurchase: float = Field(..., example=50.0, description="Intervalle moyen entre commandes")
    country: str = Field("United Kingdom", example="United Kingdom")


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: CustomerFeatures):
    """Renvoie la probabilité de churn et le niveau de risque."""
    return predict_one(features.model_dump())
