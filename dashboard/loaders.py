"""Accès aux données et artefacts du pipeline, mis en cache pour toutes les pages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config, resolve

cfg = load_config()

# Couches du pipeline : clé de config -> (libellé, rôle)
LAYERS = {
    "raw": ("Brute", "CSV de ventes tel qu'il est reçu : une ligne par article vendu, défauts compris "
                     "(clients anonymes, annulations, prix nuls)."),
    "interim": ("Nettoyée", "Règles métier appliquées : lignes invalides retirées, types corrigés, "
                            "chiffre d'affaires calculé. Format Parquet."),
    "features": ("Features", "Une ligne par client : indicateurs RFM et étiquette churn. "
                             "Jeu d'entrée du modèle."),
}


@st.cache_data
def load_layer(name: str) -> pd.DataFrame:
    path = resolve(cfg["paths"][name])
    if name == "raw":
        return pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str})
    return pd.read_parquet(path)


@st.cache_data
def layer_stats(name: str) -> dict:
    df = load_layer(name)
    path = resolve(cfg["paths"][name])
    return {"rows": len(df), "cols": df.shape[1], "size_mb": path.stat().st_size / 1e6,
            "missing": df.isna().mean().mean(), "format": path.suffix[1:].upper()}


@st.cache_data
def column_profile(name: str) -> pd.DataFrame:
    df = load_layer(name)
    return pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Manquants (%)": (df.isna().mean() * 100).round(1).values,
        "Valeurs distinctes": df.nunique().values,
        "Exemple": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
    })


@st.cache_data
def load_metrics() -> dict:
    path = resolve(cfg["paths"]["metrics"])
    return json.loads(path.read_text()) if path.exists() else {}


def figure_path(name: str) -> Path:
    return resolve(cfg["paths"]["figures"]) / name


MODEL_NAMES = {"xgboost": "XGBoost", "logistic_regression": "Régression logistique",
               "random_forest": "Random Forest"}


def champion_label(metrics: dict) -> str:
    name = metrics.get("champion_deployed", "—")
    return MODEL_NAMES.get(name, name.replace("_", " ").title())
