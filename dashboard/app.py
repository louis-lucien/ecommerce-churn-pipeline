"""
Dashboard Streamlit — exploration, résultats du modèle et scoring interactif.

Lancer :  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import load_config, resolve
from src.predict import predict_one
from src.train import NUMERIC

st.set_page_config(page_title="E-commerce Churn", page_icon="📉", layout="wide")
cfg = load_config()


@st.cache_data
def load_features():
    return pd.read_parquet(resolve(cfg["paths"]["features"]))


@st.cache_data
def load_metrics():
    p = resolve(cfg["paths"]["metrics"])
    return json.loads(p.read_text()) if p.exists() else {}


st.title("📉 Prédiction du churn client — E-commerce")
st.caption("Pipeline Data Engineering + ML | RFM + XGBoost")

feats = load_features()
metrics = load_metrics()
tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Performance du modèle", "Scorer un client"])

# --- Onglet 1 : vue d'ensemble ---
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients modélisés", f"{len(feats):,}")
    c2.metric("Taux de churn", f"{feats['churn'].mean():.1%}")
    c3.metric("CA moyen / client", f"{feats['monetary'].mean():,.0f}")
    c4.metric("Récence médiane (j)", f"{feats['recency'].median():.0f}")
    st.subheader("Analyse exploratoire")
    figdir = resolve(cfg["paths"]["figures"])
    r1 = st.columns(2)
    for col, name in zip(r1, ["eda_monthly_revenue.png", "eda_top_products.png"]):
        f = figdir / name
        if f.exists():
            col.image(str(f), width='stretch')
    r2 = st.columns(2)
    for col, name in zip(r2, ["eda_recency_by_churn.png", "eda_rfm_distributions.png"]):
        f = figdir / name
        if f.exists():
            col.image(str(f), width='stretch')
    st.subheader("Aperçu des features")
    st.dataframe(feats.head(20), width='stretch')

# --- Onglet 2 : performance ---
with tab2:
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC (test)", metrics["roc_auc_test"])
        c2.metric("Précision", metrics["precision"])
        c3.metric("Rappel", metrics["recall"])
        c4.metric("F1-score", metrics["f1"])
    figdir = resolve(cfg["paths"]["figures"])
    cols = st.columns(4)
    for col, name in zip(cols, ["roc_curve.png", "confusion_matrix.png",
                                "feature_importance.png",
                                "baseline_comparison.png"]):
        f = figdir / name
        if f.exists():
            col.image(str(f), width='stretch')

# --- Onglet 3 : scoring interactif ---
with tab3:
    st.write("Ajustez les indicateurs d'un client pour estimer son risque de départ.")
    c1, c2 = st.columns(2)
    with c1:
        recency = st.slider("Récence (jours depuis dernier achat)", 0, 365, 60)
        frequency = st.slider("Fréquence (nb commandes)", 1, 50, 5)
        monetary = st.number_input("Monetary (CA total)", 0.0, 1e5, 400.0)
        avg_basket = st.number_input("Panier moyen", 0.0, 1e4, 80.0)
    with c2:
        n_products = st.slider("Nb produits distincts", 1, 60, 10)
        total_qty = st.slider("Quantité totale", 1, 2000, 150)
        tenure = st.slider("Ancienneté (jours)", 1, 400, 250)
        avg_interpurchase = st.number_input("Intervalle moyen entre commandes", 1.0, 365.0, 50.0)
    country = st.selectbox("Pays", sorted(feats["country"].unique()))

    if st.button("Prédire le risque", type="primary"):
        payload = dict(recency=recency, frequency=frequency, monetary=monetary,
                       avg_basket=avg_basket, n_products=n_products,
                       total_qty=total_qty, tenure=tenure,
                       avg_interpurchase=avg_interpurchase, country=country)
        res = predict_one(payload)
        p = res["churn_probability"]
        st.metric("Probabilité de churn", f"{p:.1%}", res["risk_level"])
        st.progress(p)
        if res["churn_prediction"]:
            st.error(f"⚠️ Client à risque ({res['risk_level']}) — action de rétention recommandée.")
        else:
            st.success("✅ Client fidèle — pas d'alerte.")
