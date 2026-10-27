import streamlit as st

import ui
from loaders import load_layer
from src.predict import predict_one

features = load_layer("features")

ui.page_header("Modèle", "Scoring d'un client",
               "Renseignez le profil RFM d'un client : le modèle déployé calcule sa probabilité de départ.")

with st.form("scoring"):
    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Récence (jours depuis le dernier achat)", 0, 365, 60)
        frequency = st.number_input("Fréquence (nombre de commandes)", 1, 50, 5)
        monetary = st.number_input("Montant total dépensé (€)", 0.0, 1e5, 400.0, step=50.0)
    with c2:
        avg_basket = st.number_input("Panier moyen (€)", 0.0, 1e4, 80.0, step=10.0)
        n_products = st.number_input("Produits distincts achetés", 1, 60, 10)
        total_qty = st.number_input("Quantité totale", 1, 2000, 150)
    with c3:
        tenure = st.number_input("Ancienneté (jours)", 1, 400, 250)
        avg_interpurchase = st.number_input("Intervalle moyen entre commandes (jours)", 1.0, 365.0, 50.0)
        country = st.selectbox("Pays", sorted(features["country"].unique()))
    submitted = st.form_submit_button("Calculer le score", type="primary", width="stretch")

if submitted:
    result = predict_one(dict(recency=recency, frequency=frequency, monetary=monetary, avg_basket=avg_basket,
                              n_products=n_products, total_qty=total_qty, tenure=tenure,
                              avg_interpurchase=avg_interpurchase, country=country))
    proba = result["churn_probability"]
    ui.section("Résultat")
    ui.kpi_row([
        ("Probabilité de churn", f"{proba:.1%}", "", True),
        ("Niveau de risque", result["risk_level"].capitalize()),
        ("Décision", "Action de rétention" if result["churn_prediction"] else "Aucune action"),
    ])
    st.progress(proba)
    if result["churn_prediction"]:
        st.warning("Client à risque : une action de rétention ciblée est recommandée.")
    else:
        st.success("Client fidèle : pas de budget de rétention à engager.")
