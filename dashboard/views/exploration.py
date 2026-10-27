import streamlit as st

import ui
from loaders import figure_path, load_layer

features = load_layer("features")

ui.page_header("Données", "Analyse exploratoire",
               "Comportement d'achat et premiers signaux du churn, avant toute modélisation.")

ui.kpi_row([
    ("Clients", ui.fmt_int(len(features)), "", True),
    ("Taux de churn", f"{features['churn'].mean():.1%}"),
    ("CA moyen / client", f"{features['monetary'].mean():,.0f} €".replace(",", " ")),
    ("Récence médiane", f"{features['recency'].median():.0f} j", "depuis le dernier achat"),
])

ui.section("Activité commerciale")
c1, c2 = st.columns(2)
with c1:
    ui.figure(figure_path("eda_monthly_revenue.png"), "Chiffre d'affaires mensuel")
with c2:
    ui.figure(figure_path("eda_top_products.png"), "Produits les plus vendus")

ui.section("Profil des clients", "La récence sépare nettement clients fidèles et clients partis.")
c1, c2 = st.columns(2)
with c1:
    ui.figure(figure_path("eda_recency_by_churn.png"), "Récence selon le statut de churn")
with c2:
    ui.figure(figure_path("eda_rfm_distributions.png"), "Distributions RFM")

ui.section("Comparaison des segments")
summary = (features.groupby("churn")[["recency", "frequency", "monetary", "avg_basket", "tenure"]]
           .median().rename(index={0: "Fidèles", 1: "Churn"}).T.round(1))
summary.index = ["Récence (j)", "Fréquence (commandes)", "Montant (€)", "Panier moyen (€)", "Ancienneté (j)"]
st.dataframe(summary, width="stretch")
st.caption("Valeurs médianes par segment.")
