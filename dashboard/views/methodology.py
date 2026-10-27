import streamlit as st

import ui

ui.page_header("Documentation", "Méthodologie",
               "Choix de conception du pipeline et garde-fous contre les biais d'évaluation.")

ui.section("Définition de la cible")
st.markdown("""
Le churn n'existe pas en tant que colonne dans les données transactionnelles : il est **construit**.
L'historique est coupé à une date **T0** :

- les indicateurs du client sont calculés sur la période **antérieure à T0** ;
- l'étiquette vaut 1 si le client **n'achète pas** dans la fenêtre **[T0, T0 + horizon]**.

Ce découpage garantit qu'aucune information postérieure à T0 n'entre dans les variables explicatives.
""")

ui.section("Variables (RFM étendu)")
st.markdown("""
| Variable | Définition |
|---|---|
| `recency` | Jours entre le dernier achat et T0 |
| `frequency` | Nombre de commandes distinctes |
| `monetary` | Chiffre d'affaires cumulé |
| `avg_basket` | Montant moyen par commande |
| `n_products` | Nombre de produits distincts achetés |
| `total_qty` | Quantité totale d'articles |
| `tenure` | Ancienneté du client à T0 |
| `avg_interpurchase` | Intervalle moyen entre deux commandes |
| `country` | Pays (variable catégorielle encodée) |
""")

ui.section("Protocole d'évaluation")
st.markdown("""
- Jeu de test tenu à l'écart, jamais utilisé pour la sélection.
- Sélection du champion par **validation croisée** sur l'entraînement.
- Comparaison systématique à une **baseline** (régression logistique).
- Métrique principale : **ROC-AUC** (indépendante du seuil) ; précision et rappel au seuil de décision.
""")

ui.section("Architecture technique")
st.markdown("""
| Brique | Implémentation |
|---|---|
| Ingestion | `src/ingestion.py` : acquisition (Kaggle ou local) et contrôle du schéma |
| Nettoyage | `src/cleaning.py` : règles métier traçables, sortie Parquet |
| Features | `src/features.py` : agrégation RFM par client |
| Entraînement | `src/train.py` : pipeline scikit-learn, recherche d'hyperparamètres |
| Évaluation | `src/evaluate.py` : métriques et figures dans `reports/` |
| Orchestration | `pipelines/flow.py`, `run_pipeline.py` |
| Service | `api/main.py` (API REST), `dashboard/` (cette application) |
| Configuration | `config/config.yaml` : chemins, paramètres, seuils |
""")

ui.section("Limites")
st.markdown("""
- Jeu de démonstration synthétique, calibré sur le schéma du dataset *Online Retail* (Kaggle).
- Horizon de churn fixe : un client saisonnier peut être étiqueté à tort.
- Pas de suivi de dérive des données en production.
""")
