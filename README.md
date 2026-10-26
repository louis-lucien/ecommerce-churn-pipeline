# 📉 E-commerce Churn Pipeline — Data Engineering + ML

Pipeline **de bout en bout** qui, à partir de l'historique de transactions
d'un e-commerce, prédit quels clients risquent de **ne plus jamais acheter**
(*churn*), et expose cette prédiction via une **API** et un **dashboard**.

> Dataset de référence : [Online Retail](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
> (transactions d'un détaillant en ligne britannique, 2010–2011).

---

## 🎯 Ce que fait ce projet

```
CSV brut ──► Ingestion ──► Nettoyage ──► Feature Engineering ──► Modèle ML ──► API + Dashboard
 (Extract)   (contrôle    (règles       (RFM + label churn      (XGBoost,     (FastAPI +
             qualité)     métier)       par fenêtre temporelle)  ROC-AUC)     Streamlit)
```

Le pipeline suit une architecture **en couches** (medallion) :
`data/raw` → `data/interim` (Parquet propre) → `data/processed` (features).

---

## 📁 Structure

```
ecommerce-churn-pipeline/
├── config/config.yaml         # tous les paramètres (aucun code en dur)
├── data/{raw,interim,processed}
├── src/
│   ├── config.py              # config + logging
│   ├── ingestion.py           # E : kagglehub/local + contrôle qualité
│   ├── cleaning.py            # T : nettoyage traçable -> Parquet
│   ├── features.py            # RFM + définition du churn (fenêtrage)
│   ├── eda.py                 # analyse exploratoire (4 figures)
│   ├── train.py               # baseline LogReg + XGBoost + CV
│   ├── evaluate.py            # métriques + figures
│   ├── predict.py             # inférence
│   └── generate_sample_data.py# génère un jeu synthétique fidèle
├── pipelines/flow.py          # orchestration Prefect (optionnel)
├── api/main.py                # API REST FastAPI
├── dashboard/app.py           # dashboard Streamlit
├── tests/test_pipeline.py     # tests unitaires
├── run_pipeline.py            # ▶ point d'entrée principal
└── requirements.txt
```

---

## 🚀 Démarrage rapide

```bash
# 1. Environnement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Données — automatique via kagglehub (mode par défaut)
#    Le pipeline télécharge lui-même le vrai dataset Kaggle au premier run :
#       import kagglehub
#       path = kagglehub.dataset_download("carrie1/ecommerce-data")
#    (intégré dans src/ingestion.py — rien à faire de ta part)
#
#    Sans internet ? Deux replis possibles :
#       python -m src.generate_sample_data      # jeu synthétique fidèle
#    ou placer manuellement le CSV dans data/raw/online_retail.csv
#    (et mettre data_source.mode: "local" dans config/config.yaml)

# 3. Lancer tout le pipeline (téléchargement -> ... -> évaluation)
python run_pipeline.py

# 4. Servir le modèle
uvicorn api.main:app --reload          # API  -> http://127.0.0.1:8000/docs
streamlit run dashboard/app.py         # Dashboard

# (optionnel) Orchestration Prefect
python -m pipelines.flow
```

---

## 📊 Résultats (jeu de démonstration)

| Métrique        | Valeur |
|-----------------|--------|
| ROC-AUC (CV)    | 0.895  |
| ROC-AUC (test)  | 0.925  |
| Baseline LogReg (test) | 0.931 |
| Précision       | 0.815  |
| Rappel          | 0.847  |
| F1-score        | 0.831  |

Variable la plus prédictive : **la récence** (jours depuis le dernier achat) —
résultat attendu et cohérent avec la littérature RFM.

---

## 🧠 Choix méthodologiques clés

1. **Définition honnête du churn.** Pas de seuil arbitraire : on coupe
   l'historique à un *snapshot* T0, on calcule les features **avant** T0 et
   on observe le retour du client **après** T0. Aucune fuite de données.
2. **Pipeline scikit-learn unique** englobant préprocessing + modèle : le
   même objet sert en entraînement et en production.
3. **Config-driven** : chaque paramètre métier est dans `config.yaml`.
4. **Traçabilité** : chaque filtre de nettoyage est loggé et compté.
5. **Reproductibilité** : graine aléatoire unique, données versionnées en Parquet.

Voir `GUIDE_PEDAGOGIQUE.md` pour l'explication détaillée, étape par étape.

---

## 🧪 Tests

```bash
pytest -q
```

---

## 🌍 Déploiement

- **API** : conteneurisable (Docker) puis déployable sur Render / Railway / Fly.io.
- **Dashboard** : Streamlit Community Cloud ou Hugging Face Spaces.
- **Pipeline** : planifiable via Prefect Cloud (déploiement `cron`).
