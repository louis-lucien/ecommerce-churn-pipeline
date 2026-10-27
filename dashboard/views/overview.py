import streamlit as st

import ui
from loaders import champion_label, layer_stats, load_layer, load_metrics

metrics = load_metrics()
features = load_layer("features")

ui.page_header("Vue d'ensemble", "Prédiction de l'attrition client",
               "Identifier, à partir de l'historique de ventes, les clients qui ne rachèteront pas "
               "dans la fenêtre d'observation, afin de cibler les actions de rétention.")

ui.kpi_row([
    ("Clients modélisés", ui.fmt_int(len(features)), "une ligne par client", True),
    ("Taux de churn", f"{features['churn'].mean():.1%}", "sur la base modélisée"),
    ("ROC-AUC test", f"{metrics.get('roc_auc_test', 0):.3f}", "clients jamais vus", True),
    ("Modèle déployé", champion_label(metrics), "sélection sur validation croisée"),
])

ui.section("Chaîne de traitement", "Du fichier de ventes brut au service de scoring.")
raw, interim, feats = (layer_stats(k) for k in ("raw", "interim", "features"))
ui.flow([
    ("Ingestion", ui.fmt_int(raw["rows"]), "lignes de vente · contrôle du schéma"),
    ("Nettoyage", ui.fmt_int(interim["rows"]), "transactions valides · Parquet"),
    ("Features RFM", ui.fmt_int(feats["rows"]), "clients · récence, fréquence, montant"),
    ("Étiquetage", f"{features['churn'].mean():.0%}", "de churn · fenêtre [T0, T0 + horizon]"),
    ("Modèle", champion_label(metrics), "comparé à une baseline logistique"),
    ("Service", "API + dashboard", "scoring unitaire"),
])

ui.section("Points clés")
c1, c2 = st.columns(2)
with c1:
    ui.note("<b>Anti-fuite temporelle.</b> Les indicateurs sont calculés avant T0, l'étiquette est "
            "observée après T0 : le modèle n'a jamais accès au futur pendant l'apprentissage.")
with c2:
    gain = metrics.get("gain_vs_baseline_test")
    if gain is not None:
        ui.note(f"<b>Champion vs baseline.</b> Écart de ROC-AUC test de {gain:+.4f} face à la régression "
                "logistique : le modèle simple est au niveau. Voir la page Performance.")
