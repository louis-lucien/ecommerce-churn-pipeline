import pandas as pd
import streamlit as st

import ui
from loaders import champion_label, figure_path, load_metrics

metrics = load_metrics()

ui.page_header("Modèle", "Performance du modèle",
               f"Évaluation de {champion_label(metrics)} sur un jeu de test de "
               f"{ui.fmt_int(metrics.get('n_test', 0))} clients jamais vus à l'entraînement.")

if not metrics:
    st.warning("Aucune métrique trouvée : lancez `python run_pipeline.py`.")
    st.stop()

ui.kpi_row([
    ("ROC-AUC test", f"{metrics['roc_auc_test']:.3f}", f"CV : {metrics['roc_auc_cv']:.3f}", True),
    ("Précision", f"{metrics['precision']:.1%}", "alertes justifiées"),
    ("Rappel", f"{metrics['recall']:.1%}", "départs détectés"),
    ("F1-score", f"{metrics['f1']:.3f}", f"seuil {metrics['threshold']}"),
])

ui.section("Champion vs baseline", "Le champion est choisi sur la validation croisée, pas sur le test.")
baseline = metrics.get("baseline_logistic_regression", {})
comparison = pd.DataFrame({
    "Modèle": [champion_label(metrics), "Régression logistique (baseline)"],
    "ROC-AUC CV": [metrics["roc_auc_cv"], baseline.get("roc_auc_cv")],
    "ROC-AUC test": [metrics["roc_auc_test"], baseline.get("roc_auc_test")],
})
st.dataframe(comparison, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("ROC-AUC CV", "ROC-AUC test")})
gain = metrics.get("gain_vs_baseline_test", 0)
ui.note(f"Écart de ROC-AUC test : <b>{gain:+.4f}</b>. "
        + ("Le champion n'apporte pas de gain significatif sur le test : la baseline, plus simple et plus "
           "interprétable, reste une alternative crédible en production."
           if gain <= 0.005 else "Le champion apporte un gain mesurable sur la baseline."))

ui.section("Diagnostics")
c1, c2 = st.columns(2)
with c1:
    ui.figure(figure_path("roc_curve.png"), "Courbe ROC")
with c2:
    ui.figure(figure_path("confusion_matrix.png"), "Matrice de confusion")
c1, c2 = st.columns(2)
with c1:
    ui.figure(figure_path("feature_importance.png"), "Importance des variables")
with c2:
    ui.figure(figure_path("baseline_comparison.png"), "Comparaison à la baseline")

with st.expander("Hyperparamètres et métriques brutes (metrics.json)"):
    st.json(metrics)
