"""
Dashboard Streamlit — E-commerce Churn.

Point d'entrée : configuration, navigation et barre latérale.
Chaque page vit dans `dashboard/views/`, les composants visuels dans `ui.py`
et l'accès aux données (mis en cache) dans `loaders.py`.

Lancer :  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # rend ui / loaders importables par les pages

import ui  # noqa: E402
from loaders import cfg, champion_label, load_metrics  # noqa: E402
from src.config import resolve  # noqa: E402

st.set_page_config(page_title="Churn Analytics", page_icon=":material/insights:", layout="wide")
ui.apply_style()

VIEWS = Path(__file__).resolve().parent / "views"
navigation = st.navigation({
    "Projet": [
        st.Page(VIEWS / "overview.py", title="Vue d'ensemble", icon=":material/dashboard:", default=True),
    ],
    "Données": [
        st.Page(VIEWS / "data_layers.py", title="Catalogue des données", icon=":material/table_view:"),
        st.Page(VIEWS / "exploration.py", title="Analyse exploratoire", icon=":material/query_stats:"),
    ],
    "Modèle": [
        st.Page(VIEWS / "performance.py", title="Performance", icon=":material/monitoring:"),
        st.Page(VIEWS / "scoring.py", title="Scoring client", icon=":material/person_search:"),
    ],
    "Documentation": [
        st.Page(VIEWS / "methodology.py", title="Méthodologie", icon=":material/menu_book:"),
    ],
})

with st.sidebar:
    st.markdown("**Churn Analytics**  \nE-commerce · rétention client")
    metrics = load_metrics()
    metrics_file = resolve(cfg["paths"]["metrics"])
    if metrics:
        st.caption(ui.fr(f"Modèle : {champion_label(metrics)}  \n"
                   f"ROC-AUC test : {metrics.get('roc_auc_test', 0):.3f}  \n"
                   f"Évalué le : {datetime.fromtimestamp(metrics_file.stat().st_mtime):%d/%m/%Y %H:%M}"))
    else:
        st.caption("Modèle non entraîné : lancez `python run_pipeline.py`.")

navigation.run()
