import streamlit as st

import ui
from loaders import LAYERS, cfg, column_profile, layer_stats, load_layer

ui.page_header("Données", "Catalogue des couches de données",
               "Chaque étape du pipeline produit une couche persistée. Comparez les volumes, "
               "inspectez le schéma et filtrez les enregistrements.")

stats = {name: layer_stats(name) for name in LAYERS}
ui.section("Lignage", "Volumétrie de chaque couche, de la source au jeu d'entraînement.")
ui.flow([(label, ui.fmt_int(stats[k]["rows"]),
          f"lignes · {stats[k]['cols']} colonnes · {stats[k]['size_mb']:.1f} Mo {stats[k]['format']}")
         for k, (label, _) in LAYERS.items()])

ui.section("Couche à examiner")
layer = st.segmented_control("Couche", list(LAYERS), format_func=lambda k: LAYERS[k][0],
                             default="raw", key="data_layer", label_visibility="collapsed") or "raw"
data = load_layer(layer)
s = stats[layer]
st.caption(f"{LAYERS[layer][1]}  ·  `{cfg['paths'][layer]}`")

overview, schema, explorer = st.tabs(["Aperçu", "Schéma", "Explorateur"])

with overview:
    ui.kpi_row([
        ("Lignes", ui.fmt_int(s["rows"]), "", True),
        ("Colonnes", s["cols"]),
        ("Taille", f"{s['size_mb']:.1f} Mo", s["format"]),
        ("Cellules vides", f"{s['missing']:.1%}", "valeurs manquantes"),
    ])
    st.write("")
    st.markdown("**Échantillon (10 premières lignes)**")
    st.dataframe(data.head(10), hide_index=True, width="stretch")
    st.markdown("**Statistiques descriptives (colonnes numériques)**")
    st.dataframe(data.describe(include="number").T.round(2), width="stretch")

with schema:
    st.dataframe(column_profile(layer), hide_index=True, width="stretch",
                 column_config={"Manquants (%)": st.column_config.ProgressColumn(
                     "Manquants (%)", min_value=0, max_value=100, format="%.1f %%")})

with explorer:
    view = data
    with st.container(border=True):
        f1, f2 = st.columns(2)
        country_col = "Country" if "Country" in data else "country"
        countries = f1.multiselect("Pays", sorted(data[country_col].dropna().unique()), key=f"pays_{layer}")
        if countries:
            view = view[view[country_col].isin(countries)]
        if "Description" in data:
            search = f2.text_input("Produit (recherche)", key=f"search_{layer}")
            if search:
                view = view[view["Description"].str.contains(search, case=False, na=False)]
        elif "churn" in data:
            status = f2.selectbox("Statut", ["Tous", "Churn (1)", "Fidèle (0)"], key=f"churn_{layer}")
            if status != "Tous":
                view = view[view["churn"] == (1 if status.startswith("Churn") else 0)]
    st.caption(f"{ui.fmt_int(len(view))} lignes correspondent aux filtres · 500 premières affichées")
    st.dataframe(view.head(500), hide_index=True, width="stretch")
    st.download_button("Exporter la sélection (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name=f"{layer}_selection.csv", mime="text/csv", icon=":material/download:")
