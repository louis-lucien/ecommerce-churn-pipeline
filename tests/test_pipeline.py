"""
Tests unitaires du pipeline (pytest).

    pytest -q

On vérifie les invariants critiques SANS dépendre d'un run préalable :
chaque test construit lui-même ses données en mémoire (pas de lecture de
Parquet ni de modèle sur disque). Le cœur du projet — la définition du churn
sans fuite temporelle — est testé par construction.
"""
import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean
from src.config import load_config
from src.features import build_features


@pytest.fixture
def raw_sample():
    """Petit jeu synthétique avec des lignes volontairement invalides."""
    return pd.DataFrame({
        "InvoiceNo": ["1001", "C1002", "1003", "1004", "1005"],
        "StockCode": ["A", "B", "C", "D", "E"],
        "Description": ["x"] * 5,
        "Quantity": [5, 3, -2, 4, 10],           # -2 invalide
        "InvoiceDate": pd.to_datetime([
            "2011-01-01", "2011-01-02", "2011-01-03",
            "2011-01-04", "2011-01-05"]),
        "UnitPrice": [2.0, 3.0, 1.0, 0.0, 5.0],   # 0.0 invalide
        "CustomerID": [111, 222, 333, 444, np.nan],  # NaN invalide
        "Country": ["United Kingdom"] * 5,
    })


def test_cleaning_removes_invalid(raw_sample):
    cfg = load_config()
    out = clean(raw_sample.copy(), cfg)
    # Annulation, quantité<0, prix=0 et CustomerID manquant retirés -> 1 ligne
    assert (out["Quantity"] > 0).all()
    assert (out["UnitPrice"] > 0).all()
    assert out["CustomerID"].notna().all()
    assert not out["InvoiceNo"].str.startswith("C").any()


def _clean_transactions():
    """Jeu de transactions PROPRES, déterministe. On raisonne en jours
    relatifs à une date d'ancrage finale pour maîtriser exactement où tombe
    le snapshot (date_max - horizon). Horizon par défaut = 90 j.

    date_max = 2011-12-31 (achat d'ancrage du client 3).
    snapshot  = 2011-12-31 - 90 j = 2011-10-02.
    obs_end   = snapshot + 90 j   = 2011-12-31.
      - client 1 : achats avant T0 ET dans [T0, obs_end)   -> churn = 0
      - client 2 : achats uniquement avant T0              -> churn = 1
      - client 3 : achats uniquement APRÈS obs_end         -> absent du modèle
                   (son achat d'ancrage sert juste à fixer date_max)
    """
    rows = [
        # (CustomerID, date, InvoiceNo)
        (1, "2011-06-01", "A1"), (1, "2011-07-01", "A2"), (1, "2011-11-01", "A3"),
        (2, "2011-05-10", "B1"), (2, "2011-06-10", "B2"),
        (3, "2011-12-31", "C1"),
    ]
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceDate", "InvoiceNo"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = "SKU"
    df["Quantity"] = 1
    df["Revenue"] = 10.0
    df["Country"] = "United Kingdom"
    return df


def test_churn_label_is_leakage_free():
    """Invariant central : les features ne voient QUE le passé, le label
    ne regarde QUE le futur. Testé sur un jeu dont on connaît la vérité."""
    cfg = load_config()
    feats = build_features(_clean_transactions(), cfg,
                           save=False).set_index("CustomerID")

    # Le client 3 (actif uniquement hors fenêtre passée) n'est PAS modélisé :
    # on ne construit aucune feature à partir du futur.
    assert 3 not in feats.index
    # Client 1 revient dans [T0, obs_end) -> non churné ; client 2 ne revient
    # pas -> churné.
    assert feats.loc[1, "churn"] == 0
    assert feats.loc[2, "churn"] == 1
    # recency du client 2 = snapshot(2011-10-02) - dernier achat passé
    # (2011-06-10) = 114 j, calculé sur le passé uniquement.
    assert feats.loc[2, "recency"] == 114
    assert (feats["recency"] >= 0).all() and (feats["tenure"] >= 0).all()


def test_no_future_transaction_leaks_into_features():
    """Ajouter des achats APRÈS le snapshot ne doit changer AUCUNE feature
    d'un client (seul son label churn peut basculer)."""
    cfg = load_config()
    base = _clean_transactions()
    feats_a = build_features(base, cfg, save=False).set_index("CustomerID")

    # On injecte une grosse commande du client 2 DANS la fenêtre d'observation
    # [T0, obs_end) (le 2011-11-15) : son retour est désormais observé.
    extra = base.iloc[[0]].copy()
    extra["CustomerID"] = 2
    extra["InvoiceDate"] = pd.to_datetime("2011-11-15")
    extra["InvoiceNo"] = "B99"
    extra["Revenue"] = 9999.0
    feats_b = build_features(pd.concat([base, extra], ignore_index=True),
                             cfg, save=False).set_index("CustomerID")

    cols = ["recency", "frequency", "monetary", "total_qty", "tenure"]
    # Les features (calculées sur le passé) sont INCHANGÉES...
    pd.testing.assert_series_equal(feats_a.loc[2, cols], feats_b.loc[2, cols])
    # ...mais le label bascule : le client 2 revient maintenant après T0.
    assert feats_a.loc[2, "churn"] == 1 and feats_b.loc[2, "churn"] == 0


def test_prediction_range():
    """Smoke test d'inférence. Nécessite un modèle entraîné : ignoré
    proprement s'il n'existe pas encore (test d'intégration)."""
    from pathlib import Path
    from src.config import resolve
    if not Path(resolve(load_config()["paths"]["model"])).exists():
        pytest.skip("Modèle non entraîné (lancer run_pipeline.py d'abord)")
    from src.predict import predict_one
    demo = {"recency": 30, "frequency": 8, "monetary": 600.0,
            "avg_basket": 75.0, "n_products": 12, "total_qty": 200,
            "tenure": 300, "avg_interpurchase": 40.0, "country": "United Kingdom"}
    res = predict_one(demo)
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["churn_prediction"] in (0, 1)
