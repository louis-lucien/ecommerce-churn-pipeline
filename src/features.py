"""
ÉTAPE Feature Engineering + Définition du churn.

Problème : le jeu Online Retail n'a PAS de colonne 'churn'. Il faut la
construire honnêtement, sans tricher (sans fuite de données).

Méthode = fenêtrage temporel :

   |------------- fenêtre d'observation (features) -----------|snapshot|-- horizon --|
   début                                                    T0                    T0+H

  - snapshot T0 = date_max - horizon
  - FEATURES : calculées UNIQUEMENT sur les transactions AVANT T0
  - LABEL churn = 1 si le client N'ACHÈTE PAS entre T0 et T0+H, sinon 0

Ainsi, on prédit le futur à partir du passé, exactement comme en production.
Les features n'utilisent jamais d'information postérieure au snapshot :
pas de fuite (data leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger, load_config, resolve

log = get_logger("features")


def build_features(df: pd.DataFrame | None = None,
                   cfg: dict | None = None,
                   save: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    if df is None:
        df = pd.read_parquet(resolve(cfg["paths"]["interim"]))

    horizon = cfg["churn"]["observation_horizon_days"]
    # Le snapshot T0 se positionne à `snapshot_offset_days` avant la fin du jeu.
    # Séparer offset et horizon permet d'observer le retour du client sur une
    # fenêtre [T0, T0+horizon] qui peut être plus courte que la marge laissée
    # en fin de jeu (offset >= horizon). Par défaut offset == horizon.
    offset = cfg["churn"].get("snapshot_offset_days", horizon)
    date_max = df["InvoiceDate"].max().normalize()
    snapshot = date_max - pd.Timedelta(days=offset)
    obs_end = snapshot + pd.Timedelta(days=horizon)
    log.info("Date max du jeu : %s", date_max.date())
    log.info("Snapshot T0     : %s  (offset = %d j, horizon = %d j)",
             snapshot.date(), offset, horizon)

    past = df[df["InvoiceDate"] < snapshot].copy()
    future = df[(df["InvoiceDate"] >= snapshot) & (df["InvoiceDate"] < obs_end)].copy()
    log.info("Transactions passées (features) : %d | fenêtre d'observation : %d",
             len(past), len(future))

    # --- Features RFM sur la fenêtre passée --------------------------------
    grp = past.groupby("CustomerID")
    feats = pd.DataFrame({
        # Recency : jours depuis le dernier achat (avant T0). Plus c'est
        # grand, plus le client est "froid".
        "recency": (snapshot - grp["InvoiceDate"].max()).dt.days,
        # Frequency : nombre de factures distinctes
        "frequency": grp["InvoiceNo"].nunique(),
        # Monetary : chiffre d'affaires total généré
        "monetary": grp["Revenue"].sum(),
        # Panier moyen
        "avg_basket": grp["Revenue"].sum() / grp["InvoiceNo"].nunique(),
        # Diversité produit
        "n_products": grp["StockCode"].nunique(),
        # Quantité totale achetée
        "total_qty": grp["Quantity"].sum(),
        # Ancienneté du client (jours entre 1er achat et snapshot)
        "tenure": (snapshot - grp["InvoiceDate"].min()).dt.days,
    })

    # Intervalle moyen RÉEL entre deux commandes successives : moyenne des
    # écarts (en jours) entre dates de factures distinctes d'un même client.
    # Pour un client à une seule commande, l'intervalle est indéfini -> on
    # retombe sur l'ancienneté (borne haute, cohérente avec "revient rarement").
    def _mean_gap(dates: pd.Series) -> float:
        d = dates.drop_duplicates().sort_values()
        if len(d) < 2:
            return float("nan")
        return d.diff().dropna().dt.days.mean()

    gaps = grp["InvoiceDate"].agg(_mean_gap)
    feats["avg_interpurchase"] = gaps.fillna(feats["tenure"])
    feats["country"] = grp["Country"].agg(lambda s: s.mode().iloc[0])

    # --- Label churn -------------------------------------------------------
    returning = set(future["CustomerID"].unique())
    feats["churn"] = (~feats.index.isin(returning)).astype(int)

    feats = feats.reset_index()

    rate = feats["churn"].mean()
    log.info("Clients modélisés : %d | Taux de churn observé : %.1f%%",
             len(feats), 100 * rate)

    if save:
        out = resolve(cfg["paths"]["features"])
        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(out, index=False)
        log.info("Table de features sauvegardée : %s", out)
    return feats


if __name__ == "__main__":
    build_features()
