"""
ÉTAPE T (Transform) — Nettoyage des transactions.

On applique des règles métier explicites et TRAÇABLES : chaque filtre est
loggé avec le nombre de lignes supprimées. Aucune suppression silencieuse.
Résultat sauvegardé en Parquet (couche 'interim').
"""
from __future__ import annotations

import pandas as pd

from src.config import get_logger, load_config, resolve
from src.ingestion import extract

log = get_logger("cleaning")


def _drop(df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    n = int(mask.sum())
    if n:
        log.info("  - %-42s : %6d lignes retirées", reason, n)
    return df.loc[~mask]


def clean(df: pd.DataFrame | None = None, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    rules = cfg["cleaning"]
    if df is None:
        df = extract(cfg)

    n0 = len(df)
    log.info("Nettoyage — %d lignes en entrée", n0)

    # 1. Dates invalides
    df = _drop(df, df["InvoiceDate"].isna(), "date invalide")

    # 2. Annulations (factures 'C')
    if rules["drop_cancellations"]:
        df = _drop(df, df["InvoiceNo"].str.startswith("C"), "annulations (facture C)")

    # 3. Transactions sans client (inexploitables pour le churn)
    if rules["drop_missing_customer"]:
        df = _drop(df, df["CustomerID"].isna(), "CustomerID manquant")

    # 4. Quantités et prix non valides
    df = _drop(df, df["Quantity"] < rules["min_quantity"], "quantité <= 0")
    df = _drop(df, df["UnitPrice"] <= rules["min_unit_price"], "prix <= 0")

    # 5. Doublons stricts
    dup = df.duplicated()
    df = _drop(df, dup, "doublons exacts")

    # 6. Types finaux + colonne dérivée 'Revenue'
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    log.info("Nettoyage terminé : %d -> %d lignes (%.1f%% conservées)",
             n0, len(df), 100 * len(df) / n0)

    out = resolve(cfg["paths"]["interim"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Transactions propres sauvegardées : %s", out)
    return df


if __name__ == "__main__":
    clean()
