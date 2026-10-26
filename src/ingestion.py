"""
ÉTAPE E (Extract) — Acquisition + ingestion des données brutes.

Deux sources possibles (config: data_source.mode) :
  - "kagglehub" : télécharge le VRAI dataset Kaggle (carrie1/ecommerce-data)
                  et le copie dans data/raw. Repli automatique sur le fichier
                  local si le téléchargement échoue (pas de réseau, etc.).
  - "local"     : lit directement data/raw/online_retail.csv.

Rôle : obtenir le fichier, typer correctement les colonnes, et faire un
contrôle qualité minimal. On ne nettoie RIEN ici : on charge fidèlement
et on constate l'état des données.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config import get_logger, load_config, resolve

log = get_logger("ingestion")

EXPECTED_COLS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def acquire(cfg: dict) -> Path:
    """Obtient le CSV brut selon la source configurée et renvoie son chemin."""
    raw_path = resolve(cfg["paths"]["raw"])
    mode = cfg.get("data_source", {}).get("mode", "local")

    if mode == "kagglehub":
        dataset = cfg["data_source"]["kaggle_dataset"]
        try:
            import kagglehub
            log.info("Téléchargement du dataset Kaggle : %s ...", dataset)
            dl_dir = Path(kagglehub.dataset_download(dataset))
            log.info("Dataset téléchargé dans : %s", dl_dir)
            # Le dataset contient un unique CSV ('data.csv') : on prend le
            # premier CSV trouvé et on le copie dans notre couche raw.
            csvs = sorted(dl_dir.rglob("*.csv"))
            if not csvs:
                raise FileNotFoundError(f"Aucun CSV trouvé dans {dl_dir}")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(csvs[0], raw_path)
            log.info("Copié %s -> %s", csvs[0].name, raw_path)
        except Exception as e:
            log.warning("kagglehub indisponible (%s).", e)
            if raw_path.exists():
                log.warning("Repli sur le fichier local : %s", raw_path)
            else:
                raise RuntimeError(
                    "Téléchargement Kaggle impossible et aucun fichier local. "
                    "Solutions : (1) vérifier la connexion internet, ou "
                    "(2) générer un jeu de démo : python -m src.generate_sample_data, "
                    "ou (3) placer le CSV manuellement dans data/raw/."
                ) from e
    return raw_path


def extract(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = acquire(cfg)
    log.info("Lecture du fichier brut : %s", path)

    # encoding latin-1 : le vrai jeu Online Retail n'est pas en UTF-8
    df = pd.read_csv(path, encoding="latin-1")

    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la source : {missing}")

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    # Le vrai CSV Kaggle utilise le format américain '12/1/2010 8:26' ;
    # notre jeu de démo utilise l'ISO. On tente les deux, sans ambiguïté.
    dates = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M",
                           errors="coerce")
    if dates.isna().mean() > 0.5:            # pas le format Kaggle -> ISO/auto
        dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["InvoiceDate"] = dates
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # --- Rapport qualité à l'ingestion ---
    log.info("Lignes             : %d", len(df))
    log.info("Période            : %s -> %s",
             df["InvoiceDate"].min(), df["InvoiceDate"].max())
    log.info("Clients uniques    : %d", df["CustomerID"].nunique())
    log.info("CustomerID manquant: %d (%.1f%%)",
             df["CustomerID"].isna().sum(),
             100 * df["CustomerID"].isna().mean())
    log.info("Annulations ('C')  : %d",
             df["InvoiceNo"].str.startswith("C").sum())
    return df


if __name__ == "__main__":
    extract()
