"""
ÉTAPE Analyse exploratoire (EDA) — Comprendre les ventes et les clients.

Répond à l'exigence du projet : « Analyse des ventes et comportement client ».
Produit 4 figures dans reports/figures/ :

  1. eda_monthly_revenue.png   : chiffre d'affaires mensuel (tendance)
  2. eda_top_products.png      : top 10 des produits par CA
  3. eda_rfm_distributions.png : distributions Recency / Frequency / Monetary
  4. eda_recency_by_churn.png  : la récence sépare-t-elle churn et actifs ?

Ces figures se lisent AVANT la modélisation : elles justifient les choix
(le n°4, en particulier, montre visuellement pourquoi le RFM est pertinent).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_logger, load_config, resolve

log = get_logger("eda")
plt.rcParams["figure.dpi"] = 120
BLUE, RED = "#2b6cb0", "#c53030"


def run_eda(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    figdir = resolve(cfg["paths"]["figures"])
    figdir.mkdir(parents=True, exist_ok=True)

    tx = pd.read_parquet(resolve(cfg["paths"]["interim"]))
    feats = pd.read_parquet(resolve(cfg["paths"]["features"]))

    # 1. CA mensuel ----------------------------------------------------------
    monthly = (tx.set_index("InvoiceDate")["Revenue"]
                 .resample("ME").sum())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    monthly.plot(ax=ax, marker="o", color=BLUE)
    ax.set_title("Chiffre d'affaires mensuel")
    ax.set_xlabel(""); ax.set_ylabel("CA")
    fig.tight_layout(); fig.savefig(figdir / "eda_monthly_revenue.png"); plt.close(fig)

    # 2. Top produits --------------------------------------------------------
    top = (tx.groupby("Description")["Revenue"].sum()
             .sort_values(ascending=True).tail(10))
    fig, ax = plt.subplots(figsize=(7, 4))
    top.plot.barh(ax=ax, color=BLUE)
    ax.set_title("Top 10 produits par chiffre d'affaires")
    ax.set_xlabel("CA")
    fig.tight_layout(); fig.savefig(figdir / "eda_top_products.png"); plt.close(fig)

    # 3. Distributions RFM ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, col, title in zip(
            axes, ["recency", "frequency", "monetary"],
            ["Récence (jours)", "Fréquence (commandes)", "Monetary (CA)"]):
        feats[col].clip(upper=feats[col].quantile(0.99)).hist(
            bins=30, ax=ax, color=BLUE, alpha=0.85)
        ax.set_title(title); ax.grid(alpha=0.3)
    fig.suptitle("Distributions RFM des clients", y=1.02)
    fig.tight_layout(); fig.savefig(figdir / "eda_rfm_distributions.png",
                                    bbox_inches="tight"); plt.close(fig)

    # 4. Récence par statut churn — LA figure qui justifie le modèle ---------
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, color, name in [(0, BLUE, "Actif"), (1, RED, "Churn")]:
        sub = feats.loc[feats["churn"] == label, "recency"]
        sub.clip(upper=feats["recency"].quantile(0.99)).hist(
            bins=30, ax=ax, alpha=0.6, color=color, label=name)
    ax.set_title("Récence selon le statut : les churners ont décroché plus tôt")
    ax.set_xlabel("Jours depuis le dernier achat (avant T0)")
    ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "eda_recency_by_churn.png"); plt.close(fig)

    log.info("4 figures EDA écrites dans %s", figdir)
    log.info("CA total : %.0f | Commandes : %d | Clients : %d",
             tx["Revenue"].sum(), tx["InvoiceNo"].nunique(),
             tx["CustomerID"].nunique())


if __name__ == "__main__":
    run_eda()
