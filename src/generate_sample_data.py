"""
Générateur de données synthétiques fidèle au dataset 'Online Retail'
(Kaggle : carrie1/ecommerce-data).

Colonnes reproduites à l'identique :
  InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
  UnitPrice, CustomerID, Country

On reproduit aussi les DÉFAUTS du vrai jeu, car le pipeline doit savoir
les gérer :
  - CustomerID manquant sur ~25 % des lignes
  - factures d'annulation ('C' + Quantity négative)
  - quelques prix <= 0

>>> Sur la vraie donnée : remplace data/raw/online_retail.csv par le CSV
    Kaggle. Aucune autre ligne de code ne change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger, resolve

log = get_logger("generate_data")
RNG = np.random.default_rng(42)

N_CUSTOMERS = 3000
COUNTRIES = (
    ["United Kingdom"] * 85 + ["Germany"] * 4 + ["France"] * 4 +
    ["EIRE"] * 2 + ["Spain"] * 1 + ["Netherlands"] * 1 +
    ["Belgium"] * 1 + ["Portugal"] * 1
)
PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("47566", "PARTY BUNTING", 4.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("22197", "POPCORN HOLDER", 0.85),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95),
    ("23203", "JUMBO BAG VINTAGE DOILY", 2.08),
    ("22386", "JUMBO BAG PINK POLKADOT", 2.08),
]

START = pd.Timestamp("2010-12-01")
END = pd.Timestamp("2011-12-09")   # mêmes bornes que le vrai jeu


def _customer_profiles() -> pd.DataFrame:
    """Chaque client a une 'loyauté' latente qui pilote son comportement."""
    loyalty = RNG.beta(2, 2, N_CUSTOMERS)           # 0 = volatil, 1 = fidèle
    # Intervalle moyen entre deux achats (jours) : les fidèles reviennent vite.
    mean_gap = 12 + (1 - loyalty) * 70
    # Date du premier achat, étalée dans le temps
    first_day = RNG.integers(0, 250, N_CUSTOMERS)
    # Durée de vie du client : les fidèles restent actifs longtemps, les
    # volatils décrochent tôt. Après 'lifespan', le client n'achète plus.
    # C'est ce cycle de vie qui rend la récence prédictive du churn.
    lifespan = 30 + loyalty * 350 + RNG.normal(0, 25, N_CUSTOMERS)
    return pd.DataFrame({
        "CustomerID": np.arange(12346, 12346 + N_CUSTOMERS),
        "loyalty": loyalty,
        "mean_gap": mean_gap,
        "first_day": first_day,
        "last_active": first_day + lifespan.clip(min=15),
        "country": RNG.choice(COUNTRIES, N_CUSTOMERS),
    })


def generate() -> pd.DataFrame:
    profiles = _customer_profiles()
    rows = []
    invoice_counter = 536365

    for _, c in profiles.iterrows():
        day = c["first_day"]
        # On génère les factures tant que le client est actif (dans sa durée
        # de vie) et dans la période couverte par le jeu.
        while day < min(373, c["last_active"]):
            invoice_date = START + pd.Timedelta(days=int(day),
                                                hours=int(RNG.integers(7, 20)),
                                                minutes=int(RNG.integers(0, 60)))
            if invoice_date > END:
                break
            invoice_no = str(invoice_counter)
            invoice_counter += 1

            # Un panier = 1 à 6 lignes produits
            n_lines = RNG.integers(1, 7)
            chosen = RNG.choice(len(PRODUCTS), n_lines, replace=False)
            for idx in chosen:
                code, desc, base_price = PRODUCTS[idx]
                qty = int(RNG.integers(1, 25))
                price = round(base_price * RNG.uniform(0.9, 1.1), 2)
                cust = c["CustomerID"]
                # ~25 % des lignes sans CustomerID (comme le vrai jeu)
                if RNG.random() < 0.25:
                    cust = np.nan
                rows.append((invoice_no, code, desc, qty,
                             invoice_date, price, cust, c["country"]))

            # ~2 % de factures annulées ('C' + quantités négatives)
            if RNG.random() < 0.02:
                cinvoice = "C" + str(invoice_counter)
                invoice_counter += 1
                code, desc, base_price = PRODUCTS[RNG.integers(len(PRODUCTS))]
                rows.append((cinvoice, code, desc, -int(RNG.integers(1, 5)),
                             invoice_date + pd.Timedelta(hours=1),
                             round(base_price, 2), c["CustomerID"], c["country"]))

            # Prochain achat : loi Gamma (rythme régulier autour de mean_gap).
            gap = RNG.gamma(shape=4.0, scale=c["mean_gap"] / 4.0)
            day += max(1, gap)

    df = pd.DataFrame(rows, columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    ])
    # Quelques prix aberrants (0 ou négatifs), comme dans le réel
    bad = RNG.choice(len(df), size=int(len(df) * 0.003), replace=False)
    df.loc[bad, "UnitPrice"] = 0.0
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # mélange
    return df


if __name__ == "__main__":
    df = generate()
    out = resolve("data/raw/online_retail.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Jeu synthétique généré : %d lignes -> %s", len(df), out)
    log.info("Clients uniques : %d | Annulations : %d | CustomerID manquants : %d",
             df["CustomerID"].nunique(),
             df["InvoiceNo"].astype(str).str.startswith("C").sum(),
             df["CustomerID"].isna().sum())
