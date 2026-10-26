# 📘 Guide pédagogique complet — du problème à l'application déployée

**Projet 15 — Pipeline Data Engineering + ML sur données e-commerce**

Ce document explique **tout** ce qui a été fait, dans l'ordre, avec des mots
simples et des analogies. Objectif : qu'une personne qui débute comprenne le
raisonnement du début à la fin, et que tu puisses **défendre chaque choix**
devant un jury.

---

## Table de traçabilité : chaque section du guide ↔ le code qui l'implémente

Chaque affirmation de ce guide correspond à du code **réel et exécuté**.
Voici la correspondance exacte :

| Section du guide | Fichier(s) | Ce qui est vérifiable |
|---|---|---|
| 4. Comprendre la donnée | `src/generate_sample_data.py`, `src/eda.py` | 4 figures EDA dans `reports/figures/eda_*.png` |
| 5. Ingestion (Extract) | `src/ingestion.py` (`acquire` + `extract`) | Téléchargement kagglehub + repli local, rapport qualité dans les logs |
| 6. Nettoyage (Transform) | `src/cleaning.py` | Chaque filtre compté dans les logs, sortie `data/interim/*.parquet` |
| 7. Définition du churn | `src/features.py` (snapshot T0, fenêtre 90 j) | Paramètres dans `config.yaml → churn` |
| 8. Feature engineering RFM | `src/features.py` | Sortie `data/processed/customer_features.parquet` |
| 9. Modèle ML (+ baseline) | `src/train.py` | Baseline LogReg + XGBoost + GridSearchCV, modèle `models/churn_model.joblib` |
| 10. Évaluation | `src/evaluate.py` | `reports/metrics.json` + 4 figures (ROC, confusion, importance, baseline) |
| 11. API + Dashboard | `api/main.py`, `dashboard/app.py` | `/docs` FastAPI, 3 onglets Streamlit |
| 12. Orchestration & MLOps | `pipelines/flow.py`, `config/config.yaml`, `tests/` | Flow Prefect à 6 tâches, 3 tests pytest, config centralisée |
| — Point d'entrée global | `run_pipeline.py` | Une commande = tout le pipeline (~8 s sur le jeu de démo) |

---

## Table des matières

1. [Le problème en une phrase](#1-le-problème-en-une-phrase)
2. [Pourquoi c'est un projet « riche »](#2-pourquoi-cest-un-projet-riche)
3. [Vue d'ensemble : la chaîne complète](#3-vue-densemble--la-chaîne-complète)
4. [Étape 1 — Comprendre la donnée](#4-étape-1--comprendre-la-donnée)
5. [Étape 2 — Ingestion (Extract)](#5-étape-2--ingestion-extract)
6. [Étape 3 — Nettoyage (Transform)](#6-étape-3--nettoyage-transform)
7. [Étape 4 — Le concept-clé : définir le churn sans tricher](#7-étape-4--le-concept-clé--définir-le-churn-sans-tricher)
8. [Étape 5 — Feature engineering (RFM)](#8-étape-5--feature-engineering-rfm)
9. [Étape 6 — Le modèle de Machine Learning](#9-étape-6--le-modèle-de-machine-learning)
10. [Étape 7 — Évaluer honnêtement](#10-étape-7--évaluer-honnêtement)
11. [Étape 8 — Servir le modèle (API + Dashboard)](#11-étape-8--servir-le-modèle-api--dashboard)
12. [Étape 9 — Orchestration & MLOps](#12-étape-9--orchestration--mlops)
13. [Les résultats et comment les lire](#13-les-résultats-et-comment-les-lire)
14. [Comment défendre ce projet devant un jury](#14-comment-défendre-ce-projet-devant-un-jury)
15. [Glossaire](#15-glossaire)

---

## 1. Le problème en une phrase

> **Certains clients arrêtent d'acheter et ne reviennent jamais. On veut les
> repérer *à l'avance*, pendant qu'il est encore temps de les retenir.**

C'est le problème du **churn** (« désabonnement » / « attrition »).

**Analogie.** Imagine une boutique de quartier à Dakar. Le gérant connaît ses
habitués : « Ça fait trois mois que je n'ai pas vu Fatou… d'habitude elle
passe chaque semaine. » Son cerveau vient de faire de la prédiction de churn.
Notre projet fait exactement ça, mais automatiquement, pour **30 000 clients**
à la fois, et **avant** que le client parte pour de bon.

Pourquoi c'est utile ? Parce qu'en marketing, **retenir un client existant
coûte 5 à 7 fois moins cher que d'en acquérir un nouveau**. Si on sait qui est
sur le point de partir, on peut lui envoyer une promo ciblée au bon moment.

---

## 2. Pourquoi c'est un projet « riche »

La plupart des projets de data science s'arrêtent au modèle : on entraîne, on
affiche une précision, fini. Ici, on couvre **toute la chaîne de valeur de la
donnée**, comme dans une vraie entreprise :

| Compétence | Où elle apparaît |
|---|---|
| **Data Engineering** | Pipeline ETL : ingestion, nettoyage, stockage en couches |
| **Data Science** | Feature engineering, modélisation, validation |
| **MLOps** | Config centralisée, orchestration, tests, reproductibilité |
| **Software / Déploiement** | API REST + dashboard interactif |

C'est ce spectre large qui rend le projet valorisant : tu ne montres pas
seulement que tu sais entraîner un modèle, mais que tu sais **construire un
système** autour.

---

## 3. Vue d'ensemble : la chaîne complète

```
 ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────────┐   ┌────────┐   ┌────────────┐
 │  CSV brut│──▶│ Ingestion │──▶│ Nettoyage│──▶│   Features   │──▶│ Modèle │──▶│ API + Dash │
 │  (ventes)│   │ (Extract) │   │(Transform)│  │ (RFM+churn)  │   │(XGBoost)│  │(FastAPI/St)│
 └──────────┘   └───────────┘   └──────────┘   └──────────────┘   └────────┘   └────────────┘
   data/raw       contrôle        Parquet          Parquet          .joblib      temps réel
                  qualité         propre           features         + métriques
```

**Le principe des couches** (dit *medallion*, ou bronze/argent/or) :

- **`data/raw`** (bronze) : la donnée brute, telle qu'on la reçoit. On n'y
  touche jamais.
- **`data/interim`** (argent) : la donnée nettoyée, fiable.
- **`data/processed`** (or) : la donnée prête pour le modèle (les features).

**Analogie.** C'est comme préparer un thiéboudiène. Le poisson et le riz bruts
(bronze), puis les ingrédients lavés et découpés (argent), puis le plat prêt à
cuire (or). On ne jette jamais les ingrédients bruts : si on se trompe dans la
découpe, on peut recommencer depuis le début.

---

## 4. Étape 1 — Comprendre la donnée

Le dataset **Online Retail** contient **une ligne par produit acheté**. Ses
colonnes :

| Colonne | Signification | Exemple |
|---|---|---|
| `InvoiceNo` | Numéro de facture | `536365` (ou `C536365` = annulation) |
| `StockCode` | Code produit | `85123A` |
| `Description` | Nom du produit | `WHITE HANGING HEART T-LIGHT HOLDER` |
| `Quantity` | Quantité achetée | `6` |
| `InvoiceDate` | Date/heure de la vente | `2010-12-01 08:26` |
| `UnitPrice` | Prix unitaire | `2.55` |
| `CustomerID` | Identifiant client | `17850` |
| `Country` | Pays du client | `United Kingdom` |

**Point important : plusieurs lignes = une facture.** Si Fatou achète 3
produits d'un coup, ça fait 3 lignes avec le même `InvoiceNo`. Il faudra s'en
souvenir pour compter correctement « le nombre de commandes ».

> 🔧 **Dans le projet** : l'acquisition des données est **automatique**. Au
> premier lancement, `src/ingestion.py` télécharge le vrai dataset via
> **kagglehub** (`kagglehub.dataset_download("carrie1/ecommerce-data")`) et le
> copie dans `data/raw/`. Si le téléchargement échoue (pas d'internet), le
> pipeline se **replie automatiquement** sur un fichier local — soit le vrai
> CSV placé à la main, soit le jeu synthétique de
> `src/generate_sample_data.py`, qui reproduit **exactement les mêmes colonnes
> et les mêmes défauts** (annulations, IDs manquants, prix nuls). C'est un
> signe de bonne conception : la source est configurable
> (`config.yaml → data_source.mode`) et le reste du code ne change jamais.

---

## 5. Étape 2 — Ingestion (Extract)

**Fichier : `src/ingestion.py`**

Rôle : **lire** le fichier et **constater** son état, sans rien corriger
encore.

Ce qu'on fait ici :
1. Lire le CSV (avec l'encodage `latin-1`, car le vrai jeu n'est pas en UTF-8).
2. **Typer** les colonnes : transformer les dates en vraies dates, les prix en
   nombres, etc. Une date lue comme du texte est inutilisable pour des calculs.
3. Produire un **rapport qualité** : combien de lignes ? quelle période ?
   combien de `CustomerID` manquants ? combien d'annulations ?

**Pourquoi séparer l'ingestion du nettoyage ?** Parce qu'on veut d'abord
**diagnostiquer** avant de **soigner**. Un médecin ausculte avant de prescrire.
Sur notre jeu, l'ingestion révèle ~25 % de lignes sans `CustomerID` et des
factures d'annulation : on sait maintenant ce qu'on devra traiter.

---

## 6. Étape 3 — Nettoyage (Transform)

**Fichier : `src/cleaning.py`**

On applique des **règles métier explicites**, et surtout **traçables** :
chaque suppression est comptée et affichée dans les logs. Jamais de
suppression silencieuse.

Les règles appliquées, et **pourquoi** :

| Règle | Raison |
|---|---|
| Retirer les dates invalides | On ne peut pas situer la vente dans le temps |
| Retirer les annulations (`InvoiceNo` commençant par `C`) | Ce ne sont pas des achats mais des retours |
| Retirer les lignes sans `CustomerID` | **Impossible de suivre un client anonyme dans le temps** → inexploitable pour le churn |
| Retirer `Quantity ≤ 0` et `UnitPrice ≤ 0` | Erreurs, dons, ajustements comptables |
| Retirer les doublons exacts | Bruit |

Puis on crée une colonne dérivée : **`Revenue = Quantity × UnitPrice`** (le
chiffre d'affaires de chaque ligne).

Enfin, on sauvegarde en **Parquet** plutôt qu'en CSV. Pourquoi ?
- **Plus léger** (compressé) et **plus rapide** à relire.
- **Il garde les types** : une date reste une date, pas besoin de re-typer.

**Analogie.** Le CSV, c'est écrire une recette sur une serviette en papier :
ça marche mais c'est fragile et lent à relire. Le Parquet, c'est la même
recette dans un classeur bien rangé, avec des onglets.

> Sur notre jeu : on passe d'environ **57 000 lignes brutes à ~42 000 lignes
> propres** (~74 % conservées). C'est normal : le nettoyage retire surtout les
> lignes sans client.

---

## 7. Étape 4 — Le concept-clé : définir le churn sans tricher

**C'est LE point qui distingue un projet sérieux d'un projet naïf. À bien
comprendre pour la soutenance.**

Problème : le dataset **n'a pas de colonne « churn »**. Personne n'a écrit
« ce client est parti ». Il faut la **fabriquer** nous-mêmes. Et c'est
piégeux.

### La mauvaise façon (à éviter, mais à savoir critiquer)

« Un client est churné si sa dernière commande date de plus de 90 jours. »
C'est un **seuil arbitraire**, et surtout ça mélange la cause et la
conséquence : on utiliserait la récence à la fois comme **réponse** et comme
**indice**. Le modèle « apprendrait » une évidence circulaire. On dit qu'il y
a **fuite de données** (*data leakage*).

### La bonne façon : le fenêtrage temporel

On coupe l'histoire en deux à une date qu'on appelle **snapshot** (T0) :

```
   |―――――――――― PASSÉ (on regarde) ――――――――――|  T0  |―― FUTUR (on vérifie) ――|
  début                                       snapshot                    T0 + 90 jours
   \_________________ on calcule les FEATURES ici       on calcule le LABEL ici _/
```

- **Features** = calculées **uniquement** sur le passé (avant T0).
- **Label churn** = on regarde le futur : **le client a-t-il racheté entre T0
  et T0+90 jours ?**
  - S'il **revient** → `churn = 0` (fidèle)
  - S'il **ne revient pas** → `churn = 1` (parti)

**Pourquoi c'est honnête ?** Parce qu'on se met exactement dans la situation
réelle : aujourd'hui (T0), avec ce que je sais du passé du client, je **parie**
sur son comportement futur. Le modèle ne voit jamais le futur pendant qu'il
apprend. **Zéro fuite.**

**Analogie.** C'est un examen à livre fermé. Les features, c'est ce que
l'élève a révisé (le passé). Le label, c'est sa note à l'examen (le futur).
On ne lui donne pas les réponses de l'examen pendant qu'il révise — sinon la
note ne veut rien dire.

> 🔧 **Dans le projet** (`src/features.py`) : le snapshot est calculé comme
> `date_max − 90 jours`. Sur notre jeu, cela donne un **taux de churn observé
> d'environ 45 %** — un jeu bien équilibré pour apprendre.

---

## 8. Étape 5 — Feature engineering (RFM)

**Fichier : `src/features.py`**

Le modèle ne comprend pas « des factures ». Il comprend des **nombres qui
décrivent un client**. On résume donc tout l'historique de chaque client en
quelques indicateurs. Le socle classique s'appelle **RFM** :

| Indicateur | Question | Calcul |
|---|---|---|
| **R**ecency (récence) | Il a acheté il y a combien de temps ? | jours entre le dernier achat et T0 |
| **F**requency (fréquence) | Il commande souvent ? | nombre de factures distinctes |
| **M**onetary (montant) | Il dépense combien ? | somme du chiffre d'affaires |

On ajoute des features complémentaires qui enrichissent le portrait :

- **`avg_basket`** : panier moyen (CA ÷ nombre de commandes)
- **`n_products`** : diversité (nombre de produits différents achetés)
- **`total_qty`** : quantité totale
- **`tenure`** : ancienneté (jours depuis le tout premier achat)
- **`avg_interpurchase`** : rythme moyen entre deux commandes
- **`country`** : pays (variable catégorielle)

**Intuition.** Un client qui a acheté hier (récence faible), souvent
(fréquence élevée), et pour beaucoup (montant élevé) est un bon client, peu
susceptible de partir. À l'inverse, quelqu'un vu une seule fois il y a 5 mois
est probablement déjà parti. Le modèle va **quantifier** ces intuitions.

**Analogie.** Le RFM, c'est la fiche client mentale du gérant de boutique :
« Celle-là passe souvent (F), dépense bien (M), et je l'ai vue récemment (R) —
cliente en or. »

---

## 9. Étape 6 — Le modèle de Machine Learning

**Fichier : `src/train.py`**

### a) Séparer apprentissage et test

On coupe les clients en deux paquets :
- **Train (75 %)** : le modèle apprend là-dessus.
- **Test (25 %)** : on le garde de côté pour l'évaluer sur des clients
  **jamais vus**.

La séparation est **stratifiée** : on garde le même taux de churn (~45 %) dans
les deux paquets, pour que le test soit représentatif.

**Analogie.** On révise sur les exercices du cours (train), mais on juge
l'élève sur un sujet d'examen inédit (test). Sinon il récite par cœur sans
comprendre : c'est le **surapprentissage** (*overfitting*).

### b) Le Pipeline scikit-learn

On enchaîne **préprocessing + modèle** dans un **seul objet** :

1. **Mise à l'échelle** des variables numériques (`StandardScaler`) : pour que
   « 500 € » et « 30 jours » soient comparables et qu'aucune variable n'écrase
   les autres juste parce que ses nombres sont plus grands.
2. **Encodage one-hot** du pays : transformer « France / UK / Germany » en
   colonnes 0/1, car un modèle ne mange que des nombres.
3. **Le modèle** : **XGBoost**, un algorithme d'arbres de décision *boostés*,
   très performant sur des données tabulaires comme les nôtres.

**Pourquoi un seul Pipeline ?** Parce que le même objet servira en production :
on lui donne un client brut, il fait *tout seul* la mise à l'échelle,
l'encodage, puis la prédiction. Et surtout, ça **empêche la fuite** : la mise
à l'échelle est apprise uniquement sur le train.

### c) Régler les hyperparamètres (validation croisée)

Un modèle a des **réglages** (profondeur des arbres, vitesse d'apprentissage…).
On teste plusieurs combinaisons avec la **validation croisée à 5 plis** :

> On découpe le train en 5 morceaux ; on entraîne sur 4 et on valide sur le
> 5ᵉ, en tournant 5 fois. On garde les réglages qui marchent le mieux **en
> moyenne**. C'est plus fiable que de tester une seule fois.

### d) Gérer le déséquilibre

Si une classe est rare, le modèle a tendance à l'ignorer. On corrige avec
`scale_pos_weight` (XGBoost) qui **donne plus de poids** à la classe minoritaire
pour forcer le modèle à y prêter attention.

---

## 10. Étape 7 — Évaluer honnêtement

**Fichier : `src/evaluate.py`**

On mesure la performance **sur le test** (clients jamais vus). Les métriques :

| Métrique | Ce qu'elle dit | Analogie |
|---|---|---|
| **ROC-AUC** | Capacité à **classer** un client à risque devant un client fidèle | Note globale sur 1.0 ; 0.5 = hasard, 1.0 = parfait |
| **Précision** | Parmi ceux que j'ai alertés, combien partaient vraiment ? | Évite de « crier au loup » pour rien |
| **Rappel** | Parmi ceux qui partaient, combien ai-je attrapés ? | Évite de **rater** des clients qui partent |
| **F1-score** | Équilibre entre précision et rappel | Compromis global |

**Précision vs rappel — le dilemme métier.** Si on alerte trop (rappel élevé),
on gaspille des promos sur des gens qui seraient restés. Si on alerte trop peu
(précision élevée), on laisse partir des clients. Le bon équilibre dépend du
**coût** d'une campagne vs. la **valeur** d'un client. C'est pour ça que le
**seuil d'alerte est un paramètre réglable** dans `config.yaml`
(`alert_threshold`), et non une valeur figée.

On génère 3 figures (dans `reports/figures/`) :
- **`roc_curve.png`** : la courbe ROC (plus elle « colle » au coin haut-gauche,
  mieux c'est).
- **`confusion_matrix.png`** : le tableau vrais/faux positifs/négatifs.
- **`feature_importance.png`** : quelles variables pèsent le plus.

---

## 11. Étape 8 — Servir le modèle (API + Dashboard)

Un modèle dans un fichier `.joblib` ne sert à personne. Il faut le **rendre
utilisable**.

### L'API (`api/main.py`, FastAPI)

Une **API REST** = une prise électrique standard : n'importe quel logiciel
peut « brancher » le modèle et lui envoyer un client pour obtenir une
prédiction, sans rien connaître de l'intérieur.

- `POST /predict` : on envoie les indicateurs d'un client, on reçoit
  `{ probabilité de churn, niveau de risque }`.
- `GET /health` : vérifie que le service tourne (utile en production).
- FastAPI génère **automatiquement une doc interactive** sur `/docs`.

### Le Dashboard (`dashboard/app.py`, Streamlit)

Une interface visuelle en 3 onglets, pour les **non-techniciens** (marketing,
direction) :
1. **Vue d'ensemble** : nombre de clients, taux de churn, distributions.
2. **Performance** : les métriques et les figures du modèle.
3. **Scorer un client** : des curseurs pour simuler un client et voir son
   risque en direct.

**Analogie.** L'API, c'est le moteur ; le dashboard, c'est le tableau de bord
de la voiture. Le directeur marketing ne veut pas ouvrir le capot, il veut
juste voir les voyants.

---

## 12. Étape 9 — Orchestration & MLOps

**Fichier : `pipelines/flow.py` (Prefect)**

Faire tourner les étapes à la main, une par une, ça va pour un TP. En
entreprise, on veut **automatiser** et **surveiller**. C'est le rôle d'un
**orchestrateur** (Prefect ou Airflow) :

- Il enchaîne les étapes dans le bon ordre.
- Il **relance** automatiquement une étape qui échoue (`retries`).
- Il **journalise** durées et erreurs, avec une interface visuelle.
- Il peut **planifier** l'exécution (ex. ré-entraîner chaque nuit).

**Analogie.** Sans orchestrateur, tu es le cuisinier qui surveille chaque
casserole. Avec, tu as un chef de cuisine qui lance les plats dans l'ordre,
te prévient si un plat brûle, et recommence au besoin.

Les autres briques **MLOps** présentes dans le projet :
- **Configuration centralisée** (`config.yaml`) : on change un paramètre sans
  toucher au code.
- **Logging** uniforme et horodaté partout.
- **Tests unitaires** (`pytest`) : on vérifie que le nettoyage retire bien les
  lignes invalides et que le modèle renvoie une probabilité valide.
- **Reproductibilité** : graine aléatoire unique (`random_state = 42`) →
  mêmes résultats à chaque exécution.
- **Stockage versionné** en Parquet.

---

## 13. Les résultats et comment les lire

Sur le jeu de démonstration :

| Métrique | Valeur | Lecture |
|---|---|---|
| ROC-AUC (validation croisée) | **0.895** | Stable pendant l'entraînement |
| ROC-AUC (test) | **0.925** | Excellente séparation churn / fidèle |
| Précision | **0.815** | 82 % des alertes sont justifiées |
| Rappel | **0.847** | On attrape 85 % des vrais départs |
| F1-score | **0.831** | Bon équilibre global |

Le fait que l'AUC **test (0.925) ≈ AUC validation (0.895)** est **rassurant** :
le modèle **généralise** bien, il n'a pas simplement mémorisé le train.

La variable la plus prédictive est la **récence** — ce qui confirme
l'intuition métier et le bon sens : *plus un client est resté silencieux
longtemps, plus il risque d'être parti*. Un modèle qui aurait donné un poids
énorme à une variable absurde aurait été suspect.

> ⚠️ **Honnêteté intellectuelle** (à assumer en soutenance) : ces chiffres
> sont obtenus sur un **jeu synthétique** conçu pour ressembler au réel. Sur le
> vrai Online Retail, on attend typiquement une AUC de l'ordre de **0.75–0.90**
> pour ce type de modèle RFM. La méthode, elle, est identique.

---

## 14. Comment défendre ce projet devant un jury

Quelques phrases-clés à avoir en tête :

**Sur la valeur métier**
> « Repérer un client sur le point de partir permet une action de rétention
> ciblée, 5 à 7 fois moins chère qu'une nouvelle acquisition. »

**Sur la rigueur (le point fort)**
> « Le churn n'était pas étiqueté dans les données. Je l'ai défini par
> fenêtrage temporel : features sur le passé, label sur le futur, ce qui évite
> toute fuite de données et reproduit les conditions réelles de production. »

**Sur l'architecture**
> « J'ai construit une architecture en couches — brut, intermédiaire, features
> — avec un pipeline scikit-learn unique qui garantit la cohérence entre
> l'entraînement et la production. »

**Sur le MLOps**
> « Le projet est piloté par configuration, testé, journalisé, reproductible,
> et orchestrable avec Prefect ; il est prêt à être industrialisé. »

**Questions pièges probables et réponses**

- *« Pourquoi pas juste un seuil de récence pour définir le churn ? »*
  → Parce que ce serait circulaire (la récence serait à la fois cause et
  effet) : fuite de données. Le fenêtrage temporel sépare proprement le passé
  (features) du futur (label).

- *« Pourquoi XGBoost et pas une régression logistique ? »*
  → La régression logistique **est implémentée comme baseline** dans
  `src/train.py` et comparée systématiquement (figure
  `baseline_comparison.png`). Résultat honnête sur le jeu de démo : la
  baseline fait **jeu égal** avec XGBoost (~0.93 vs ~0.93 d'AUC), car le
  signal synthétique est essentiellement linéaire (dominé par la récence).
  C'est exactement ce qu'une baseline sert à révéler : *si le modèle complexe
  ne bat pas le modèle simple, la complexité ne se justifie pas encore*. Sur
  le vrai jeu (interactions non-linéaires réelles), XGBoost prend
  généralement l'avantage — et si ce n'est pas le cas, la bonne décision
  d'ingénieur est de déployer le modèle simple, plus interprétable et plus
  léger.

- *« Comment choisir le seuil d'alerte ? »*
  → C'est une décision métier : arbitrage entre coût d'une campagne (faux
  positifs) et valeur d'un client perdu (faux négatifs). D'où un paramètre
  réglable, pas une valeur figée.

- *« Le modèle va-t-il rester bon dans le temps ? »*
  → Non, le comportement client dérive (*data drift*). D'où l'orchestration :
  on ré-entraîne périodiquement et on surveille les métriques.

---

## 15. Glossaire

| Terme | Définition simple |
|---|---|
| **Churn** | Un client qui cesse d'acheter / de s'abonner |
| **ETL** | Extract-Transform-Load : extraire, transformer, charger la donnée |
| **Feature** | Une variable descriptive donnée au modèle |
| **RFM** | Recency, Frequency, Monetary : les 3 indicateurs socles |
| **Label** | La réponse que le modèle doit apprendre à prédire (ici : churn 0/1) |
| **Data leakage** | Fuite : le modèle voit une info qu'il n'aurait pas en réalité |
| **Snapshot (T0)** | Date de coupure passé/futur pour définir le churn |
| **Overfitting** | Le modèle mémorise au lieu de généraliser |
| **Validation croisée** | Tester le modèle sur plusieurs découpes pour plus de fiabilité |
| **ROC-AUC** | Score de 0.5 (hasard) à 1.0 (parfait) mesurant la capacité à classer |
| **Précision / Rappel** | Justesse des alertes / capacité à ne pas rater de départs |
| **Parquet** | Format de fichier compressé et typé, rapide à relire |
| **Pipeline (sklearn)** | Objet qui enchaîne préprocessing + modèle |
| **API REST** | Interface standard pour interroger le modèle depuis un logiciel |
| **Orchestration** | Automatiser et surveiller l'enchaînement des étapes |
| **MLOps** | Les pratiques d'industrialisation d'un projet ML |
| **Data drift** | Le comportement réel change avec le temps, le modèle vieillit |

---

*Fin du guide. Pour l'utilisation pratique (commandes), voir `README.md`.*
