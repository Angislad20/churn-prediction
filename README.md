# Prédiction du churn client — Banque européenne

Projet de data science de bout en bout : cadrage business, exploration des données, modélisation, évaluation, industrialisation (batch + Docker) et monitoring, sur un cas de prédiction de résiliation client bancaire.

## Sommaire

- [Contexte business](#contexte-business)
- [Données](#données)
- [Démarche](#démarche)
- [Résultats](#résultats)
- [Interprétabilité](#interprétabilité)
- [Structure du projet](#structure-du-projet)
- [Installation et utilisation](#installation-et-utilisation)
- [Limites et pistes d'amélioration](#limites-et-pistes-damélioration)

---

## Contexte business

Une banque européenne (clients en France, Allemagne, Espagne) souhaite identifier en amont les clients à risque de résiliation, pour permettre à l'équipe marketing/rétention d'agir avant le départ plutôt que de le subir.

**Décisions de cadrage :**

| Point | Décision | Justification |
|---|---|---|
| Utilisateur final | Équipe marketing/rétention | Action proactive plutôt que gestion réactive des résiliations |
| Mode de déploiement | Batch nocturne | Le churn bancaire se construit sur des semaines/mois, pas en temps réel — un score recalculé chaque nuit suffit et reste plus simple à industrialiser qu'une API |
| Métrique principale | Rappel (recall) | Un client perdu sans action (faux négatif) coûte plus cher qu'une relance envoyée à tort (faux positif) |
| Seuil de référence | 79,63 % d'accuracy | Performance d'un modèle "idiot" qui prédirait toujours "le client reste" — tout modèle retenu doit le dépasser nettement, et surtout être jugé sur le rappel, pas l'accuracy seule |

## Données

- **Source** : [Churn Modelling Dataset (Kaggle)](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Volume** : 10 000 clients
- **Cible** : `Exited` (1 = a quitté la banque, 0 = est resté) — 20,37 % de churn réel

| Variable | Description |
|---|---|
| `CreditScore` | Score de crédit du client |
| `Geography` | Pays de résidence (France / Allemagne / Espagne) |
| `Gender` | Genre |
| `Age` | Âge du client |
| `Tenure` | Ancienneté (années) |
| `Balance` | Solde du compte |
| `NumOfProducts` | Nombre de produits bancaires détenus |
| `HasCrCard` | Possède une carte de crédit (0/1) |
| `IsActiveMember` | Client actif (0/1) |
| `EstimatedSalary` | Salaire estimé |

## Démarche

### 1. Exploration des données (EDA)

**Variables exclues et pourquoi :**
- `RowNumber`, `CustomerId`, `Surname` — identifiants techniques sans valeur prédictive
- `Gender` — exclu pour raison éthique (risque de biais discriminatoire dans les décisions du modèle)

**Variables les plus prédictives identifiées :**
- `NumOfProducts` : relation fortement **non linéaire** avec le churn — 27,7 % (1 produit) → 7,6 % (2 produits) → 82,7 % (3 produits) → 100 % (4 produits). Les catégories 3 et 4 ont été regroupées ("3+") pour plus de robustesse statistique, le nombre de clients concernés étant faible.
- `Age` : progression régulière du taux de churn selon l'âge (7,6 % chez les plus jeunes → 45,9 % chez les plus âgés).
- `Geography` : effet concentré sur l'Allemagne (32,4 % de churn, contre ~16 % en France et en Espagne — France et Espagne ne se distinguent presque pas entre elles).
- `IsActiveMember` : les clients inactifs churnent deux fois plus (27 % vs 14 %).

**Variable dérivée testée et écartée** : `solde_nul` (solde bancaire à zéro). Un signal apparent fort (13,8 % vs 24,1 % de churn) s'est révélé être en grande partie un effet caché de `Geography` — aucun client allemand n'a de solde nul dans ce dataset, contre ~48 % en France/Espagne. Redondance confirmée, variable écartée du modèle final.

### 2. Préparation des données

- Split train/test 80/20, stratifié sur la cible
- Encodage One-Hot de `Geography` et de `NumOfProducts` (regroupé)
- Mise à l'échelle (`StandardScaler`) appliquée uniquement pour la régression logistique
- Sélection de variables différenciée selon le modèle : les variables faiblement prédictives en bivarié (`CreditScore`, `Tenure`, `EstimatedSalary`, `HasCrCard`) sont retirées pour la régression logistique, mais conservées pour les modèles à base d'arbres, capables de détecter des effets d'interaction

### 3. Modélisation

Deux modèles comparés :
- **Baseline** : régression logistique (`class_weight="balanced"`)
- **Modèle retenu** : Random Forest (`class_weight="balanced"`), avec ajustement du seuil de décision

Le seuil de classification par défaut (0,5) favorisait excessivement la précision au détriment du rappel. Après analyse du compromis précision-rappel, un **seuil de 0,3** a été retenu — meilleur rappel que la baseline, pour une perte de précision limitée.

## Résultats

| Modèle | Rappel (Part) | Précision (Part) | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Régression logistique (baseline) | 74 % | 44 % | 0,618 | 0,831 |
| Random Forest (seuil 0,5) | 60 % | 62 % | — | — |
| **Random Forest (seuil 0,3) — retenu** | **78 %** | **42 %** | **0,671** | **0,843** |

Le modèle retenu détecte 78 % des clients qui résilient réellement, contre 74 % pour la baseline, avec une précision comparable — un choix aligné sur la priorité business fixée en amont (minimiser les clients perdus sans action).

## Interprétabilité

Une analyse SHAP a été menée pour comprendre les prédictions individuelles du modèle, au-delà de sa performance globale.

**Variables les plus influentes** : `Age`, `NumOfProducts`, `IsActiveMember`, `Geography_Germany`.

**Distinction prédictif / actionnable** — un point clé pour l'équipe métier :

| Variable | Prédictive | Actionnable par le marketing |
|---|---|---|
| `Age`, `Geography` | Oui | Non — facteurs de risque non modifiables |
| `NumOfProducts`, `IsActiveMember` | Oui | Oui — offre de produit complémentaire, campagne de réengagement |

**Recommandation métier** : le modèle identifie *qui* est à risque (âge, pays), mais les leviers d'action concrets pour la rétention portent sur le nombre de produits détenus et l'engagement du client.

## Structure du projet

```
churn-prediction/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploration des données
│   ├── 02_preprocessing.ipynb    # Split, encodage, feature engineering
│   └── 03_modeling.ipynb         # Baseline, Random Forest, évaluation, SHAP
├── src/
│   ├── preprocessing.py          # Fonctions de transformation réutilisables
│   ├── scoring.py                # Logique de scoring batch
│   └── monitoring.py             # Détection de dérive des données
├── scripts/
│   └── run_batch_scoring.py      # Point d'entrée du scoring batch
├── tests/
│   └── test_preprocessing.py
├── models/                       # Modèle et encodeurs entraînés (.pkl)
├── data/
│   ├── raw/                      # Données brutes / nouveaux clients à scorer
│   └── processed/                # Données transformées
├── outputs/                      # Résultats de scoring, historique
├── Dockerfile
├── requirements.txt              # Environnement de développement complet
├── requirements-docker.txt       # Dépendances minimales pour le conteneur
└── README.md
```

## Installation et utilisation

### Environnement de développement (notebooks)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exécuter les notebooks dans l'ordre : `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_modeling.ipynb`.

### Scoring batch en local

```bash
python -m scripts.run_batch_scoring
```

Lit `data/raw/nouveaux_clients.csv`, produit `outputs/clients_a_risque.csv` et met à jour `outputs/historique_scoring.csv`.

### Scoring batch via Docker

```bash
docker build -t churn-scoring .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" churn-scoring
```

### Tests

```bash
python -m pytest tests/
```

## Limites et pistes d'amélioration

- **Coûts business non chiffrés précisément** : la priorité donnée au rappel repose sur un jugement qualitatif (faux négatif plus coûteux que faux positif), pas sur un chiffrage réel du coût d'une relance ou d'un client perdu. Un chiffrage métier affinerait le choix du seuil de décision.
- **Monitoring limité** : la détection de dérive actuelle compare les nouvelles données aux statistiques d'entraînement (moyennes, proportions) mais ne mesure pas la performance réelle du modèle dans le temps, faute de pouvoir observer le churn réel des clients scorés à court terme.
- **`EstimatedSalary`** présente une distribution suspicieusement uniforme, cohérente avec une variable simulée plutôt qu'observée — son absence de pouvoir prédictif dans ce dataset ne doit pas être généralisée à un contexte réel.
- **Variables 3+ produits** : le signal est très fort mais repose sur un faible nombre de clients ; à surveiller si le modèle est réappliqué sur une population différente.