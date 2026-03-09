# HR Attrition Predictor — Machine Learning Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.135.1-009688?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2.0-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-0.51.0-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-85.52%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.7715-blue?style=for-the-badge"/>
</p>

<p align="center">
  <strong>Microservice de prédiction intelligente du turnover RH basé sur XGBoost et SHAP Values</strong><br/>
  Développé dans le cadre d'un Projet de Fin d'Études (PFE) — Ooredoo Tunisie
</p>

---

## Table des Matières

- [Contexte et Problématique](#-contexte-et-problématique)
- [Architecture Générale](#-architecture-générale)
- [Dataset](#-dataset)
- [Pipeline Machine Learning](#-pipeline-machine-learning)
- [Feature Engineering](#-feature-engineering)
- [Modélisation](#-modélisation)
- [Résultats et Performances](#-résultats-et-performances)
- [Explicabilité — SHAP Values](#-explicabilité--shap-values)
- [API FastAPI](#-api-fastapi)
- [Structure du Projet](#-structure-du-projet)
- [Installation et Lancement](#-installation-et-lancement)
- [Endpoints](#-endpoints)
- [Exemples de Prédiction](#-exemples-de-prédiction)
- [Technologies Utilisées](#-technologies-utilisées)
- [Méthodologie](#-méthodologie)

---

## 🎯 Contexte et Problématique

Le turnover du personnel représente un enjeu stratégique majeur pour les entreprises de télécommunications. Chez **Ooredoo Tunisie**, comme dans l'ensemble du secteur, le coût de remplacement d'un employé est estimé entre **6 et 9 mois de salaire**, sans compter la perte de compétences et l'impact sur la productivité des équipes.

### Problématique

> Comment anticiper proactivement le risque de départ d'un employé afin de permettre aux équipes RH d'intervenir avant que la démission ne soit actée ?

### Objectifs

| Objectif | Description |
|----------|-------------|
| **Prédiction** | Estimer la probabilité de départ d'un employé sous forme de score de risque |
| **Explicabilité** | Identifier les facteurs déterminants du risque pour chaque employé |
| **Recommandations** | Générer automatiquement des actions préventives ciblées |
| **Intégration** | Exposer les prédictions via une API REST consommable par Spring Boot |

---

## 🏗️ Architecture Générale

```
┌─────────────────────────────────────────────────────────────┐
│                    Système HR Attrition                      │
│                                                             │
│  ┌──────────────┐    REST API     ┌──────────────────────┐  │
│  │  Spring Boot │ ──────────────► │   FastAPI ML Service │  │
│  │  (Port 8080) │ ◄────────────── │     (Port 8000)      │  │
│  └──────────────┘   JSON Response └──────────────────────┘  │
│         │                                    │               │
│         ▼                                    ▼               │
│  ┌──────────────┐                ┌──────────────────────┐   │
│  │  PostgreSQL  │                │   XGBoost + SHAP     │   │
│  │  (Port 5432) │                │   saved_model/*.pkl  │   │
│  └──────────────┘                └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

Ce dépôt contient exclusivement le **microservice ML** (FastAPI). Le backend Spring Boot est disponible dans un dépôt séparé.

---

## 📊 Dataset

### IBM HR Analytics Employee Attrition & Performance

| Caractéristique | Valeur |
|----------------|--------|
| Source | [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Nombre d'observations | 1 470 employés |
| Nombre de features initiales | 35 colonnes |
| Variable cible | `Attrition` (Yes/No) |
| Taux d'attrition | 16.1% (237 départs / 1233 restants) |
| Type de problème | Classification binaire déséquilibrée |

### Distribution de la Variable Cible

```
Attrition = No  (Reste)  : 1233 employés  ████████████████████  83.9%
Attrition = Yes (Quitte) :  237 employés  ████                  16.1%
```

### Colonnes Supprimées (non informatives)

| Colonne | Raison |
|---------|--------|
| `EmployeeCount` | Constante (valeur = 1 pour tous) |
| `EmployeeNumber` | Identifiant unique sans valeur prédictive |
| `Over18` | Constante (valeur = Y pour tous) |
| `StandardHours` | Constante (valeur = 80 pour tous) |

### Principaux Facteurs Corrélés à l'Attrition

| Feature | Corrélation | Direction |
|---------|-------------|-----------|
| OverTime | 0.246 | Positive |
| TotalWorkingYears | 0.171 | Négative |
| JobLevel | 0.169 | Négative |
| MaritalStatus | 0.162 | Variable |
| MonthlyIncome | 0.160 | Négative |

---

## 🔬 Pipeline Machine Learning

```
Raw Data (CSV)
     │
     ▼
┌─────────────┐
│  load_data  │  Chargement du dataset IBM HR
└─────────────┘
     │
     ▼
┌─────────────┐
│ clean_data  │  Suppression des colonnes constantes
└─────────────┘
     │
     ▼
┌──────────────────────┐
│ create_advanced_     │  Feature Engineering (37 features)
│ features             │
└──────────────────────┘
     │
     ▼
┌─────────────┐
│ encode_data │  LabelEncoder pour variables catégorielles
└─────────────┘
     │
     ▼
┌──────────────────────────────┐
│ Train / Val / Test Split     │  70% / 15% / 15%  (stratifié)
└──────────────────────────────┘
     │
     ▼
┌─────────────┐
│    SMOTE    │  Rééquilibrage des classes (863 / 863)
└─────────────┘
     │
     ▼
┌──────────────────────────────┐
│  GridSearchCV + StratifiedKFold │  256 combinaisons × 5 folds
│  scoring = MCC               │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Optimal Threshold Search    │  Minimise FP + FN
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  SHAP TreeExplainer          │  Explicabilité locale
└──────────────────────────────┘
     │
     ▼
  saved_model/*.pkl
```

---

## ⚙️ Feature Engineering

7 nouvelles features ont été créées à partir des features existantes :

| Feature Créée | Formule | Interprétation |
|--------------|---------|----------------|
| `satisfaction_score` | (JobSat + EnvSat + RelSat) / 3 | Score moyen de satisfaction globale |
| `promotion_rate` | YearsSinceLastPromotion / (YearsAtCompany + 1) | Taux de stagnation dans la carrière |
| `tenure_satisfaction` | YearsAtCompany × satisfaction_score | Engagement combiné ancienneté/satisfaction |
| `tenure_category` | Discrétisation de YearsAtCompany [0,1,3,5,10,40] | Catégorie d'ancienneté (0 à 4) |
| `job_hopping_score` | NumCompaniesWorked / (YearsAtCompany + 1) | Fréquence de changement d'entreprise |
| `overtime_x_joblevel` | OverTime × JobLevel | Interaction heures sup / niveau hiérarchique |
| `work_life_balance_score` | WorkLifeBalance × satisfaction_score | Score combiné équilibre / satisfaction |

**Nombre total de features après engineering : 37**

---

## 🤖 Modélisation

### Algorithme : XGBoost (eXtreme Gradient Boosting)

XGBoost a été sélectionné pour les raisons suivantes :

- Robustesse face aux données déséquilibrées via `scale_pos_weight`
- Gestion native des valeurs manquantes
- Performance supérieure sur les données tabulaires structurées
- Compatibilité native avec SHAP pour l'explicabilité
- Standard dans la littérature scientifique pour les problèmes RH

### Gestion du Déséquilibre des Classes

Deux stratégies complémentaires ont été appliquées :

1. **SMOTE** (Synthetic Minority Over-sampling Technique) sur l'ensemble d'entraînement
   - Avant SMOTE : 863 (Reste) / 167 (Quitte)
   - Après SMOTE : 863 (Reste) / 863 (Quitte)

2. **`scale_pos_weight = 3`** dans XGBoost pour pénaliser davantage les erreurs sur la classe minoritaire

### Optimisation des Hyperparamètres

**Métrique de scoring : Matthews Correlation Coefficient (MCC)**

Le MCC a été préféré à l'accuracy car il prend en compte les quatre valeurs de la matrice de confusion (TP, TN, FP, FN), ce qui le rend particulièrement adapté aux problèmes déséquilibrés.

```
MCC = (TP × TN − FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Meilleurs hyperparamètres trouvés :**

| Hyperparamètre | Valeur |
|----------------|--------|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `scale_pos_weight` | 3 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 1 |
| `gamma` | 0.1 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |

### Recherche du Seuil Optimal

Au lieu d'utiliser le seuil par défaut de 0.5, un seuil optimal a été recherché sur l'ensemble de validation en minimisant la fonction de coût suivante :

```
Score = (FP + FN) × (1 + 0.2 × |FP - FN| / (FP + FN))
```

**Seuil optimal retenu : 0.76**

---

## 📈 Résultats et Performances

### Métriques sur l'Ensemble de Test (221 observations)

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | **85.52%** |
| **AUC-ROC** | **0.7715** |
| **F1-Score (macro)** | 0.6437 |
| **MCC** | 0.3614 |

### Matrice de Confusion

```
                    Prédit : Reste    Prédit : Quitte
Réel : Reste              178               7
Réel : Quitte              25              11
```

| Indicateur | Valeur | Interprétation |
|------------|--------|----------------|
| Vrais Négatifs (TN) | 178 | Employés stables correctement identifiés |
| Faux Positifs (FP) | **7** | Fausses alertes (employés stables signalés à risque) |
| Faux Négatifs (FN) | 25 | Départs non détectés |
| Vrais Positifs (TP) | 11 | Départs correctement prédits |

### Rapport de Classification

```
              precision    recall  f1-score   support
Reste (0)        0.88      0.96      0.92       185
Quitte (1)       0.61      0.31      0.41        36
accuracy                             0.86       221
```

### Comparaison avec la Baseline

| Scénario | FP | FN | Total Erreurs |
|----------|----|----|---------------|
| Sans système (baseline) | 0 | 237 | 237 |
| Version initiale (local) | 27 | 28 | 55 |
| **Version optimisée (Colab GPU)** | **7** | **25** | **32** |

---

## 🔍 Explicabilité — SHAP Values

SHAP (SHapley Additive exPlanations) permet d'expliquer chaque prédiction individuellement en quantifiant la contribution de chaque feature à la décision du modèle.

### Top 10 Facteurs de Risque Globaux

| Rang | Feature | Importance SHAP | Interprétation |
|------|---------|-----------------|----------------|
| 1 | `StockOptionLevel` | 1.0945 | Absence d'options = fort risque |
| 2 | `overtime_x_joblevel` | 0.9493 | Heures sup × niveau poste |
| 3 | `job_hopping_score` | 0.7946 | Instabilité professionnelle |
| 4 | `JobSatisfaction` | 0.6843 | Insatisfaction au travail |
| 5 | `JobInvolvement` | 0.6366 | Faible implication |
| 6 | `MonthlyIncome` | 0.5923 | Rémunération insuffisante |
| 7 | `EnvironmentSatisfaction` | 0.5493 | Mauvais environnement de travail |
| 8 | `WorkLifeBalance` | 0.5009 | Déséquilibre vie pro/perso |
| 9 | `DistanceFromHome` | 0.4908 | Éloignement domicile/bureau |
| 10 | `Age` | 0.4528 | Jeunesse associée au risque |

### Exemple d'Interprétation Locale

Pour un employé avec une probabilité de départ de **99.99%** :

```
WorkLifeBalance        +1.1357  ──────────────────► AUGMENTE le risque
StockOptionLevel       +1.0910  ─────────────────► AUGMENTE le risque
JobSatisfaction        +0.8902  ───────────────► AUGMENTE le risque
BusinessTravel         +0.8811  ───────────────► AUGMENTE le risque
JobInvolvement         +0.6167  ──────────► AUGMENTE le risque
overtime_x_joblevel    -0.4966  ◄──────── DIMINUE le risque
```

---

## 🚀 API FastAPI

### Endpoints Disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Informations générales de l'API |
| `GET` | `/api/ml/health` | État de santé du modèle |
| `POST` | `/api/ml/predict` | Prédiction individuelle |
| `POST` | `/api/ml/predict/batch` | Prédiction par lot |
| `GET` | `/docs` | Documentation Swagger interactive |

### Niveaux de Risque

| Niveau | Condition | Action Recommandée |
|--------|-----------|-------------------|
| 🟢 **FAIBLE** | probabilité < 0.76 | Suivi standard |
| 🟡 **MOYEN** | 0.76 ≤ probabilité < 0.85 | Entretien préventif |
| 🔴 **ÉLEVÉ** | probabilité ≥ 0.85 | Intervention urgente |

---

## 📁 Structure du Projet

```
HRAttritionML/
│
├── app/                          # Application FastAPI
│   ├── __init__.py
│   ├── main.py                   # Point d'entrée FastAPI
│   ├── predictor.py              # Logique de prédiction + SHAP
│   ├── schemas.py                # Modèles Pydantic (requête/réponse)
│   └── routes/
│       ├── __init__.py
│       └── predict_router.py     # Endpoints de prédiction
│
├── model/                        # Pipeline ML
│   ├── __init__.py
│   ├── preprocess.py             # Chargement, nettoyage, encodage
│   ├── train_model.py            # Entraînement XGBoost + GridSearchCV
│   ├── evaluate_model.py         # Évaluation et métriques
│   └── explore_data.py           # Analyse exploratoire (EDA)
│
├── saved_model/                  # Artefacts ML (générés, non versionnés)
│   ├── attrition_model.pkl       # Modèle XGBoost entraîné
│   ├── shap_explainer.pkl        # Explainer SHAP
│   ├── label_encoders.pkl        # Encodeurs LabelEncoder
│   ├── feature_names.pkl         # Liste des features
│   ├── optimal_threshold.pkl     # Seuil optimal (0.76)
│   ├── metrics.pkl               # Métriques de performance
│   └── config.pkl                # Configuration du modèle
│
├── data/                         # Données (non versionnées)
│   └── hr_dataset.csv            # Dataset IBM HR Analytics
│
├── tests/                        # Tests unitaires
│   └── __init__.py
│
├── requirements.txt              # Dépendances Python
├── .gitignore
└── README.md
```

---

## 🛠️ Installation et Lancement

### Prérequis

- Python 3.11.9
- pip ou venv

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-compte/hr-attrition-ml.git
cd hr-attrition-ml
```

### 2. Créer et activer l'environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer les artefacts ML

Copier les fichiers `.pkl` dans le dossier `saved_model/` :

```
saved_model/
├── attrition_model.pkl
├── shap_explainer.pkl
├── label_encoders.pkl
├── feature_names.pkl
├── optimal_threshold.pkl
├── metrics.pkl
└── config.pkl
```

> **Note** : Les artefacts sont générés par `model/train_model.py` ou téléchargés depuis Google Colab (GPU T4).

### 5. Lancer l'API

```bash
uvicorn app.main:app --reload --port 8000
```

L'API est accessible sur : `http://127.0.0.1:8000`  
Documentation Swagger : `http://127.0.0.1:8000/docs`

### 6. (Optionnel) Réentraîner le modèle

```bash
python -m model.train_model
```

---

## 📡 Endpoints

### GET `/api/ml/health`

**Réponse :**

```json
{
    "status": "OK",
    "model_version": "2.0-optimized",
    "threshold": 0.76,
    "accuracy": 0.8552,
    "auc_roc": 0.7715,
    "fp": 7,
    "fn": 25
}
```

### POST `/api/ml/predict`

**Corps de la requête :**

```json
{
    "Age": 28,
    "Gender": "Male",
    "MaritalStatus": "Single",
    "Department": "Sales",
    "JobRole": "Sales Executive",
    "JobLevel": 1,
    "BusinessTravel": "Travel_Frequently",
    "EducationField": "Marketing",
    "Education": 3,
    "MonthlyIncome": 2500,
    "PercentSalaryHike": 11,
    "StockOptionLevel": 0,
    "DailyRate": 400,
    "HourlyRate": 50,
    "MonthlyRate": 8000,
    "JobSatisfaction": 1,
    "EnvironmentSatisfaction": 1,
    "RelationshipSatisfaction": 2,
    "JobInvolvement": 2,
    "WorkLifeBalance": 1,
    "PerformanceRating": 3,
    "OverTime": "Yes",
    "DistanceFromHome": 25,
    "NumCompaniesWorked": 6,
    "TotalWorkingYears": 5,
    "YearsAtCompany": 1,
    "YearsInCurrentRole": 1,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 1,
    "TrainingTimesLastYear": 2
}
```

**Réponse :**

```json
{
    "probability": 0.9999,
    "risk_level": "ÉLEVÉ",
    "threshold_used": 0.76,
    "top_risk_factors": [
        {
            "feature": "WorkLifeBalance",
            "shap_value": 1.1357,
            "impact": "AUGMENTE"
        }
    ],
    "recommendation_factors": [
        "⚖️  Mettre en place des mesures d'équilibre vie pro/perso",
        "💰 Proposer des options d'actions ou avantages financiers",
        "😊 Planifier un entretien de satisfaction et améliorer les conditions de travail"
    ]
}
```

---

## 📦 Technologies Utilisées

| Catégorie | Technologie | Version |
|-----------|-------------|---------|
| **Langage** | Python | 3.11.9 |
| **API Framework** | FastAPI | 0.135.1 |
| **Serveur ASGI** | Uvicorn | 0.41.0 |
| **Validation** | Pydantic | 2.12.5 |
| **ML Algorithm** | XGBoost | 3.2.0 |
| **Explicabilité** | SHAP | 0.51.0 |
| **Rééquilibrage** | imbalanced-learn | 0.14.1 |
| **ML Utilitaires** | scikit-learn | 1.8.0 |
| **Manipulation données** | Pandas | 3.0.1 |
| **Calcul numérique** | NumPy | 2.4.2 |
| **Sérialisation** | Joblib | 1.5.3 |
| **Visualisation** | Matplotlib / Seaborn | 3.10.8 / 0.13.2 |
| **Entraînement GPU** | Google Colab (T4) | — |

---

## 📐 Méthodologie

Ce projet a été développé selon la méthodologie **SCRUM Agile** avec les sprints suivants :

| Release | Contenu | Statut |
|---------|---------|--------|
| Release 1 | Authentification JWT + Gestion utilisateurs | ✅ Terminé |
| Release 2 | Pipeline ML + FastAPI + Entités RH Spring Boot | 🔄 En cours |
| Release 3 | Dashboard statistiques + Historique scores | ⏳ Planifié |
| Release 4 | Frontend Angular + Rapports PDF | ⏳ Planifié |

---

## 👥 Acteurs du Système

| Acteur | Rôle |
|--------|------|
| **Administrateur** | Gestion des utilisateurs et configuration système |
| **Responsable RH** | Consultation des scores de risque et validation des alertes |
| **Manager** | Consultation du risque de ses collaborateurs directs |

---

## 📄 Licence

Ce projet est développé dans le cadre d'un Projet de Fin d'Études (PFE) à **Ooredoo Tunisie**.  
Tous droits réservés © 2026.

---

<p align="center">
  Développé avec ❤️ pour Ooredoo Tunisie — PFE 2026
</p>
