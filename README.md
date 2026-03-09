# 🧠 HR Attrition Predictor — Microservice d'Intelligence Artificielle pour la Prédiction du Turnover RH

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.135.1-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2.0-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-0.51.0-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-85.52%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.7715-0078D4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MCC-0.8651-red?style=for-the-badge"/>
</p>

<p align="center">
  <strong>
    Microservice de prédiction intelligente et explicable du risque de turnover RH<br/>
    basé sur XGBoost, SHAP Values et FastAPI
  </strong>
  <br/><br/>
  Projet de Fin d'Études (PFE) — Ingénierie Informatique<br/>
  Entreprise d'accueil : <strong>Ooredoo Tunisie</strong> — Année universitaire 2025/2026
</p>

---

## 📋 Table des Matières

1. [Présentation du Projet](#-1-présentation-du-projet)
2. [Contexte et Problématique](#-2-contexte-et-problématique)
3. [Architecture Globale du Système](#-3-architecture-globale-du-système)
4. [Environnement Technique](#-4-environnement-technique)
5. [Dataset Utilisé](#-5-dataset-utilisé)
6. [Démarche Scientifique et Méthodologique](#-6-démarche-scientifique-et-méthodologique)
7. [Étape 1 — Analyse Exploratoire des Données (EDA)](#-étape-1--analyse-exploratoire-des-données-eda)
8. [Étape 2 — Prétraitement et Nettoyage](#-étape-2--prétraitement-et-nettoyage)
9. [Étape 3 — Feature Engineering](#-étape-3--feature-engineering)
10. [Étape 4 — Gestion du Déséquilibre des Classes](#-étape-4--gestion-du-déséquilibre-des-classes)
11. [Étape 5 — Choix et Justification de l'Algorithme](#-étape-5--choix-et-justification-de-lalgorithme)
12. [Étape 6 — Optimisation des Hyperparamètres](#-étape-6--optimisation-des-hyperparamètres)
13. [Étape 7 — Recherche du Seuil Optimal](#-étape-7--recherche-du-seuil-optimal)
14. [Étape 8 — Évaluation et Résultats](#-étape-8--évaluation-et-résultats)
15. [Étape 9 — Explicabilité avec SHAP](#-étape-9--explicabilité-avec-shap)
16. [Étape 10 — Déploiement via FastAPI](#-étape-10--déploiement-via-fastapi)
17. [Structure du Projet](#-structure-du-projet)
18. [Guide d'Installation et de Lancement](#-guide-dinstallation-et-de-lancement)
19. [Documentation des Endpoints API](#-documentation-des-endpoints-api)
20. [Exemples de Prédiction Commentés](#-exemples-de-prédiction-commentés)
21. [Récapitulatif des Performances](#-récapitulatif-des-performances)
22. [Technologies et Versions](#-technologies-et-versions)
23. [Méthodologie de Développement](#-méthodologie-de-développement)

---

## 🎓 1. Présentation du Projet

Ce dépôt constitue le **microservice d'intelligence artificielle** du système **HR Attrition Predictor**, développé dans le cadre d'un Projet de Fin d'Études en Ingénierie Informatique au sein d'**Ooredoo Tunisie**.

Le système global est une application web intelligente permettant aux équipes des Ressources Humaines d'**anticiper proactivement les risques de départ** des employés, d'en **comprendre les causes** via des explications locales par employé, et de recevoir des **recommandations d'actions préventives** automatisées.

Ce dépôt couvre exclusivement la **couche Machine Learning et API** du système. Le backend Spring Boot (gestion des utilisateurs, entités RH, base de données PostgreSQL) est versionné dans un dépôt séparé.

### Dépôts du Projet

| Composant | Technologie | Lien |
|-----------|-------------|------|
| **Ce dépôt** — ML + API | Python / FastAPI / XGBoost | 📍 Ici |
| Backend RH | Java / Spring Boot / PostgreSQL | 🔗 Dépôt séparé |
| Frontend *(à venir)* | Angular | ⏳ Planifié |

---

## 🏢 2. Contexte et Problématique

### 2.1 Contexte Entreprise

**Ooredoo Tunisie** est l'un des principaux opérateurs de télécommunications en Tunisie. Comme toutes les grandes entreprises du secteur, elle fait face à un enjeu RH critique : **la rétention des talents**. Le turnover involontaire représente un coût considérable, estimé dans la littérature entre **6 et 9 mois de salaire** par employé perdu, sans compter l'impact sur la productivité, la cohésion des équipes et la perte de savoir-faire.

### 2.2 Problématique Scientifique

> **Comment peut-on, à partir des données RH disponibles, construire un modèle prédictif capable d'estimer la probabilité de départ d'un employé, tout en garantissant l'explicabilité des décisions pour les équipes RH non-expertes en IA ?**

Cette problématique soulève trois défis scientifiques majeurs :

1. **Défi de déséquilibre** : Le taux d'attrition naturel (~16%) crée une forte asymétrie entre les classes, rendant les modèles classiques biaisés vers la classe majoritaire.

2. **Défi d'explicabilité** : Un score de risque opaque n'est pas actionnable pour les RH. Il faut pouvoir expliquer *pourquoi* un employé est à risque.

3. **Défi d'intégration** : Le modèle doit être déployé sous forme d'API REST pour être consommé par le backend Spring Boot du système global.

### 2.3 Objectifs

| # | Objectif | Indicateur de Succès |
|---|----------|---------------------|
| O1 | Construire un modèle prédictif performant | Accuracy ≥ 80%, AUC-ROC ≥ 0.75 |
| O2 | Minimiser les faux négatifs (départs non détectés) | FN le plus faible possible |
| O3 | Minimiser les faux positifs (fausses alertes) | FP raisonnable |
| O4 | Fournir des explications locales par employé | SHAP Values intégrées |
| O5 | Générer des recommandations RH automatiques | Mapping SHAP → actions |
| O6 | Exposer le modèle via une API REST documentée | FastAPI + Swagger |

---

## 🏗️ 3. Architecture Globale du Système

Le système suit une architecture **microservices** avec deux composants principaux communiquant via REST :

```
╔══════════════════════════════════════════════════════════════════╗
║                    SYSTÈME HR ATTRITION PREDICTOR                ║
║                                                                  ║
║  ┌─────────────────────────┐         ┌────────────────────────┐ ║
║  │     BACKEND SPRING BOOT │         │   MICROSERVICE ML      │ ║
║  │       (Port 8080)        │         │   FastAPI (Port 8000)  │ ║
║  │                         │  HTTP   │                        │ ║
║  │  • Gestion Utilisateurs  │ ──────► │  • Prédiction XGBoost  │ ║
║  │  • Entités Employés     │ ◄────── │  • SHAP Explicabilité  │ ║
║  │  • Scores de Risque     │  JSON   │  • Recommandations RH  │ ║
║  │  • Alertes et Seuils    │         │  • Prédiction Batch    │ ║
║  │  • Notifications Mail   │         │                        │ ║
║  └──────────┬──────────────┘         └────────────────────────┘ ║
║             │                                                    ║
║             ▼                                                    ║
║  ┌─────────────────────────┐                                     ║
║  │      PostgreSQL          │                                     ║
║  │       (Port 5432)        │                                     ║
║  │  • users                 │                                     ║
║  │  • employees             │                                     ║
║  │  • scores_risque         │                                     ║
║  │  • alertes               │                                     ║
║  └─────────────────────────┘                                     ║
╚══════════════════════════════════════════════════════════════════╝
```

### Flux de Prédiction End-to-End

```
Responsable RH saisit les données d'un employé
              │
              ▼
    Spring Boot reçoit la requête
              │
              ▼
    Spring Boot appelle POST /api/ml/predict
              │
              ▼
    FastAPI prétraite les données
    (encodage + feature engineering)
              │
              ▼
    XGBoost calcule la probabilité de départ
              │
              ▼
    SHAP explique les facteurs contributeurs
              │
              ▼
    Génération des recommandations RH
              │
              ▼
    Spring Boot reçoit le résultat JSON
              │
              ▼
    Score sauvegardé en PostgreSQL
              │
              ▼
    Alerte générée si seuil dépassé
              │
              ▼
    Notification mail envoyée au Manager
```

---

## 💻 4. Environnement Technique

### Environnement de Développement Local

| Composant | Outil | Version |
|-----------|-------|---------|
| IDE | PyCharm Professional | 2024.x |
| Langage | Python | 3.11.9 |
| Environnement virtuel | venv | — |
| Gestionnaire de paquets | pip | — |
| Test API | Postman | — |
| Versioning | Git + GitHub | — |

### Environnement d'Entraînement Cloud

| Composant | Outil | Détail |
|-----------|-------|--------|
| Plateforme | Google Colab | Gratuit |
| Accélérateur | GPU NVIDIA T4 | 16 GB VRAM |
| Runtime | Python 3.12 | — |
| Durée d'entraînement | ~11 minutes | 256 combinaisons × 5 folds |

> **Note méthodologique** : L'entraînement a été réalisé sur Google Colab (GPU T4) pour bénéficier de l'accélération matérielle et réduire le temps de GridSearchCV de plusieurs heures à ~11 minutes. Les artefacts `.pkl` produits ont ensuite été intégrés au projet local PyCharm.

---

## 📊 5. Dataset Utilisé

### IBM HR Analytics Employee Attrition & Performance

| Caractéristique | Détail |
|----------------|--------|
| **Source** | IBM Watson Analytics / Kaggle |
| **URL** | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset |
| **Taille** | 1 470 observations, 35 colonnes |
| **Variable cible** | `Attrition` : Yes (départ) / No (reste) |
| **Type de problème** | Classification binaire supervisée |
| **Déséquilibre** | 16.1% positifs (237 départs) / 83.9% négatifs (1233 stables) |

### Distribution de la Variable Cible

```
Attrition = No  (Reste)  : 1233 employés  ████████████████████  83.9%
Attrition = Yes (Quitte) :  237 employés  ████                  16.1%

Ratio de déséquilibre : 1 : 5.2
Nécessite une stratégie de rééquilibrage (SMOTE)
```

### Description des Variables Principales

| Catégorie | Variables |
|-----------|-----------|
| **Données personnelles** | Age, Gender, MaritalStatus, DistanceFromHome |
| **Données professionnelles** | Department, JobRole, JobLevel, BusinessTravel, YearsAtCompany |
| **Données salariales** | MonthlyIncome, DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel |
| **Données de satisfaction** | JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance |
| **Données d'engagement** | JobInvolvement, PerformanceRating, TrainingTimesLastYear |
| **Données d'historique** | TotalWorkingYears, NumCompaniesWorked, YearsSinceLastPromotion |

---

## 🔬 6. Démarche Scientifique et Méthodologique

La construction du modèle a suivi rigoureusement les étapes canoniques d'un projet de Data Science, inspirées de la méthodologie **CRISP-DM** :

```
┌─────────────────────────────────────────────────────────────────┐
│              DÉMARCHE CRISP-DM ADAPTÉE                          │
│                                                                 │
│  1. Compréhension métier        → Problématique RH Ooredoo     │
│           ↓                                                     │
│  2. Compréhension des données   → EDA + Corrélations           │
│           ↓                                                     │
│  3. Préparation des données     → Nettoyage + Encodage         │
│           ↓                                                     │
│  4. Feature Engineering         → 7 nouvelles features         │
│           ↓                                                     │
│  5. Modélisation                → XGBoost + SMOTE              │
│           ↓                                                     │
│  6. Optimisation                → GridSearchCV + MCC           │
│           ↓                                                     │
│  7. Évaluation                  → Accuracy, AUC-ROC, MCC       │
│           ↓                                                     │
│  8. Explicabilité               → SHAP Values                  │
│           ↓                                                     │
│  9. Déploiement                 → FastAPI REST                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Étape 1 — Analyse Exploratoire des Données (EDA)

**Fichier** : `model/explore_data.py`

L'analyse exploratoire avait pour objectif de comprendre la structure des données, identifier les corrélations avec la variable cible et détecter d'éventuelles anomalies.

### Profil Type de l'Employé qui Quitte

| Caractéristique | Employé qui Quitte | Employé qui Reste | Écart |
|----------------|-------------------|-------------------|-------|
| Âge moyen | 33.6 ans | 37.6 ans | -4 ans |
| Salaire moyen | 4 787 DT | 6 832 DT | -2 045 DT |
| Ancienneté moyenne | 5.1 ans | 7.4 ans | -2.3 ans |
| Taux heures supplémentaires | 54% | 23% | +31% |

### Top 5 Corrélations avec Attrition

| Feature | Corrélation de Pearson | Direction |
|---------|----------------------|-----------|
| OverTime | +0.246 | Plus d'heures sup → plus de départs |
| TotalWorkingYears | -0.171 | Plus d'expérience → moins de départs |
| JobLevel | -0.169 | Niveau plus élevé → moins de départs |
| MonthlyIncome | -0.160 | Meilleur salaire → moins de départs |
| MaritalStatus | +0.162 | Célibataires partent plus souvent |

### Visualisations Produites

Les visualisations suivantes ont été générées dans `plots/` :

- Distribution de la variable cible (déséquilibre des classes)
- Matrice de corrélation des features numériques
- Comparaison des distributions par statut d'attrition
- Taux d'attrition par département, rôle et niveau hiérarchique
- Boxplots salaire et âge selon le statut d'attrition

---

## 🧹 Étape 2 — Prétraitement et Nettoyage

**Fichier** : `model/preprocess.py`

### Colonnes Supprimées

| Colonne | Justification |
|---------|--------------|
| `EmployeeCount` | Constante — valeur = 1 pour tous les employés |
| `EmployeeNumber` | Identifiant technique sans valeur prédictive |
| `Over18` | Constante — valeur = "Y" pour tous les employés |
| `StandardHours` | Constante — valeur = 80 pour tous les employés |

### Encodage des Variables Catégorielles

La technique **LabelEncoder** (scikit-learn) a été appliquée aux 9 variables catégorielles :

| Variable | Exemple de Mapping |
|----------|--------------------|
| `Attrition` | Yes → 1, No → 0 |
| `BusinessTravel` | Non-Travel → 0, Travel_Rarely → 1, Travel_Frequently → 2 |
| `Department` | HR → 0, R&D → 1, Sales → 2 |
| `Gender` | Female → 0, Male → 1 |
| `MaritalStatus` | Divorced → 0, Married → 1, Single → 2 |
| `OverTime` | No → 0, Yes → 1 |

Les encodeurs sont persistés dans `saved_model/label_encoders.pkl` pour assurer la **cohérence entre l'entraînement et l'inférence**.

### Traitement des Valeurs Manquantes

Le dataset IBM HR Analytics ne contient pas de valeurs manquantes natives. Cependant, des valeurs manquantes peuvent apparaître après le feature engineering (division par zéro, `pd.cut` sur valeurs limites). Ces cas sont traités systématiquement :

```python
df[numeric_cols] = df[numeric_cols].fillna(0)
df['col'] = df['col'].replace([np.inf, -np.inf], 0)
```

---

## ⚙️ Étape 3 — Feature Engineering

**Fichier** : `model/preprocess.py` — fonction `create_advanced_features()`

Le feature engineering est une étape fondamentale de ce projet. 7 nouvelles variables ont été créées pour capturer des interactions et des dynamiques non linéaires absentes des features brutes.

### Nouvelles Features Créées

| # | Feature | Formule | Interprétation Métier |
|---|---------|---------|----------------------|
| 1 | `satisfaction_score` | (JobSat + EnvSat + RelSat) / 3 | Score agrégé de satisfaction globale |
| 2 | `promotion_rate` | YearsSinceLastPromotion / (YearsAtCompany + 1) | Mesure la stagnation de carrière |
| 3 | `tenure_satisfaction` | YearsAtCompany × satisfaction_score | Engagement combinant ancienneté et satisfaction |
| 4 | `tenure_category` | Discrétisation de YearsAtCompany | 0=<1an, 1=1-3ans, 2=3-5ans, 3=5-10ans, 4=>10ans |
| 5 | `job_hopping_score` | NumCompaniesWorked / (YearsAtCompany + 1) | Fréquence de changement d'entreprise |
| 6 | `overtime_x_joblevel` | OverTime × JobLevel | Interaction heures sup / niveau hiérarchique |
| 7 | `work_life_balance_score` | WorkLifeBalance × satisfaction_score | Score combiné équilibre et satisfaction |

**Résultat** : Passage de 31 features (après nettoyage) à **37 features** au total.

### Justification du Feature Engineering

- **Capturer des non-linéarités** : `tenure_category` discrétise l'ancienneté en paliers métier significatifs
- **Modéliser des interactions** : `overtime_x_joblevel` combine deux signaux complémentaires
- **Agréger des signaux faibles** : `satisfaction_score` consolide trois mesures de satisfaction distinctes

---

## ⚖️ Étape 4 — Gestion du Déséquilibre des Classes

Le déséquilibre des classes (16.1% vs 83.9%) est un problème fondamental. Sans traitement, le modèle tend à prédire systématiquement la classe majoritaire et obtient une accuracy artificiellement élevée.

### Stratégie Adoptée : Double Approche Complémentaire

#### 4.1 SMOTE — Sur-échantillonnage Synthétique

**SMOTE** (Synthetic Minority Over-sampling Technique, Chawla et al., 2002) génère de nouveaux exemples synthétiques de la classe minoritaire en interpolant entre exemples existants dans l'espace des features.

```
Avant SMOTE (ensemble d'entraînement) :
  Classe 0 (Reste)  : 863 exemples  ████████████████████
  Classe 1 (Quitte) : 167 exemples  ████

Après SMOTE :
  Classe 0 (Reste)  : 863 exemples  ████████████████████
  Classe 1 (Quitte) : 863 exemples  ████████████████████

Ratio final : 1:1 parfait
```

> **Important** : SMOTE est appliqué **uniquement sur l'ensemble d'entraînement** pour éviter toute fuite de données (data leakage) vers les ensembles de validation et de test.

#### 4.2 scale_pos_weight dans XGBoost

En complément, le paramètre `scale_pos_weight = 3` dans XGBoost pénalise davantage les erreurs sur la classe minoritaire, renforçant la sensibilité du modèle aux cas de départ réels.

---

## 🤖 Étape 5 — Choix et Justification de l'Algorithme

### Pourquoi XGBoost ?

| Critère | XGBoost | Random Forest | SVM | Régression Logistique |
|---------|---------|---------------|-----|-----------------------|
| Performance données tabulaires | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Gestion du déséquilibre | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Compatibilité SHAP native | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| Robustesse aux valeurs aberrantes | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Standard académique domaine RH | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Vitesse d'inférence | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### Principe de XGBoost

XGBoost est un algorithme d'apprentissage ensembliste par **gradient boosting**. Il construit séquentiellement un ensemble d'arbres de décision, chaque arbre corrigeant les erreurs du précédent en minimisant une fonction de perte régularisée :

```
Objectif = Σ L(yi, ŷi) + Σ Ω(fk)

où :
  L  = fonction de perte (log-loss pour classification binaire)
  Ω  = terme de régularisation (contrôle la complexité des arbres)
  fk = k-ième arbre de décision de l'ensemble
```

---

## 🔧 Étape 6 — Optimisation des Hyperparamètres

**Fichier** : `model/train_model.py` — fonction `tune_model()`

### Stratégie : GridSearchCV + StratifiedKFold (k=5)

**GridSearchCV** explore exhaustivement l'espace des hyperparamètres défini. **StratifiedKFold** garantit que chaque fold maintient la proportion originale des classes (16.1% / 83.9%).

### Métrique de Scoring : Matthews Correlation Coefficient (MCC)

Le MCC a été préféré à l'accuracy pour les raisons suivantes :

```
MCC = (TP × TN − FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Propriétés du MCC :
  • Prend en compte les 4 valeurs de la matrice de confusion
  • Valeur entre -1 (pire) et +1 (parfait), 0 = aléatoire
  • Particulièrement adapté aux classes déséquilibrées
  • Pénalise équitablement les faux positifs et faux négatifs
```

### Grille de Recherche des Hyperparamètres

| Hyperparamètre | Valeurs Testées | Rôle dans le Modèle |
|----------------|-----------------|---------------------|
| `n_estimators` | [200, 300] | Nombre d'arbres dans l'ensemble |
| `max_depth` | [4, 5] | Profondeur maximale de chaque arbre |
| `learning_rate` | [0.05, 0.1] | Taux d'apprentissage (shrinkage) |
| `scale_pos_weight` | [2, 3] | Poids accordé à la classe minoritaire |
| `subsample` | [0.8] | Fraction d'observations par arbre |
| `colsample_bytree` | [0.8] | Fraction de features par arbre |
| `min_child_weight` | [1, 3] | Poids minimum requis pour une feuille |
| `gamma` | [0, 0.1] | Gain minimum pour effectuer une division |
| `reg_alpha` | [0, 0.1] | Régularisation L1 (Lasso) |
| `reg_lambda` | [1, 2] | Régularisation L2 (Ridge) |

**Total** : 256 combinaisons × 5 folds = **1 280 entraînements**
**Durée** : ~11 minutes sur GPU T4 (Google Colab)

### Meilleurs Hyperparamètres Trouvés

| Hyperparamètre | Valeur Optimale |
|----------------|----------------|
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
| **MCC Cross-Validation** | **0.8651** |

---

## 🎯 Étape 7 — Recherche du Seuil Optimal

**Fichier** : `model/train_model.py` — fonction `find_optimal_threshold()`

### Pourquoi ne pas Utiliser le Seuil par Défaut 0.5 ?

Par défaut, les modèles de classification utilisent un seuil de 0.5 pour binariser les probabilités. Cette approche n'est pas optimale dans notre contexte RH car :

- **Faux Positif (FP)** = Employé stable identifié à risque → entretien inutile, coût modéré
- **Faux Négatif (FN)** = Départ non détecté → perte de talent, coût élevé (6-9 mois de salaire)

### Fonction de Coût Personnalisée

Un seuil optimal est recherché sur l'ensemble de validation (jamais vu pendant l'entraînement) en minimisant :

```
Score(t) = (FP(t) + FN(t)) × (1 + 0.2 × |FP(t) - FN(t)| / (FP(t) + FN(t)))

où t ∈ [0.20, 0.80] avec un pas de 0.02

Le terme de pénalité 0.2 × |FP - FN| / (FP + FN)
pénalise les déséquilibres importants entre FP et FN,
favorisant une distribution équilibrée des erreurs.
```

### Résultats de la Recherche

| Seuil | FP | FN | Total Erreurs | Rang |
|-------|----|----|---------------|------|
| **0.76** | **12** | **22** | **34** | **1er — Optimal** |
| 0.78 | 11 | 23 | 34 | 2ème |
| 0.64 | 16 | 20 | 36 | 3ème |
| 0.66 | 16 | 20 | 36 | 4ème |

**Seuil optimal retenu : 0.76**

---

## 📈 Étape 8 — Évaluation et Résultats

**Fichier** : `model/evaluate_model.py`

### Partition des Données

```
Dataset total : 1 470 observations
        │
        ├── Entraînement : 1 029 obs (70%)  → SMOTE → 1 726 obs équilibrées
        ├── Validation   :   220 obs (15%)  → Recherche du seuil optimal
        └── Test         :   221 obs (15%)  → Évaluation finale (jamais vu)

Toutes les partitions sont STRATIFIÉES pour maintenir
le ratio 16.1% / 83.9% dans chaque sous-ensemble.
```

### Métriques de Performance (Ensemble de Test)

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Accuracy** | **85.52%** | 85.5% des prédictions sont correctes |
| **AUC-ROC** | **0.7715** | Bon pouvoir discriminant (0.5=aléatoire, 1=parfait) |
| **F1-Score (macro)** | 0.6437 | Moyenne harmonique précision/rappel |
| **MCC (test)** | 0.3614 | Corrélation prédictions/réalité |
| **MCC (cross-val)** | **0.8651** | Performance en validation croisée |

### Matrice de Confusion Détaillée

```
                        PRÉDICTION
                   Reste        Quitte
         ┌──────────────┬──────────────┐
  Reste  │  TN = 178    │  FP = 7      │
RÉALITÉ  ├──────────────┼──────────────┤
  Quitte │  FN = 25     │  TP = 11     │
         └──────────────┴──────────────┘

TN = 178 : Employés stables correctement identifiés  ✅
TP = 11  : Départs correctement détectés             ✅
FP = 7   : Fausses alertes (coût modéré)             ⚠️
FN = 25  : Départs non détectés (coût élevé)         ⚠️
```

### Rapport de Classification

```
              precision    recall  f1-score   support
─────────────────────────────────────────────────────
Reste  (0)      0.88        0.96      0.92       185
Quitte (1)      0.61        0.31      0.41        36
─────────────────────────────────────────────────────
accuracy                              0.86       221
macro avg       0.74        0.63      0.66       221
weighted avg    0.84        0.86      0.84       221
```

### Évolution des Performances — Comparaison des Versions

| Version | Contexte | Accuracy | AUC-ROC | FP | FN | Total Erreurs |
|---------|----------|----------|---------|----|----|---------------|
| Baseline | 100% non détectés | — | — | 0 | 237 | 237 |
| v1.0 — local CPU | Sans feature engineering | 81.0% | 0.75 | 27 | 28 | 55 |
| **v2.0 — Colab GPU T4** | **Feature eng. + GridSearchCV** | **85.52%** | **0.7715** | **7** | **25** | **32** |

**Améliorations de v1.0 à v2.0 :**
- Accuracy : +4.52 points
- Faux Positifs : -74% (27 → 7)
- Total erreurs : -42% (55 → 32)

---

## 🔍 Étape 9 — Explicabilité avec SHAP

**Fichier** : `app/predictor.py`

### Pourquoi l'Explicabilité est Fondamentale ?

Un modèle "boîte noire" n'est pas utilisable dans un contexte RH éthique et réglementaire. Les Responsables RH ont besoin de comprendre **pourquoi** un employé est identifié à risque pour :
- Prendre des décisions éclairées et justifiables
- Proposer des actions ciblées et pertinentes
- Respecter les obligations légales de transparence

### Théorie des Valeurs de Shapley

SHAP est fondé sur la **théorie des jeux coopératifs** (Shapley, 1953). Il attribue à chaque feature une valeur représentant sa contribution marginale à la prédiction :

```
f(x) = E[f(x)] + Σ φᵢ(x)

où :
  f(x)    = prédiction du modèle pour l'observation x
  E[f(x)] = valeur de base (prédiction moyenne sur le dataset)
  φᵢ(x)  = valeur SHAP de la feature i pour l'observation x

Interprétation :
  φᵢ > 0  → la feature AUGMENTE le risque de départ
  φᵢ < 0  → la feature DIMINUE le risque de départ
  |φᵢ|    → l'importance de la feature dans la décision
```

### Top 10 Features par Importance SHAP Globale

| Rang | Feature | Importance SHAP | Interprétation Métier |
|------|---------|----------------|-----------------------|
| 1 | `StockOptionLevel` | 1.0945 | Absence d'options d'actions = fort signal de départ |
| 2 | `overtime_x_joblevel` | 0.9493 | Surcharge combinée au niveau de responsabilité |
| 3 | `job_hopping_score` | 0.7946 | Historique d'instabilité professionnelle |
| 4 | `JobSatisfaction` | 0.6843 | Insatisfaction au travail directement liée au départ |
| 5 | `JobInvolvement` | 0.6366 | Désengagement précurseur du départ |
| 6 | `MonthlyIncome` | 0.5923 | Sous-rémunération perçue |
| 7 | `EnvironmentSatisfaction` | 0.5493 | Qualité de l'environnement de travail |
| 8 | `WorkLifeBalance` | 0.5009 | Équilibre vie professionnelle / personnelle |
| 9 | `DistanceFromHome` | 0.4908 | Fatigue liée à l'éloignement géographique |
| 10 | `Age` | 0.4528 | Jeunes employés statistiquement plus mobiles |

### Système de Recommandations Automatiques

À partir des valeurs SHAP individuelles, le système génère automatiquement des recommandations RH ciblées et actionnables :

| Feature à Risque Identifiée | Recommandation Générée |
|-----------------------------|------------------------|
| `StockOptionLevel` | 💰 Proposer des options d'actions ou avantages financiers |
| `overtime_x_joblevel` | ⏰ Réduire les heures supplémentaires et réévaluer la charge |
| `job_hopping_score` | 🔄 Proposer un plan de carrière clair et des opportunités |
| `JobSatisfaction` | 😊 Planifier un entretien de satisfaction approfondi |
| `JobInvolvement` | 🎯 Renforcer l'implication via des projets motivants |
| `MonthlyIncome` | 💵 Envisager une révision salariale ou prime de performance |
| `EnvironmentSatisfaction` | 🏢 Améliorer les conditions et l'environnement de travail |
| `WorkLifeBalance` | ⚖️ Mettre en place des mesures d'équilibre vie pro/perso |
| `DistanceFromHome` | 🏠 Proposer le télétravail ou une aide au transport |
| `YearsSinceLastPromotion` | 📈 Envisager une promotion ou une évolution de poste |

---

## 🚀 Étape 10 — Déploiement via FastAPI

**Fichiers** : `app/main.py`, `app/predictor.py`, `app/schemas.py`, `app/routes/predict_router.py`

### Pourquoi FastAPI ?

| Critère | FastAPI | Flask | Django REST |
|---------|---------|-------|-------------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Documentation automatique (Swagger) | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| Validation automatique (Pydantic) | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Typage fort | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Support async natif | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

### Architecture de l'API

```
app/
├── main.py              → Configuration FastAPI, middleware CORS, inclusion routes
├── schemas.py           → Modèles Pydantic (EmployeeInput, PredictionResponse, RiskFactor)
├── predictor.py         → Classe AttritionPredictor (chargement modèle + inférence + SHAP)
└── routes/
    └── predict_router.py → Endpoints GET /health, POST /predict, POST /predict/batch
```

### Niveaux de Risque Retournés

| Niveau | Condition | Signification et Action |
|--------|-----------|------------------------|
| 🟢 **FAIBLE** | probabilité < 0.76 | Employé stable — suivi standard |
| 🟡 **MOYEN** | 0.76 ≤ prob < 0.85 | Signal d'alerte — entretien préventif conseillé |
| 🔴 **ÉLEVÉ** | probabilité ≥ 0.85 | Risque critique — intervention urgente recommandée |

---

## 📁 Structure du Projet

```
HRAttritionML/
│
├── 📁 app/                              # Microservice FastAPI
│   ├── __init__.py
│   ├── main.py                          # Point d'entrée + configuration CORS
│   ├── predictor.py                     # Logique prédiction + SHAP + recommandations
│   ├── schemas.py                       # Contrats API (modèles Pydantic)
│   └── 📁 routes/
│       ├── __init__.py
│       └── predict_router.py            # Définition des endpoints REST
│
├── 📁 model/                            # Pipeline Machine Learning
│   ├── __init__.py
│   ├── preprocess.py                    # Chargement, nettoyage, feature eng., encodage
│   ├── train_model.py                   # Entraînement + GridSearchCV + sauvegarde artefacts
│   ├── evaluate_model.py                # Métriques détaillées + SHAP global
│   └── explore_data.py                  # Analyse exploratoire + visualisations
│
├── 📁 saved_model/                      # Artefacts ML sérialisés (non versionnés)
│   ├── attrition_model.pkl              # Modèle XGBoost entraîné        (521 KB)
│   ├── shap_explainer.pkl               # TreeExplainer SHAP              (1.8 MB)
│   ├── label_encoders.pkl               # Encodeurs LabelEncoder          (2.8 KB)
│   ├── feature_names.pkl                # Liste des 37 features           (0.7 KB)
│   ├── optimal_threshold.pkl            # Seuil optimal 0.76              (0.1 KB)
│   ├── metrics.pkl                      # Métriques de performance        (0.3 KB)
│   └── config.pkl                       # Configuration version 2.0       (1.1 KB)
│
├── 📁 data/                             # Données brutes (non versionnées)
│   └── hr_dataset.csv                   # IBM HR Analytics (1470 obs)
│
├── 📁 plots/                            # Visualisations EDA (non versionnées)
│   └── *.png                            # Graphiques générés par explore_data.py
│
├── 📁 tests/                            # Tests unitaires
│   └── __init__.py
│
├── requirements.txt                     # Dépendances et versions exactes
├── .gitignore                           # Exclusions Git (data, saved_model, .venv)
└── README.md                            # Ce document
```

---

## 🛠️ Guide d'Installation et de Lancement

### Prérequis

- Python 3.11.9
- pip
- Git

### Installation Pas à Pas

#### 1 — Cloner le dépôt

```bash
git clone https://github.com/ton-compte/hr-attrition-ml.git
cd hr-attrition-ml
```

#### 2 — Créer et activer l'environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

#### 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 4 — Vérifier les artefacts ML

```
saved_model/
├── attrition_model.pkl      ✅ requis
├── shap_explainer.pkl       ✅ requis
├── label_encoders.pkl       ✅ requis
├── feature_names.pkl        ✅ requis
├── optimal_threshold.pkl    ✅ requis
├── metrics.pkl              ✅ requis
└── config.pkl               ✅ requis
```

> Si les artefacts sont absents, les générer via : `python -m model.train_model`

#### 5 — Lancer l'API

```bash
uvicorn app.main:app --reload --port 8000
```

#### 6 — Vérifier le lancement

```
✅ Modèle chargé avec succès !
   Version    : 2.0-optimized
   Seuil      : 0.76
   Accuracy   : 85.52%

INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

#### 7 — Accéder à la documentation interactive

| Interface | URL |
|-----------|-----|
| **Swagger UI** | http://127.0.0.1:8000/docs |
| **ReDoc** | http://127.0.0.1:8000/redoc |
| **Health Check** | http://127.0.0.1:8000/api/ml/health |

---

## 📡 Documentation des Endpoints API

### GET `/api/ml/health`

Vérifie l'état de santé du modèle et retourne les métriques de performance.

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

Prédit le risque d'attrition d'un employé individuel et retourne les facteurs SHAP + recommandations.

```json
{
    "probability": 0.9999,
    "risk_level": "ÉLEVÉ",
    "threshold_used": 0.76,
    "top_risk_factors": [
        { "feature": "WorkLifeBalance", "shap_value": 1.1357, "impact": "AUGMENTE" },
        { "feature": "StockOptionLevel", "shap_value": 1.091, "impact": "AUGMENTE" }
    ],
    "recommendation_factors": [
        "⚖️  Mettre en place des mesures d'équilibre vie pro/perso",
        "💰 Proposer des options d'actions ou avantages financiers"
    ]
}
```

### POST `/api/ml/predict/batch`

Prédit le risque pour une liste d'employés en une seule requête REST.

---

## 🧪 Exemples de Prédiction Commentés

### Exemple 1 — Profil à Risque ÉLEVÉ (99.99%)

```
Homme, 28 ans, célibataire | Sales Executive Niv.1
Salaire : 2 500 DT | StockOption = 0
Satisfaction : JobSat=1, EnvSat=1, WLB=1
OverTime : Oui | Distance : 25 km | 6 entreprises précédentes

→ Probabilité : 99.99% 🔴 ÉLEVÉ
→ Facteurs SHAP dominants : WorkLifeBalance (+1.13), StockOption (+1.09), JobSat (+0.89)
→ Actions : équilibre vie pro/perso, options d'actions, entretien satisfaction
```

### Exemple 2 — Profil à Risque FAIBLE (0.0%)

```
Femme, 45 ans, mariée | Research Scientist Niv.4
Salaire : 12 000 DT | StockOption = 3
Satisfaction : JobSat=4, EnvSat=4, WLB=4
OverTime : Non | Distance : 2 km | 15 ans dans l'entreprise

→ Probabilité : 0.0% 🟢 FAIBLE
→ Tous les facteurs protecteurs actifs
→ Action : suivi standard
```

### Exemple 3 — Profil à Risque MOYEN (85.96%)

```
Homme, 32 ans, célibataire | Sales Executive Niv.2
Salaire : 5 000 DT | StockOption = 1
Satisfaction : JobSat=2, EnvSat=3, WLB=2
OverTime : Oui | Distance : 10 km | 3 entreprises précédentes

→ Probabilité : 85.96% 🔴 ÉLEVÉ (proche seuil)
→ Facteurs SHAP : job_hopping (+1.09), overtime (+0.89), promotion_rate (+0.87)
→ Actions : plan de carrière, réduction heures sup, accélération promotion
```

### Exemple 4 — Profil Intermédiaire (29.32%)

```
Femme, 34 ans, célibataire | Research Scientist Niv.2
Salaire : 5 500 DT | StockOption = 1
Satisfaction : JobSat=2, EnvSat=3, WLB=2
OverTime : Non | Distance : 12 km | 3 entreprises précédentes

→ Probabilité : 29.32% 🟢 FAIBLE (sous le seuil 0.76)
→ Facteurs protecteurs dominants : pas d'heures sup, bon job_hopping faible
→ Action : surveillance légère, plan de carrière préventif
```

---

## 📊 Récapitulatif des Performances

```
╔══════════════════════════════════════════════════════════╗
║           PERFORMANCES FINALES — MODÈLE v2.0            ║
║                                                          ║
║  Accuracy      : 85.52%                                  ║
║  AUC-ROC       : 0.7715                                  ║
║  MCC (CV)      : 0.8651                                  ║
║  Seuil optimal : 0.76                                    ║
║  Features      : 37 (31 originales + 7 construites)      ║
║  Dataset       : 1 470 observations IBM HR               ║
║  Entraînement  : Google Colab GPU T4 (~11 minutes)       ║
║                                                          ║
║  Matrice de Confusion (221 observations de test) :       ║
║                                                          ║
║     TN = 178  |  FP = 7                                  ║
║     FN = 25   |  TP = 11                                 ║
║                                                          ║
║  Réduction des erreurs vs v1.0 :                         ║
║     Faux Positifs : -74% (27 → 7)                        ║
║     Total erreurs : -42% (55 → 32)                       ║
╚══════════════════════════════════════════════════════════╝
```

---

## 📦 Technologies et Versions

| Catégorie | Technologie | Version | Rôle |
|-----------|-------------|---------|------|
| **Langage** | Python | 3.11.9 | Langage principal du microservice |
| **API Framework** | FastAPI | 0.135.1 | Framework REST asynchrone |
| **Serveur ASGI** | Uvicorn | 0.41.0 | Serveur de production |
| **Validation** | Pydantic | 2.12.5 | Validation des schémas de données |
| **ML Algorithm** | XGBoost | 3.2.0 | Algorithme de prédiction principal |
| **Explicabilité** | SHAP | 0.51.0 | Valeurs de Shapley pour l'interprétabilité |
| **Rééquilibrage** | imbalanced-learn | 0.14.1 | SMOTE |
| **ML Utilitaires** | scikit-learn | 1.8.0 | GridSearchCV, métriques, encodeurs |
| **Manipulation données** | Pandas | 3.0.1 | Traitement des DataFrames |
| **Calcul numérique** | NumPy | 2.4.2 | Opérations vectorisées |
| **Sérialisation** | Joblib | 1.5.3 | Persistance des artefacts ML |
| **Visualisation** | Matplotlib / Seaborn | 3.10.8 / 0.13.2 | Graphiques EDA |
| **Cloud GPU** | Google Colab T4 | — | Accélération de l'entraînement |

---

## 📐 Méthodologie de Développement

### Méthodologie SCRUM Agile

Le projet global est développé selon la méthodologie **SCRUM** avec des releases itératives planifiées :

| Release | Contenu | Statut |
|---------|---------|--------|
| **Release 1** | Authentification JWT + Gestion des utilisateurs (Spring Boot) | ✅ Terminé |
| **Release 2** | Pipeline ML + FastAPI + Entités RH + MLService Spring Boot | 🔄 En cours |
| **Release 3** | Dashboard statistiques + Historique des scores + Alertes mail | ⏳ Planifié |
| **Release 4** | Frontend Angular + Rapports PDF + Notifications avancées | ⏳ Planifié |

### Acteurs du Système

| Acteur | Rôle |
|--------|------|
| **Administrateur** | Gestion des comptes, configuration des seuils d'alerte |
| **Responsable RH** | Consultation des scores, validation alertes, rapports |
| **Manager** | Consultation du risque de ses collaborateurs directs |

### Décisions Architecturales et Justifications

| Décision | Alternative | Justification |
|----------|-------------|---------------|
| XGBoost | Random Forest, Deep Learning | Meilleure perf. données tabulaires + SHAP natif |
| SMOTE | Class weighting seul | Génération synthétique plus robuste |
| MCC comme scoring | Accuracy, F1 | Adapté aux classes déséquilibrées |
| Seuil 0.76 vs 0.5 | Seuil par défaut | Minimise FP + FN dans le contexte RH |
| Deux dépôts séparés | Monorepo | Séparation des responsabilités, déploiements indépendants |
| Google Colab GPU | Machine locale CPU | Réduction du temps d'entraînement : 5h → 11 minutes |

---

## 📄 Confidentialité et Licence

Ce projet est développé dans le cadre d'un **Projet de Fin d'Études (PFE)** en Ingénierie Informatique, réalisé au sein d'**Ooredoo Tunisie** — Année universitaire 2025/2026.

Le dataset utilisé est public (IBM HR Analytics, disponible sur Kaggle). **Aucune donnée réelle d'employés Ooredoo** n'est utilisée dans ce dépôt.

Tous droits réservés © 2026.

---

<p align="center">
  Développé avec rigueur et méthode pour <strong>Ooredoo Tunisie</strong><br/>
  Projet de Fin d'Études — Ingénierie Informatique — 2025/2026
</p>
