import numpy as np
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────
SAVE_DIR = "saved_model"

# ─────────────────────────────────────────
# Mapping Départements Ooredoo → IBM HR
# ─────────────────────────────────────────
DEPARTMENT_MAPPING = {
    # → Sales
    "DIRECTION_COMMERCIALE"             : "Sales",
    "SERVICE_APPLICATION_VENTE"         : "Sales",
    "DIRECTION_SERVICE_CLIENT"          : "Sales",

    # → Research & Development
    "DIRECTION_TECHNOLOGIQUE"           : "Research & Development",
    "DIRECTION_SYSTEMES_INFORMATION"    : "Research & Development",
    "DIRECTION_OPERATIONS"              : "Research & Development",
    "DIRECTION_TECHNIQUE_FIXE"          : "Research & Development",
    "DIRECTION_INGENIERIE_RESEAUX"      : "Research & Development",
    "DEPARTEMENT_PERFORMANCE_RESEAUX"   : "Research & Development",
    "SERVICE_DATA_ENGINEERING"          : "Research & Development",
    "SERVICE_DASHBOARD_DATA_MINING"     : "Research & Development",

    # → Human Resources
    "DIRECTION_RESSOURCES_HUMAINES"          : "Human Resources",
    "DIRECTION_GENERALE"                     : "Human Resources",
    "DIRECTION_ADMINISTRATIVE_FINANCIERE"    : "Human Resources",
    "DIRECTION_JURIDIQUE"                    : "Human Resources",
    "DIRECTION_RELATIONS_OPERATEURS"         : "Human Resources",
}

# ─────────────────────────────────────────
# Mapping JobRole Ooredoo → IBM HR
# ─────────────────────────────────────────
JOBROLE_MAPPING = {
    # → Sales
    "COMMERCIAL"                : "Sales Executive",
    "RESPONSABLE_COMMERCIAL"    : "Sales Executive",
    "DIRECTEUR_COMMERCIAL"      : "Sales Executive",
    "CHARGE_MARKETING"          : "Sales Executive",
    "CONSEILLER_CLIENT"         : "Sales Representative",
    "SUPERVISEUR_CENTRE_APPEL"  : "Sales Representative",
    "DIRECTEUR_SERVICE_CLIENT"  : "Manager",
    "RESPONSABLE_SERVICE_CLIENT": "Sales Executive",

    # → Research & Development
    "DATA_ENGINEER"             : "Research Scientist",
    "DATA_ANALYST"              : "Research Scientist",
    "DEVELOPPEUR"               : "Laboratory Technician",
    "ANALYSTE_SYSTEME"          : "Research Scientist",
    "ADMINISTRATEUR_SYSTEME"    : "Laboratory Technician",
    "INGENIEUR_RESEAU"          : "Research Scientist",
    "INGENIEUR_TELECOM"         : "Research Scientist",
    "TECHNICIEN_RESEAU"         : "Laboratory Technician",
    "ARCHITECTE_SYSTEME"        : "Research Director",
    "DIRECTEUR_TECHNIQUE"       : "Research Director",
    "DIRECTEUR_SI"              : "Research Director",

    # → Human Resources / Management
    "RESPONSABLE_RH"            : "Human Resources",
    "CHARGE_RECRUTEMENT"        : "Human Resources",
    "CHARGE_FORMATION"          : "Human Resources",
    "DIRECTEUR_RH"              : "Manager",
    "DIRECTEUR_GENERAL"         : "Manager",
    "DIRECTEUR_FINANCIER"       : "Manager",
    "COMPTABLE"                 : "Human Resources",
    "CONTROLEUR_GESTION"        : "Human Resources",
    "DIRECTEUR_JURIDIQUE"       : "Manager",
    "JURISTE"                   : "Human Resources",
    "CONSEILLER_JURIDIQUE"      : "Human Resources",
    "ASSISTANT_DIRECTION"       : "Human Resources",
}

# ─────────────────────────────────────────
# Mapping BusinessTravel Ooredoo → IBM HR
# ─────────────────────────────────────────
BUSINESS_TRAVEL_MAPPING = {
    "JAMAIS"        : "Non-Travel",
    "RAREMENT"      : "Travel_Rarely",
    "FREQUEMMENT"   : "Travel_Frequently",
    # Valeurs IBM directes (déjà compatibles)
    "Non-Travel"        : "Non-Travel",
    "Travel_Rarely"     : "Travel_Rarely",
    "Travel_Frequently" : "Travel_Frequently",
}


class AttritionPredictor:
    """
    Classe principale de prédiction du risque d'attrition
    Charge le modèle XGBoost + SHAP et effectue les prédictions
    Gère le mapping des données Ooredoo vers le format IBM HR
    """

    def __init__(self):
        print("🔄 Chargement du modèle et des artefacts...")
        self.model           = joblib.load(f"{SAVE_DIR}/attrition_model.pkl")
        self.explainer       = joblib.load(f"{SAVE_DIR}/shap_explainer.pkl")
        self.feature_names   = joblib.load(f"{SAVE_DIR}/feature_names.pkl")
        self.label_encoders  = joblib.load(f"{SAVE_DIR}/label_encoders.pkl")
        self.threshold       = joblib.load(f"{SAVE_DIR}/optimal_threshold.pkl")
        self.config          = joblib.load(f"{SAVE_DIR}/config.pkl")
        self.metrics         = joblib.load(f"{SAVE_DIR}/metrics.pkl")
        print("✅ Modèle chargé avec succès !")
        print(f"   Version    : {self.config['version']}")
        print(f"   Seuil      : {self.threshold:.2f}")
        print(f"   Accuracy   : {self.metrics['accuracy']:.2%}")

    def _apply_ooredoo_mapping(self, employee_data: dict) -> dict:
        """
        Traduit les valeurs Ooredoo vers les valeurs IBM HR
        attendues par le modèle XGBoost
        """
        # Mapping Department
        dept = employee_data.get('Department', '')
        if dept in DEPARTMENT_MAPPING:
            employee_data['Department'] = DEPARTMENT_MAPPING[dept]
            print(f"   🔄 Department : {dept} → {employee_data['Department']}")

        # Mapping JobRole
        role = employee_data.get('JobRole', '')
        if role in JOBROLE_MAPPING:
            employee_data['JobRole'] = JOBROLE_MAPPING[role]
            print(f"   🔄 JobRole    : {role} → {employee_data['JobRole']}")

        # Mapping BusinessTravel
        travel = employee_data.get('BusinessTravel', '')
        if travel in BUSINESS_TRAVEL_MAPPING:
            employee_data['BusinessTravel'] = BUSINESS_TRAVEL_MAPPING[travel]
            print(f"   🔄 Travel     : {travel} → {employee_data['BusinessTravel']}")

        return employee_data

    def _preprocess_input(self, employee_data: dict) -> pd.DataFrame:
        """Prétraite les données d'entrée"""

        # ✅ Appliquer le mapping Ooredoo → IBM HR
        employee_data = self._apply_ooredoo_mapping(employee_data)

        df = pd.DataFrame([employee_data])

        # Encodage des variables catégorielles
        categorical_cols = ['Gender', 'MaritalStatus', 'Department',
                            'JobRole', 'BusinessTravel', 'EducationField',
                            'OverTime']

        for col in categorical_cols:
            if col in self.label_encoders and col in df.columns:
                le = self.label_encoders[col]
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # Valeur inconnue → valeur par défaut 0
                    print(f"   ⚠️  Valeur inconnue pour {col} : {df[col].values[0]} → 0")
                    df[col] = 0

        # Feature engineering identique à l'entraînement
        df['satisfaction_score'] = (df['JobSatisfaction'] +
                                    df['EnvironmentSatisfaction'] +
                                    df['RelationshipSatisfaction']) / 3

        df['promotion_rate'] = df['YearsSinceLastPromotion'] / (
                df['YearsAtCompany'] + 1)

        df['tenure_satisfaction'] = df['YearsAtCompany'] * df['satisfaction_score']

        df['tenure_category'] = pd.cut(
            df['YearsAtCompany'],
            bins=[0, 1, 3, 5, 10, 40],
            labels=[0, 1, 2, 3, 4]
        ).fillna(0).astype(int)

        df['job_hopping_score'] = df['NumCompaniesWorked'] / (
                df['YearsAtCompany'] + 1)
        df['job_hopping_score'] = df['job_hopping_score'].fillna(0).replace(
            [np.inf, -np.inf], 0)

        df['overtime_x_joblevel'] = df['OverTime'] * df['JobLevel']

        df['work_life_balance_score'] = df['WorkLifeBalance'] * df['satisfaction_score']

        # Aligner les colonnes avec les features du modèle
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names]
        df = df.fillna(0)

        return df

    def _get_risk_level(self, probability: float) -> str:
        """Détermine le niveau de risque"""
        if probability >= self.threshold:
            if probability >= 0.85:
                return "ÉLEVÉ"
            else:
                return "MOYEN"
        return "FAIBLE"

    def _get_shap_factors(self, df: pd.DataFrame) -> list:
        """Calcule les facteurs SHAP pour un employé"""
        shap_values = self.explainer.shap_values(df)

        factors = []
        for i, feature in enumerate(self.feature_names):
            shap_val = float(shap_values[0][i])
            factors.append({
                'feature': feature,
                'shap_value': round(shap_val, 4),
                'impact': 'AUGMENTE' if shap_val > 0 else 'DIMINUE'
            })

        # Trier par valeur absolue décroissante
        factors = sorted(factors,
                         key=lambda x: abs(x['shap_value']),
                         reverse=True)
        return factors[:10]

    def _get_recommendations(self, shap_factors: list, probability: float) -> list:
        """
        Génère des recommandations basées sur les facteurs SHAP
        Uniquement si le risque est MOYEN ou ÉLEVÉ
        """
        recommendations = []

        # Pas de recommandations si risque faible
        if probability < self.threshold:
            return ["✅ Employé stable — Suivi standard recommandé"]

        # Mapping facteurs → recommandations
        rules = {
            'StockOptionLevel'          : "💰 Proposer des options d'actions ou avantages financiers",
            'overtime_x_joblevel'       : "⏰ Réduire les heures supplémentaires et réévaluer la charge de travail",
            'job_hopping_score'         : "🔄 Proposer un plan de carrière clair et des opportunités d'évolution",
            'JobSatisfaction'           : "😊 Planifier un entretien de satisfaction et améliorer les conditions de travail",
            'JobInvolvement'            : "🎯 Renforcer l'implication via des projets motivants",
            'MonthlyIncome'             : "💵 Envisager une révision salariale ou une prime de performance",
            'EnvironmentSatisfaction'   : "🏢 Améliorer l'environnement et les conditions de travail",
            'WorkLifeBalance'           : "⚖️  Mettre en place des mesures d'équilibre vie pro/perso",
            'DistanceFromHome'          : "🏠 Proposer le télétravail ou une aide au transport",
            'Age'                       : "👶 Adapter le parcours d'intégration pour les jeunes collaborateurs",
            'YearsAtCompany'            : "🏆 Valoriser l'ancienneté et proposer une mobilité interne",
            'YearsSinceLastPromotion'   : "📈 Envisager une promotion ou une évolution de poste",
            'satisfaction_score'        : "💬 Planifier un entretien managérial approfondi",
            'promotion_rate'            : "🚀 Accélérer le parcours de promotion",
            'YearsWithCurrManager'      : "👔 Envisager un changement de manager ou de périmètre",
            'BusinessTravel'            : "✈️  Réduire la fréquence des déplacements professionnels",
            'NumCompaniesWorked'        : "🔄 Proposer un plan de fidélisation personnalisé",
            'TotalWorkingYears'         : "🎓 Valoriser l'expérience via une progression accélérée",
            'work_life_balance_score'   : "⚖️  Améliorer l'équilibre global satisfaction / vie perso",
            'tenure_category'           : "📅 Adapter le suivi selon l'ancienneté de l'employé",
        }

        for factor in shap_factors[:5]:
            if factor['impact'] == 'AUGMENTE':
                feature = factor['feature']
                if feature in rules:
                    recommendations.append(rules[feature])

        if not recommendations:
            recommendations.append("📋 Planifier un entretien individuel de suivi approfondi")

        return recommendations

    def predict(self, employee_data: dict) -> dict:
        """
        Prédit le risque d'attrition pour un employé
        Gère automatiquement le mapping Ooredoo → IBM HR
        """
        print(f"\n🔍 Prédiction en cours...")

        # Preprocessing + mapping
        df = self._preprocess_input(employee_data)

        # Prédiction
        probability = float(self.model.predict_proba(df)[0][1])
        risk_level  = self._get_risk_level(probability)

        # SHAP
        shap_factors = self._get_shap_factors(df)

        # Recommandations (uniquement si risque MOYEN ou ÉLEVÉ)
        recommendations = self._get_recommendations(shap_factors, probability)

        print(f"   ✅ Probabilité  : {probability:.2%}")
        print(f"   ✅ Niveau risque : {risk_level}")

        return {
            'probability'           : round(probability, 4),
            'risk_level'            : risk_level,
            'threshold_used'        : round(float(self.threshold), 2),
            'top_risk_factors'      : shap_factors,
            'recommendation_factors': recommendations
        }

    def get_health(self) -> dict:
        """Retourne l'état de santé du modèle"""
        return {
            'status'        : 'OK',
            'model_version' : self.config['version'],
            'threshold'     : round(float(self.threshold), 2),
            'accuracy'      : round(float(self.metrics['accuracy']), 4),
            'auc_roc'       : round(float(self.metrics['auc_roc']), 4),
            'fp'            : int(self.metrics['fp']),
            'fn'            : int(self.metrics['fn'])
        }


# Instance globale du predictor
predictor = AttritionPredictor()