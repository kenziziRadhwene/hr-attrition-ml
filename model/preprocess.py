import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Charge le dataset IBM HR Analytics"""
    print("📂 Chargement du dataset...")
    df = pd.read_csv(filepath)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"\n📊 Distribution Attrition :\n{df['Attrition'].value_counts()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles"""
    print("\n🧹 Nettoyage des données...")
    columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop(columns=columns_to_drop)
    print(f"✅ Colonnes supprimées : {columns_to_drop}")
    return df


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering avancé"""
    print("\n🔧 Création de features avancées...")

    df['satisfaction_score'] = (df['JobSatisfaction'] +
                                df['EnvironmentSatisfaction'] +
                                df['RelationshipSatisfaction']) / 3

    df['promotion_rate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

    df['tenure_satisfaction'] = df['YearsAtCompany'] * df['satisfaction_score']

    df['tenure_category'] = pd.cut(df['YearsAtCompany'],
                                   bins=[0, 1, 3, 5, 10, 40],
                                   labels=[0, 1, 2, 3, 4])
    df['tenure_category'] = df['tenure_category'].fillna(0).astype(int)

    if 'NumCompaniesWorked' in df.columns:
        df['job_hopping_score'] = df['NumCompaniesWorked'] / (df['YearsAtCompany'] + 1)
        df['job_hopping_score'] = df['job_hopping_score'].fillna(0).replace(
            [np.inf, -np.inf], 0)

    if 'OverTime' in df.columns and 'JobLevel' in df.columns:
        df['overtime_x_joblevel'] = df['OverTime'] * df['JobLevel']

    df['work_life_balance_score'] = df['WorkLifeBalance'] * df['satisfaction_score']

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print(f"✅ {len(df.columns)} features après engineering")
    return df


def encode_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode les variables catégorielles"""
    print("\n🔄 Encodage des variables catégorielles...")
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"   ✅ {col} encodé")

    return df, label_encoders


def split_features_target(df: pd.DataFrame) -> tuple:
    """Sépare features et target"""
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    print(f"\n📈 Features : {X.shape[1]} colonnes")
    print(f"🎯 Target : {y.value_counts().to_dict()}")
    return X, y


def save_artifacts(label_encoders: dict, feature_names: list, save_dir: str):
    """Sauvegarde les artefacts de preprocessing"""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(label_encoders, f"{save_dir}/label_encoders.pkl")
    joblib.dump(feature_names, f"{save_dir}/feature_names.pkl")
    print(f"\n💾 Artefacts sauvegardés dans {save_dir}/")