import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import shap
import joblib
import matplotlib.pyplot as plt


# ─────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────
SAVE_DIR = "saved_model"


def load_artifacts():
    """Charge le modèle et les artefacts sauvegardés"""
    print("📂 Chargement des artefacts...")
    model = joblib.load(f"{SAVE_DIR}/attrition_model.pkl")
    explainer = joblib.load(f"{SAVE_DIR}/shap_explainer.pkl")
    feature_names = joblib.load(f"{SAVE_DIR}/feature_names.pkl")
    print("✅ Artefacts chargés !")
    return model, explainer, feature_names


def evaluate(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test"""
    print("\n📊 Évaluation complète du modèle :")
    print("─" * 50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métriques principales
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC   : {roc_auc_score(y_test, y_prob):.4f}")

    print(f"\nRapport complet :\n")
    print(classification_report(y_test, y_pred,
                                target_names=['Reste', 'Quitte']))

    print(f"Matrice de confusion :\n{confusion_matrix(y_test, y_pred)}")

    return y_pred, y_prob


def get_top_features(explainer, X_test, feature_names, top_n=10):
    """Retourne les features les plus importantes via SHAP"""
    print(f"\n🏆 Top {top_n} facteurs de risque (SHAP) :")
    print("─" * 50)

    shap_values = explainer.shap_values(X_test)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)

    for i, row in feature_importance.head(top_n).iterrows():
        print(f"   {row['feature']:<35} : {row['importance']:.4f}")

    return feature_importance


def get_employee_risk_factors(explainer, employee_data, feature_names):
    """Retourne les facteurs de risque pour un employé spécifique"""
    shap_values = explainer.shap_values(employee_data)

    factors = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', ascending=False)

    print("\n🔍 Facteurs de risque de l'employé :")
    print("─" * 50)
    print("Facteurs qui AUGMENTENT le risque :")
    positive = factors[factors['shap_value'] > 0]
    for _, row in positive.head(5).iterrows():
        print(f"   ⬆️  {row['feature']:<35} : +{row['shap_value']:.4f}")

    print("\nFacteurs qui DIMINUENT le risque :")
    negative = factors[factors['shap_value'] < 0]
    for _, row in negative.tail(5).iterrows():
        print(f"   ⬇️  {row['feature']:<35} : {row['shap_value']:.4f}")

    return factors