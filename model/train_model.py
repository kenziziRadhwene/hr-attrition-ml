import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, accuracy_score,
                             matthews_corrcoef, f1_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import joblib
import os

from model.preprocess import (load_data, clean_data, create_advanced_features,
                               encode_data, split_features_target, save_artifacts)

# ─────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────
DATA_PATH = "data/hr_dataset.csv"
SAVE_DIR = "saved_model"


def find_optimal_threshold(model, X_val, y_val):
    """Trouve le seuil optimal qui minimise FP + FN"""
    print("\n🎯 Recherche du seuil optimal...")
    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.2, 0.8, 0.02)

    best_score = float('inf')
    best_threshold = 0.5
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        total_errors = fp + fn
        if fp > 0 and fn > 0:
            imbalance_penalty = abs(fp - fn) / (fp + fn)
        else:
            imbalance_penalty = 1.0
        final_score = total_errors * (1 + 0.2 * imbalance_penalty)
        results.append({'threshold': threshold, 'fp': fp,
                        'fn': fn, 'total_errors': total_errors,
                        'final_score': final_score})
        if final_score < best_score:
            best_score = final_score
            best_threshold = threshold

    print("\n📊 Top 5 seuils :")
    for r in sorted(results, key=lambda x: x['final_score'])[:5]:
        print(f"   Seuil {r['threshold']:.2f}: FP={r['fp']}, "
              f"FN={r['fn']}, Total={r['total_errors']}")

    print(f"\n✅ Seuil optimal : {best_threshold:.2f}")
    return best_threshold


def tune_model(X_train, y_train, X_val, y_val):
    """Optimisation des hyperparamètres"""
    print("\n🔧 Optimisation des hyperparamètres (GridSearchCV)...")

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [4, 5],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [2, 3],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 2]
    }

    base_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='matthews_corrcoef',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    print(f"\n✅ Meilleurs paramètres :")
    for param, value in grid_search.best_params_.items():
        print(f"   {param:<25} : {value}")
    print(f"\n✅ Meilleur MCC CV : {grid_search.best_score_:.4f}")

    return model


def evaluate_model(model, X_test, y_test, threshold):
    """Évaluation complète du modèle"""
    print("\n📊 Évaluation finale :")
    print("=" * 60)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n🎯 Seuil utilisé     : {threshold:.2f}")
    print(f"\n📈 Métriques :")
    print(f"   Accuracy         : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"   AUC-ROC          : {auc_roc:.4f}")
    print(f"   F1-Score         : {f1:.4f}")
    print(f"   MCC              : {mcc:.4f}")
    print(f"\n📊 Matrice de confusion :")
    print(f"                  Prédit Reste  Prédit Quitte")
    print(f"   Réel Reste   :     {tn:5d}        {fp:5d}")
    print(f"   Réel Quitte  :     {fn:5d}        {tp:5d}")
    print(f"\n🎯 Résumé erreurs :")
    print(f"   Faux Positifs (FP) : {fp}")
    print(f"   Faux Négatifs (FN) : {fn}")
    print(f"   Total erreurs      : {fp + fn}")
    print("=" * 60)

    return {
        'accuracy': accuracy, 'auc_roc': auc_roc,
        'f1': f1, 'mcc': mcc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'threshold': threshold
    }


def train():
    """Fonction principale d'entraînement"""
    print("\n" + "=" * 60)
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 60)

    # 1. Preprocessing
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = create_advanced_features(df)
    df, label_encoders = encode_data(df)
    X, y = split_features_target(df)
    feature_names = list(X.columns)

    # 2. Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

    print(f"\n✂️  Train : {X_train.shape[0]} | Val : {X_val.shape[0]} | Test : {X_test.shape[0]}")

    # 3. SMOTE
    print("\n⚖️  Application SMOTE...")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"✅ Après SMOTE : {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")

    # 4. Entraînement
    model = tune_model(X_train_balanced, y_train_balanced, X_val, y_val)

    # 5. Seuil optimal
    optimal_threshold = find_optimal_threshold(model, X_val, y_val)

    # 6. Évaluation
    metrics = evaluate_model(model, X_test, y_test, optimal_threshold)

    # 7. SHAP
    print("\n🔍 Calcul SHAP Values...")
    explainer = shap.TreeExplainer(model)
    print("✅ SHAP Explainer créé !")

    # 8. Sauvegarde
    print("\n💾 Sauvegarde...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(model,             f"{SAVE_DIR}/attrition_model.pkl")
    joblib.dump(explainer,         f"{SAVE_DIR}/shap_explainer.pkl")
    joblib.dump(feature_names,     f"{SAVE_DIR}/feature_names.pkl")
    joblib.dump(label_encoders,    f"{SAVE_DIR}/label_encoders.pkl")
    joblib.dump(optimal_threshold, f"{SAVE_DIR}/optimal_threshold.pkl")
    joblib.dump(metrics,           f"{SAVE_DIR}/metrics.pkl")

    config = {
        'model_type': 'XGBoost',
        'threshold': optimal_threshold,
        'features': feature_names,
        'metrics': metrics,
        'version': '2.0-optimized'
    }
    joblib.dump(config, f"{SAVE_DIR}/config.pkl")
    save_artifacts(label_encoders, feature_names, SAVE_DIR)

    print("\n🎉 ENTRAÎNEMENT TERMINÉ !")
    return model, explainer, feature_names, optimal_threshold, metrics


if __name__ == "__main__":
    model, explainer, feature_names, threshold, metrics = train()
    print(f"\n📋 RÉSUMÉ FINAL")
    print(f"   Accuracy  : {metrics['accuracy']:.2%}")
    print(f"   AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"   FP        : {metrics['fp']}")
    print(f"   FN        : {metrics['fn']}")
    print(f"   Seuil     : {threshold:.2f}")