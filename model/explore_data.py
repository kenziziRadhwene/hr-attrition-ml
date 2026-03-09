import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
DATA_PATH = "data/hr_dataset.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def explore(df: pd.DataFrame):

    # ─────────────────────────────────────
    # 1. Vue générale
    # ─────────────────────────────────────
    print("=" * 60)
    print("📊 VUE GÉNÉRALE DU DATASET")
    print("=" * 60)
    print(f"Dimensions     : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"Mémoire        : {df.memory_usage().sum() / 1024:.1f} KB")
    print(f"\nTypes de données :\n{df.dtypes.value_counts()}")
    print(f"\nValeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # ─────────────────────────────────────
    # 2. Distribution de la cible
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎯 DISTRIBUTION DE LA CIBLE (Attrition)")
    print("=" * 60)
    attrition_counts = df['Attrition'].value_counts()
    attrition_pct = df['Attrition'].value_counts(normalize=True) * 100
    print(f"Reste  (No)  : {attrition_counts['No']:4d} ({attrition_pct['No']:.1f}%)")
    print(f"Quitte (Yes) : {attrition_counts['Yes']:4d} ({attrition_pct['Yes']:.1f}%)")
    print(f"⚠️  Déséquilibre détecté → SMOTE nécessaire")

    # ─────────────────────────────────────
    # 3. Statistiques descriptives
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("📈 STATISTIQUES DESCRIPTIVES")
    print("=" * 60)
    print(df.describe().round(2).to_string())

    # ─────────────────────────────────────
    # 4. Analyse par Attrition
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 ANALYSE PAR ATTRITION")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    analysis = df.groupby('Attrition')[numeric_cols].mean().round(2)
    print(analysis.T.to_string())

    # ─────────────────────────────────────
    # 5. Corrélations avec Attrition
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔗 CORRÉLATIONS AVEC ATTRITION")
    print("=" * 60)

    df_encoded = df.copy()
    df_encoded['Attrition'] = (df_encoded['Attrition'] == 'Yes').astype(int)

    # Encoder colonnes catégorielles pour corrélation
    for col in df_encoded.select_dtypes(include=['object', 'str']).columns:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes

    correlations = df_encoded.corr()['Attrition'].drop('Attrition')
    correlations = correlations.abs().sort_values(ascending=False)

    print("\nTop 15 features corrélées avec Attrition :")
    print(correlations.head(15).round(4).to_string())

    # ─────────────────────────────────────
    # 6. Graphiques
    # ─────────────────────────────────────
    print("\n📊 Génération des graphiques...")

    # Graphique 1 — Distribution Attrition
    plt.figure(figsize=(8, 5))
    colors = ['#2ecc71', '#e74c3c']
    attrition_counts.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Distribution Attrition', fontsize=14, fontweight='bold')
    plt.xlabel('Attrition')
    plt.ylabel('Nombre d\'employés')
    plt.xticks(rotation=0)
    for i, v in enumerate(attrition_counts):
        plt.text(i, v + 10, f'{v}\n({attrition_pct.iloc[i]:.1f}%)',
                 ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/1_distribution_attrition.png", dpi=150)
    plt.close()
    print("   ✅ Distribution Attrition sauvegardée")

    # Graphique 2 — Age vs Attrition
    plt.figure(figsize=(10, 5))
    df.boxplot(column='Age', by='Attrition', figsize=(8, 5))
    plt.title('Age vs Attrition', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Attrition')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/2_age_vs_attrition.png", dpi=150)
    plt.close()
    print("   ✅ Age vs Attrition sauvegardée")

    # Graphique 3 — Salaire vs Attrition
    plt.figure(figsize=(10, 5))
    df.boxplot(column='MonthlyIncome', by='Attrition', figsize=(8, 5))
    plt.title('Salaire Mensuel vs Attrition', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Attrition')
    plt.ylabel('Salaire Mensuel')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/3_salaire_vs_attrition.png", dpi=150)
    plt.close()
    print("   ✅ Salaire vs Attrition sauvegardée")

    # Graphique 4 — Heatmap corrélations
    plt.figure(figsize=(18, 14))
    df_encoded_numeric = df_encoded.select_dtypes(include=[np.number])
    mask = np.triu(np.ones_like(df_encoded_numeric.corr(), dtype=bool))
    sns.heatmap(
        df_encoded_numeric.corr(),
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7}
    )
    plt.title('Heatmap des Corrélations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/4_heatmap_correlations.png", dpi=150)
    plt.close()
    print("   ✅ Heatmap corrélations sauvegardée")

    # Graphique 5 — Top features corrélées
    plt.figure(figsize=(10, 8))
    correlations.head(15).plot(kind='barh', color='steelblue', edgecolor='black')
    plt.title('Top 15 Features Corrélées avec Attrition',
              fontsize=14, fontweight='bold')
    plt.xlabel('Corrélation absolue')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/5_top_features_correlation.png", dpi=150)
    plt.close()
    print("   ✅ Top features sauvegardée")

    print(f"\n✅ Tous les graphiques sauvegardés dans '{PLOTS_DIR}/'")
    return correlations


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    explore(df)