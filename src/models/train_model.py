# src/models/train_model.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from pipeline import build_pipeline

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
TARGET_COL = "Churn"

FIG_DIR = "reports/figures"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) Charger les données
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    mapping = {"No": 0, "Yes": 1}
    train_df[TARGET_COL] = train_df[TARGET_COL].map(mapping)
    test_df[TARGET_COL] = test_df[TARGET_COL].map(mapping)
    
    # 2) Optionnel : vérifier la distribution de la cible
    print("Distribution cible (train) :")
    print(train_df[TARGET_COL].value_counts(normalize=True))

    # 3) Séparer X / y
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # 4) Construire le pipeline
    clf, num_cols, cat_cols = build_pipeline(train_df, TARGET_COL)
    print("\nColonnes numériques :", num_cols)
    print("Colonnes catégorielles :", cat_cols)

    # 5) Entraîner le modèle
    print("\nEntraînement du modèle...")
    clf.fit(X_train, y_train)

    # 6) Prédictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # 7) Métriques
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC : {roc_auc:.4f}")

    # 8) Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve - Churn model")
    plt.legend()
    roc_path = os.path.join(FIG_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Courbe ROC sauvegardée dans {roc_path}")

    # 9) Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve - Churn model")
    pr_path = os.path.join(FIG_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Courbe PR sauvegardée dans {pr_path}")

    # 10) Sauvegarder le modèle
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModèle sauvegardé dans {MODEL_PATH}")
    print("\nEntraînement terminé ✅")


if __name__ == "__main__":
    main()
