# src/models/train_model_tuned.py

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
from sklearn.model_selection import GridSearchCV

from pipeline import build_pipeline

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model_tuned.joblib")
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

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # 2) Construire le pipeline de base
    base_pipeline, num_cols, cat_cols = build_pipeline(train_df, TARGET_COL)
    print("Colonnes numériques :", num_cols)
    print("Colonnes catégorielles :", cat_cols)

    # 3) Définir la grille d'hyperparamètres
    # 'model__' car dans notre Pipeline, l'étape du modèle s'appelle 'model'
    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }

    # 4) Définir le GridSearch
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    print("\n=== DÉBUT GRID SEARCH ===")
    grid_search.fit(X_train, y_train)
    print("=== FIN GRID SEARCH ===\n")

    print("Meilleurs hyperparamètres trouvés :")
    print(grid_search.best_params_)
    print(f"Meilleur score ROC-AUC (CV) : {grid_search.best_score_:.4f}")

    # 5) Récupérer le meilleur modèle entraîné sur tout le train
    best_model = grid_search.best_estimator_

    # 6) Évaluer sur le test
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n=== Classification report (modèle tuné) ===")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC (test) : {roc_auc:.4f}")

    # 7) Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve - Churn model (tuned)")
    plt.legend()
    roc_path = os.path.join(FIG_DIR, "roc_curve_tuned.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Courbe ROC (tuned) sauvegardée dans {roc_path}")

    # 8) Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve - Churn model (tuned)")
    pr_path = os.path.join(FIG_DIR, "precision_recall_curve_tuned.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Courbe PR (tuned) sauvegardée dans {pr_path}")

    # 9) Sauvegarder le modèle tuné
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModèle tuné sauvegardé dans {MODEL_PATH}")
    print("\nEntraînement + tuning terminés ✅")


if __name__ == "__main__":
    main()
