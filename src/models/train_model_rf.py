# src/models/train_model_rf.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
TARGET_COL = "Churn"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model_rf.joblib")

FIG_DIR = "reports/figures"


def build_rf_pipeline(df: pd.DataFrame, target_col: str):
    """
    Construit un pipeline :
    - préprocessing (scaler num + one-hot cat)
    - modèle RandomForest
    """
    X = df.drop(columns=[target_col, "customerID"])

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )

    return clf, numeric_features, categorical_features


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) Charger les données
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Cible binaire 0/1 pour les métriques plus tard
    y_train_bin = (y_train == "Yes").astype(int)
    y_test_bin = (y_test == "Yes").astype(int)

    # 2) Construire pipeline RF
    rf_pipeline, num_cols, cat_cols = build_rf_pipeline(train_df, TARGET_COL)
    print("Colonnes numériques :", num_cols)
    print("Colonnes catégorielles :", cat_cols)

    # 3) Grille d'hyperparamètres pour RF
    param_grid = {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [None, 5, 10, 15],
        "model__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    print("\n=== DÉBUT GRID SEARCH RandomForest ===")
    grid_search.fit(X_train, y_train_bin)
    print("=== FIN GRID SEARCH RandomForest ===\n")

    print("Meilleurs hyperparamètres RF :")
    print(grid_search.best_params_)
    print(f"Meilleur score ROC-AUC (CV) RF : {grid_search.best_score_:.4f}")

    # 4) Meilleur modèle
    best_rf = grid_search.best_estimator_

    # 5) Évaluation sur test
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n=== Classification report RF (threshold=0.5, classe 'Yes') ===")
    print(classification_report(y_test_bin, y_pred))

    roc_auc = roc_auc_score(y_test_bin, y_proba)
    print(f"ROC-AUC RF (test) : {roc_auc:.4f}")

    # 6) Courbe ROC
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"RF ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve - RandomForest - Churn")
    plt.legend()
    roc_path = os.path.join(FIG_DIR, "roc_curve_rf.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Courbe ROC RF sauvegardée dans {roc_path}")

    # 7) Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test_bin, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve - RandomForest - Churn")
    pr_path = os.path.join(FIG_DIR, "precision_recall_curve_rf.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Courbe PR RF sauvegardée dans {pr_path}")

    # 8) Sauvegarder le modèle
    joblib.dump(best_rf, MODEL_PATH)
    print(f"\nModèle RF sauvegardé dans {MODEL_PATH}")
    print("\nRandomForest entraîné + tuné ✅")


if __name__ == "__main__":
    main()
