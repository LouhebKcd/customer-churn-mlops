# src/models/evaluate_thresholds.py

import os
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/churn_model_tuned.joblib"  # ou churn_model.joblib si tu préfères
TARGET_COL = "Churn"

THRESHOLDS = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]


def main():
    # 1) Charger les données et le modèle
    df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]

    # On passe la cible en binaire : 1 = Yes (churn), 0 = No
    y_true = (y_test == "Yes").astype(int)

    # 2) Obtenir les probabilités de churn
    # On suppose que la classe positive (churn) est la 2e colonne
    proba = model.predict_proba(X_test)[:, 1]

    print("Évaluation à différents seuils pour la classe 'Churn' (Yes)")
    print("Threshold | Precision | Recall | F1-score")
    print("------------------------------------------")

    for thr in THRESHOLDS:
        y_pred = (proba >= thr).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"{thr:8.2f} | {precision:9.3f} | {recall:6.3f} | {f1:8.3f}")


if __name__ == "__main__":
    main()
