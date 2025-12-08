import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/churn.csv"
PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

TARGET_COL = "Churn"

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Lecture des données depuis {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH)

    print("\nTypes de colonnes AVANT nettoyage :")
    print(df.dtypes)

    # ====== 1) Nettoyage / feature engineering simple ======

    # a) Convertir TotalCharges en numérique
    # errors='coerce' transforme les valeurs non num (vides, espaces) en NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # b) Gérer les NaN dans TotalCharges
    # Ici on remplace par la médiane (stratégie simple et robuste)
    median_total_charges = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

    # c) Optionnel : vérifier que la cible est bien de type "object" ou "category"
    # df[TARGET_COL] = df[TARGET_COL].astype("category")

    print("\nTypes de colonnes APRÈS nettoyage :")
    print(df.dtypes)

    # ====== 2) Split train/test avec stratification ======
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\nTrain sauvegardé dans : {TRAIN_PATH}")
    print(f"Test sauvegardé dans  : {TEST_PATH}")
    print("\nTerminé ✅")


if __name__ == "__main__":
    main()
