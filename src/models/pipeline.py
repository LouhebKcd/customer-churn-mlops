# src/models/pipeline.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_pipeline(df: pd.DataFrame, target_col: str):
    """
    df : dataframe complet (y compris la colonne cible)
    target_col : nom de la colonne cible (ici 'Churn')

    Retourne :
    - pipeline scikit-learn (préprocessing + modèle)
    - liste des colonnes numériques
    - liste des colonnes catégorielles
    """

    # On enlève la cible + éventuellement des colonnes qu'on ne veut pas utiliser comme features
    X = df.drop(columns=[target_col, "customerID"])

    # Séparation numérique / catégoriel
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Transformations pour les colonnes numériques
    numeric_transformer = StandardScaler()

    # Transformations pour les colonnes catégorielles
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # ColumnTransformer : applique le bon traitement à chaque type de colonne
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Modèle de base : régression logistique
    model = LogisticRegression(max_iter=1000)

    # Pipeline complet = preprocessing + modèle
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf, numeric_features, categorical_features
