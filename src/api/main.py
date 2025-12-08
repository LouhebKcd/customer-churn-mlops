# src/api/main.py
import joblib
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# 1) Charger le modèle au démarrage de l'API
MODEL_PATH = "models/churn_model_tuned.joblib"
model = joblib.load(MODEL_PATH)

# Seuil par défaut basé sur ton analyse de thresholds (~bon compromis recall/precision)
DEFAULT_THRESHOLD = 0.35

app = FastAPI(
    title="Churn Prediction API",
    description="API pour prédire le churn client à partir d'un modèle ML entraîné",
    version="1.1.0",
)
# Servir le frontend depuis /app/frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


# 2) Schéma d'entrée (features du client)
class CustomerFeatures(BaseModel):
    customerID: Optional[str] = None  # optionnel, juste pour info
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def read_root():
    """
    Endpoint de healthcheck : permet de vérifier que l'API tourne.
    """
    return {"status": "ok", "message": "Churn API is running"}


@app.get("/model-info")
def model_info():
    """
    Donne des infos simples sur le modèle servi par l'API.
    Utile pour la doc et le debug.
    """
    return {
        "model_type": "LogisticRegression in sklearn Pipeline",
        "threshold_default": DEFAULT_THRESHOLD,
        "version": "1.1.0",
        "description": (
            "Modèle de churn client (Telco), entraîné avec préprocessing "
            "(StandardScaler + OneHotEncoder) et tuning de C."
        ),
    }


@app.post("/predict")
def predict_churn(
    features: CustomerFeatures,
    threshold: float = Query(
        DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Seuil de décision pour prédire le churn (par défaut 0.35)",
    ),
):
    """
    Prend les features d'un client et renvoie:
    - la probabilité de churn
    - la prédiction binaire (0 / 1) selon un seuil configurable
    """

    # 1) Convertir les features en DataFrame
    data = pd.DataFrame([features.dict()])

    # 2) Obtenir la probabilité de churn (classe positive)
    proba = model.predict_proba(data)[:, 1][0]

    # 3) Appliquer le seuil
    pred = int(proba >= threshold)

    return {
        "churn_probability": float(proba),
        "churn_prediction": pred,
        "threshold_used": threshold,
    }
