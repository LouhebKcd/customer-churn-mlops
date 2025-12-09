BEGIN README

# 📊 Churn Prediction – End-to-End MLOps Project  
### 🔥 Machine Learning • FastAPI • Docker • Frontend • End-to-End Deployment

Ce projet présente un pipeline complet de Data Science & MLOps permettant de prédire le churn client dans un contexte Telco.

Il inclut :
- Préparation & nettoyage des données
- Modélisation (Logistic Regression + Random Forest)
- Hyperparameter tuning (GridSearchCV)
- Optimisation du seuil décisionnel (business-driven threshold tuning)
- API REST en FastAPI
- Interface frontend HTML/CSS/JS
- Dockerisation complète
- Architecture professionnelle et reproductible

---

# 🧱 Architecture du projet

customer-churn-mlops/
│
├── data/
│ ├── raw/ # Données brutes
│ ├── processed/ # Données nettoyées + split train/test
│
├── models/
│ ├── churn_model_tuned.joblib # Modèle final sauvegardé
│
├── src/
│ ├── data/
│ │ └── make_dataset.py # Nettoyage + preprocessing + split
│ ├── models/
│ │ ├── train_model.py # LogReg baseline
│ │ ├── train_model_tuned.py # LogReg + GridSearch tuning
│ │ ├── train_model_rf.py # Random Forest + GridSearch
│ │ ├── evaluate_thresholds.py # Optimisation du seuil
│ ├── api/
│ └── main.py # API FastAPI
│
├── frontend/
│ └── index.html # Interface web
│
├── reports/
│ └── figures/ # Courbes ROC, PR, etc.
│
├── requirements.txt
├── Dockerfile
├── README.md


---

# 📥 1. Installation & Environnement

## 🔧 Cloner le projet

git clone https://github.com/

<ton-user>/customer-churn-mlops.git
cd customer-churn-mlops


## 🐍 Créer un environnement Python

python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows


## 📦 Installer les dépendances

pip install -r requirements.txt


---

# 🧼 2. Préparation des données

Le dataset brut (Telco Customer Churn) est placé dans :

data/raw/churn.csv


Pour nettoyer les données et créer les fichiers train/test :

python src/data/make_dataset.py


✔ Conversion des colonnes  
✔ Nettoyage des valeurs manquantes  
✔ Standardisation des types  
✔ Train/test split  
✔ Sauvegarde dans `data/processed/`

---

# 🤖 3. Entraînement des modèles

## Logistic Regression baseline

python src/models/train_model.py


## Logistic Regression + GridSearchCV

python src/models/train_model_tuned.py


### Résultats clés
- Accuracy : ~0.81  
- ROC-AUC : ~0.84  
- Recall churn : ~0.56  
- Modèle simple, efficace et interprétable

## Random Forest (modèle de comparaison)

python src/models/train_model_rf.py


---

# 🎯 4. Optimisation du seuil décisionnel

python src/models/evaluate_thresholds.py


| Threshold | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 0.50 | 0.657 | 0.559 | 0.604 |
| 0.35 | 0.546 | 0.711 | **0.618** |
| 0.25 | 0.498 | **0.813** | 0.618 |

➡️ Le seuil **0.35** maximise la détection de churners (objectif business).  
➡️ Utilisé comme **seuil par défaut dans l’API**.

---

# 🌐 5. API FastAPI

L’API expose les endpoints suivants :

### GET `/`
Health check.

### GET `/model-info`
Infos sur le modèle servi.

### POST `/predict?threshold=0.35`
Retourne :
- la probabilité de churn  
- la prédiction (0 ou 1)  
- le seuil utilisé  

### Lancer l’API :

uvicorn src.api.main:app --reload


Docs interactives :

http://127.0.0.1:8000/docs


---

# 🎨 6. Interface Web (Frontend)

L’interface web permet de :
- renseigner le profil client
- ajuster le seuil via un slider
- visualiser la probabilité + un badge (risque élevé / faible)
- consommer directement l’API

Accessible via :

http://127.0.0.1:8000/frontend


---

# 🐳 7. Dockerisation

## Construire l’image Docker

docker build -t churn-mlops .


## Lancer le conteneur

docker run -p 8000:8000 churn-mlops


Ensuite :
- API → http://localhost:8000  
- Frontend → http://localhost:8000/frontend  

✔ Fonctionne sans Python local  
✔ Déployable sur AWS / Azure / GCP / Render / Railway

---

# 📈 8. Résultats & Insights

### Pourquoi garder la Logistic Regression ?
- Meilleur recall sur churn  
- AUC élevé (~0.84)  
- Modèle léger → idéal pour API / production  
- Interprétable pour les équipes métier  

### Impact business :
- Détection de 60%+ des churners  
- Aide à cibler les clients à risque  
- Améliore les campagnes de rétention

---

# 🚀 9. Améliorations futures
- Monitoring du modèle (MLFlow, Prometheus)
- SHAP values pour explicabilité
- CI/CD de retraining
- Tests unitaires + GitHub Actions
- Déploiement cloud (ECS, App Service, Render)

---

# 👨‍💻 Auteur
KACED Louheb
Master2 Data Science / Machine Learning  
GitHub : https://github.com/LouhebKcd  

---

END README
