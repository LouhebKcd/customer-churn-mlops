# 1) Image de base : Python léger
FROM python:3.11-slim

# 2) Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3) Copier le fichier des dépendances
COPY requirements.txt .

# 4) Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copier tout le projet dans le conteneur
COPY . .

# 6) Exposer le port de l'API
EXPOSE 8000

# 7) Commande de démarrage
#    On lance uvicorn sur ton app FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

